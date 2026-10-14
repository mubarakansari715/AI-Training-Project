"""Reusable, document-agnostic RAG evaluation harness.

Works for any document the app can ingest (PDF, DOCX, TXT/MD, video,
audio) and any questions file. Nothing in here knows about a specific
document: the questions file only holds ground truth — each question,
a plain-language reference answer, and (optionally) the pages that state
the answer.

What it measures, always through the app's real `RAGPipeline` (the same
parsing, chunking, embeddings, retrieval, query rewriting, small-talk
handling and strict-mode logic users get in the chat):

- retrieval (no LLM calls): hit@k — did a chunk from an expected page come
  back? — and MRR — how high was the first one ranked?
- answers: an LLM judge (pinned model, temperature 0) compares each answer
  with the reference answer, so no per-question keyword lists exist that
  could be tuned until a test passes; plus whether the answer cites one of
  the expected pages.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config.settings import settings
from app.rag.citations import cited_pages
from app.rag.generator import AnswerGenerator
from app.rag.retriever import RetrievedChunk
from app.services.rag_service import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalConfig:
    """One pipeline configuration to evaluate. Defaults = the app's settings."""

    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap
    embedding_model: str = settings.embedding_model
    embedding_query_prefix: str = settings.embedding_query_prefix
    top_k: int = settings.top_k
    score_threshold: float = settings.similarity_threshold
    reranker_model: str = settings.reranker_model  # "" = no reranking
    rerank_candidates: int = settings.rerank_candidates
    section_aware: bool = settings.section_aware
    read_visuals: bool = settings.visual_reading_max_pages > 0

    @property
    def label(self) -> str:
        model = self.embedding_model.split("/")[-1]
        rerank = f"rerank {self.reranker_model.split('/')[-1]}@{self.rerank_candidates}" if self.reranker_model else "no rerank"
        sections = "sections" if self.section_aware else "no sections"
        visuals = " | visuals" if self.read_visuals else ""
        return f"{model} | {self.chunk_size}/{self.chunk_overlap} | k={self.top_k} | th={self.score_threshold} | {rerank} | {sections}{visuals}"


@dataclass
class _FileUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""

    path: Path
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def load_questions(path: str | Path) -> dict:
    """Load a questions file: {"document": ..., "questions": [{id, question,
    reference_answer, expected_pages?}, ...]}."""
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        data = json.load(f)
    for q in data["questions"]:
        missing = {"id", "question", "reference_answer"} - q.keys()
        if missing:
            raise ValueError(f"Question {q.get('id', '?')} is missing {sorted(missing)}")
    return data


def build_pipeline(document: str | Path, config: EvalConfig | None = None) -> tuple[RAGPipeline, int]:
    """Index `document` into a fresh, isolated pipeline; return it and its chunk count."""
    config = config or EvalConfig()
    pipeline = RAGPipeline(
        persist_dir=tempfile.mkdtemp(prefix="rag_eval_"),
        collection_name="eval",
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model,
        embedding_query_prefix=config.embedding_query_prefix,
        top_k=config.top_k,
        score_threshold=config.score_threshold,
        reranker_model=config.reranker_model,
        rerank_candidates=config.rerank_candidates,
        section_aware=config.section_aware,
        read_visuals=config.read_visuals,
    )
    num_chunks, errors = pipeline.index_files([_FileUpload(PROJECT_ROOT / document)])
    if errors or not num_chunks:
        raise ValueError(f"Could not index {document}: {errors or 'no text extracted'}")
    return pipeline, num_chunks


def set_retrieval(pipeline: RAGPipeline, top_k: int, score_threshold: float) -> None:
    """Change retrieval settings on an already-indexed pipeline (no re-indexing)."""
    pipeline.retriever.top_k = top_k
    pipeline.retriever.score_threshold = score_threshold


# ----------------------------------------------------------------- scoring


def first_hit_rank(chunks: list[RetrievedChunk], expected_pages: list[int]) -> int | None:
    """1-based rank of the first chunk from an expected page, else None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.page in expected_pages:
            return rank
    return None


JUDGE_PROMPT = """You are grading the answer of a document question-answering system.

Question: {question}

Reference answer (ground truth taken from the document): {reference}

System answer: {answer}

Grade the system answer:
- "correct" if it states the key facts of the reference answer. Wording, number
  formatting and rounding may differ when the reference shows the document
  states the figure that way. Extra detail is fine if it doesn't contradict
  the reference.
- "incorrect" if a key fact is missing, wrong or contradicted, or if it says
  the information is unavailable/unspecified while the reference provides it.
- If the reference says the document does not contain the answer, "correct"
  only if the system declines to answer instead of giving one.

Reply with JSON only: {{"verdict": "correct" or "incorrect", "reason": "<one sentence>"}}"""


def grade_answer(
    generator: AnswerGenerator, question: str, reference: str, answer: str, model_name: str | None = None
) -> dict:
    """Ask the judge LLM whether `answer` matches `reference`."""
    output = generator.complete(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer),
        model_name=model_name,
        temperature=0.0,
    )
    return parse_verdict(output)


def parse_verdict(output: str) -> dict:
    """Parse the judge's JSON reply; anything unparseable counts as incorrect."""
    match = re.search(r"\{.*\}", output or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in {"correct", "incorrect"}:
        return {"correct": False, "reason": f"Unparseable judge output: {(output or '')[:120]!r}"}
    return {"correct": verdict == "correct", "reason": str(data.get("reason", ""))}


class Throttle:
    """Spaces LLM calls out to stay under a provider's per-minute limit
    (Gemini's free tier allows 15/min; going over makes pinned calls
    silently fall back to another provider)."""

    def __init__(self, calls_per_minute: int | None):
        self.interval = 60.0 / calls_per_minute if calls_per_minute else 0.0
        self.free_at = 0.0

    def wait(self, calls: int = 1) -> None:
        now = time.monotonic()
        start = max(now, self.free_at)
        if start > now:
            time.sleep(start - now)
        self.free_at = start + self.interval * calls


# ------------------------------------------------------------- experiments


def evaluate_retrieval(pipeline: RAGPipeline, questions: list[dict]) -> list[dict]:
    """Retrieval results for every question that has expected pages."""
    rows = []
    for q in questions:
        if not q.get("expected_pages"):
            continue
        chunks = pipeline.retriever.retrieve(q["question"])
        rank = first_hit_rank(chunks, q["expected_pages"])
        rows.append({"id": q["id"], "hit": rank is not None, "rank": rank,
                     "retrieved_pages": [c.page for c in chunks]})
    return rows


def summarize_retrieval(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 3),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
        "avg_chunks": round(sum(len(r["retrieved_pages"]) for r in rows) / n, 1),
    }


def evaluate_answers(
    pipeline: RAGPipeline,
    questions: list[dict],
    model_name: str | None = None,
    judge_model: str | None = None,
    strict_mode: bool = True,
    calls_per_minute: int | None = None,
    groq_model_name: str | None = None,
    auto_mode: bool | None = False,
) -> list[dict]:
    """Ask every question through `pipeline.ask` and grade it with the judge.

    Pin the answering model so runs are comparable: `model_name` for a
    Gemini model, `groq_model_name` for a Groq model (or "auto"). Auto
    (agent) mode is off unless `auto_mode` is True. Answers use
    temperature 0. The judge defaults to the configured Gemini model — a
    different model family from the Groq answerer, so it isn't grading itself.
    """
    throttle = Throttle(calls_per_minute)
    judge_model = judge_model or settings.gemini_model
    rows = []
    for q in questions:
        throttle.wait(calls=3 if auto_mode else 2)  # agent turns / rewrite + answer
        result = pipeline.ask(q["question"], strict_mode=strict_mode, model_name=model_name, temperature=0.0,
                              groq_model_name=groq_model_name, auto_mode=auto_mode)
        throttle.wait()
        grade = grade_answer(pipeline.generator, q["question"], q["reference_answer"], result.answer, judge_model)
        expected = set(q.get("expected_pages") or [])
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "search_query": result.search_query,
            "searches": result.searches,
            "model": result.model,
            "retrieved_pages": [c.page for c in result.sources],
            "answer": result.answer,
            "correct": grade["correct"],
            "judge_reason": grade["reason"],
            "citation_ok": bool(cited_pages(result.answer) & expected) if expected else None,
        })
    return rows


def summarize_answers(rows: list[dict]) -> dict:
    n = len(rows) or 1
    cited = [r["citation_ok"] for r in rows if r["citation_ok"] is not None]
    return {
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "citation_accuracy": round(sum(cited) / len(cited), 3) if cited else None,
    }


def write_report(
    out_dir: str | Path,
    name: str,
    document: str,
    config: EvalConfig,
    retrieval_rows: list[dict],
    answer_rows: list[dict] | None,
    model_name: str | None,
) -> Path:
    """Write <name>.json (full detail) and <name>.md (readable summary)."""
    out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "document": document,
        "config": asdict(config),
        "answer_model": model_name,
        "retrieval": {"summary": summarize_retrieval(retrieval_rows), "rows": retrieval_rows},
        "answers": {"summary": summarize_answers(answer_rows), "rows": answer_rows} if answer_rows else None,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [f"# Evaluation: {Path(document).name}", "", f"- Config: `{config.label}`",
             f"- Answer/judge model: `{model_name or 'app default'}`", "",
             "## Retrieval", "", f"{summarize_retrieval(retrieval_rows)}", ""]
    if answer_rows:
        lines += ["## Answers", "", f"{summarize_answers(answer_rows)}", "",
                  "| id | correct | cites page | judge reason |", "|---|---|---|---|"]
        lines += [f"| {r['id']} | {'✅' if r['correct'] else '❌'} | {r['citation_ok']} | "
                  f"{r['judge_reason'].replace('|', '/')} |" for r in answer_rows]
    md_path = out_dir / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path
