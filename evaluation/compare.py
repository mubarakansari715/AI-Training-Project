"""Naive RAG vs Advanced RAG on one question at a time.

Both pipelines index the same document and get the same question:

- Naive RAG: similarity search only (no query rewriting or decomposition, no
  keyword search, no reranking, no section awareness, no chart/table/footnote
  reading).
- Advanced RAG: the app as configured (.env), with all of those.

Each answer is saved with what the "Use case results" page shows side by
side: the judge's verdicts, retrieval and grounding scores (see
evaluation.metrics), latency, token usage, the steps the pipeline took and
the passages the model was given. The same chunking and answer model are
used on both sides, so only the RAG techniques differ.

Examples:
    python -m evaluation.compare                      # main questions, all use cases
    python -m evaluation.compare --use-case gep --all-questions
    python -m evaluation.compare --full               # the full hand-checked sets (42 questions)

Writes evaluation/reports/<use case>_compare.json. Answers come only from
the pinned model (no fallback provider), like evaluation.run.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace

from app.rag.citations import cited_pages
from app.rag.generator import FALLBACK_MESSAGE
from evaluation.harness import (
    PROJECT_ROOT,
    EvalConfig,
    Throttle,
    build_pipeline,
    first_hit_rank,
    grade_answer,
    load_questions,
)
from evaluation.metrics import context_precision, estimate_cost, grade_grounding

USE_CASES = {
    "gpo": "evaluation/questions/gpo.json",
    "gep": "evaluation/questions/gep.json",
    "imac": "evaluation/questions/imac.json",
}

# label, what it does, EvalConfig overrides
SIDES = {
    "before": ("Naive RAG", "similarity search only",
               {"reranker_model": "", "section_aware": False, "read_visuals": False, "rewrite_queries": False,
                "hybrid_search": False, "decompose_queries": False, "chunk_strategy": "recursive",
                "layout_tables": "off"}),
    "after": ("Advanced RAG", "query rewriting and decomposition, hybrid BM25 + vector search, reranking, "
              "context compression, section-aware search, charts and footnotes", {}),
}

_ANSWER_ATTEMPTS = 3
_ANSWER_RETRY_SECONDS = 60.0


class PipelineFailed(RuntimeError):
    """The answer model gave no answer after its retries (rate limit, quota, outage)."""


def side_config(side: str) -> EvalConfig:
    return replace(EvalConfig(), **SIDES[side][2])


def question_options(questions_file: dict) -> list[dict]:
    """Main document questions (the task spec's samples) first, then related ones."""
    main = [dict(q, group="main") for q in questions_file["questions"] if q.get("sample")]
    related = [dict(q, group="related") for q in questions_file["questions"] if not q.get("sample")]
    return main + related


def describe_steps(pipeline, question: str, search_query: str, chunks: list, stats: dict,
                   sub_queries: list[str] | None = None) -> list[dict]:
    """The steps the pipeline took for one question, from what it recorded."""
    retriever = pipeline.retriever
    steps = []
    if search_query.strip() != question.strip():
        steps.append({"kind": "Query", "text": f'LLM rewrote the question for search: "{search_query}"'})
    if sub_queries:
        parts = "; ".join(f'"{q}"' for q in sub_queries)
        steps.append({"kind": "Query",
                      "text": f"LLM split the question into {len(sub_queries)} sub-questions, each searched "
                              f"on its own: {parts}"})
    named = retriever._section_filter(search_query)[0] if retriever.section_aware else []
    if named:
        sections = "; ".join(" / ".join(group) for group in named)
        steps.append({"kind": "Retrieve", "text": f"Searched only the section(s) the question names: {sections}"})
    count = pipeline.vector_store.count()
    kind = "hybrid" if retriever.keyword_index else "similarity"
    searches = (f"{stats.get('searches', 1)} {kind} searches" if stats.get("searches", 1) > 1
                else f"{kind.capitalize()} search")
    if retriever.keyword_index:
        steps.append({"kind": "Retrieve",
                      "text": f"Keyword search (BM25) merged with similarity search: "
                              f"{stats.get('keyword_only', 0)} candidate(s) found only by exact words"})
    if retriever.reranker:
        steps.append({"kind": "Retrieve",
                      "text": f"{searches} over {count} chunks: {stats.get('candidates', 0)} candidates above the "
                              f"similarity threshold (of {retriever.rerank_candidates} fetched per search)"})
        steps.append({"kind": "Post-process",
                      "text": f"Cross-encoder reranked the candidates and kept the top {stats.get('kept', 0)}"})
        if stats.get("compressed"):
            steps.append({"kind": "Post-process",
                          "text": f"Context compression dropped {stats['compressed']} passage(s) scored far below "
                                  "the best one"})
    else:
        steps.append({"kind": "Retrieve",
                      "text": f"{searches} over {count} chunks: kept the top {stats.get('kept', 0)} "
                              f"above the similarity threshold"})
    if stats.get("whole_pages"):
        steps.append({"kind": "Post-process", "text": "Swapped the best hits for their whole pages (up to 2)"})
    visuals = sum("(read from the page image)" in c.text for c in chunks)
    if visuals:
        steps.append({"kind": "Post-process",
                      "text": f"{visuals} passage(s) read from page images (charts, tables, call-outs)"})
    if stats.get("footnotes"):
        steps.append({"kind": "Post-process",
                      "text": f"{stats['footnotes']} footnote(s) attached to the passages citing them"})
    steps.append({"kind": "Generate",
                  "text": f"Grounded prompt with {len(chunks)} numbered passages: cite them, or say not found"})
    return steps


def _effect(retriever, search: str | list[str], expected: list[int], rank_with: int | None,
            precision_with: float | None):
    """Rank and precision for `search` (a query, or the sub-questions of a
    decomposed one) on this retriever, against the 'with' values."""
    chunks = retriever.retrieve_many(search) if isinstance(search, list) else retriever.retrieve(search)
    passages = [{"page": c.page, "expected": c.covers(expected)} for c in chunks]
    return {"rank_without": first_hit_rank(chunks, expected), "rank_with": rank_with,
            "precision_without": context_precision(passages, expected), "precision_with": precision_with}


def run_side(pipeline, question: dict, side: str, config: EvalConfig, model: str, judge_model: str,
             throttle: Throttle) -> dict:
    """One question through one pipeline, with everything the comparison shows."""
    label, technique, _ = SIDES[side]
    text = question["question"]
    expected = question.get("expected_pages") or []
    usage_start = len(pipeline.generator.usage)
    for attempt in range(1, _ANSWER_ATTEMPTS + 1):
        throttle.wait(calls=1 + pipeline.rewrite_queries + pipeline.decompose_queries)
        started = time.perf_counter()
        result = pipeline.ask(text, strict_mode=True, model_name=None, temperature=0.0,
                              groq_model_name=model, auto_mode=False)
        latency = time.perf_counter() - started
        if result.answer != FALLBACK_MESSAGE:
            break
        if attempt == _ANSWER_ATTEMPTS:
            raise PipelineFailed(f"{label}: the answer model ({model}) gave no answer after {_ANSWER_ATTEMPTS} "
                                 "tries. Groq's rate limit or daily quota is the usual cause; try again later.")
        time.sleep(_ANSWER_RETRY_SECONDS)
    usage = pipeline.generator.usage[usage_start:]

    # The same search the answer was built from (retrieval is deterministic);
    # `result.sources` only keeps the passages the answer cites.
    search_query = result.search_query or text
    sub_queries = result.sub_queries
    retriever = pipeline.retriever
    started = time.perf_counter()
    chunks = retriever.retrieve_many(sub_queries) if sub_queries else retriever.retrieve(search_query)
    retrieval_latency = time.perf_counter() - started
    stats = dict(retriever.last_stats)
    passages = [{"rank": i, "page": c.page, "section": c.section, "text": c.text, "expected": c.covers(expected)}
                for i, c in enumerate(chunks, start=1)]
    rank = first_hit_rank(chunks, expected)
    precision = context_precision(passages, expected)

    # What each technique did on its own, by searching again without it (no LLM calls).
    rewritten = search_query.strip() != text.strip()
    rewrite_effect = _effect(retriever, text, expected, rank, precision) if rewritten and expected else None
    decompose_effect = _effect(retriever, search_query, expected, rank, precision) if sub_queries and expected else None
    searched = sub_queries or search_query
    rerank_effect = keyword_effect = None
    if retriever.reranker and expected:
        reranker, retriever.reranker = retriever.reranker, None
        try:
            rerank_effect = _effect(retriever, searched, expected, rank, precision)
        finally:
            retriever.reranker = reranker
    if retriever.keyword_index and expected:
        keyword_index, retriever.keyword_index = retriever.keyword_index, None
        try:
            keyword_effect = _effect(retriever, searched, expected, rank, precision)
        finally:
            retriever.keyword_index = keyword_index

    throttle.wait()
    verdict = grade_answer(pipeline.generator, text, question["reference_answer"], result.answer, judge_model)
    throttle.wait()
    grounding = grade_grounding(pipeline.generator, text, question["reference_answer"], result.answer, passages,
                                judge_model)
    correct, hit = verdict["correct"], (rank is not None if expected else None)
    return {
        "label": label, "technique": technique, "config": config.label, "model": model,
        "answer": result.answer,
        # PASS needs both: the judge found the answer correct and complete, and
        # (when the document has the answer) its page was among the passages.
        "passed": bool(correct and hit is not False),
        "correct": correct, "judge_reason": verdict["reason"],
        "hit": hit, "rank": rank,
        "citation_ok": bool(cited_pages(result.answer) & set(expected)) if expected else None,
        "faithfulness": grounding["faithfulness"], "answer_relevance": grounding["answer_relevance"],
        "context_recall": grounding["context_recall"], "context_precision": precision,
        "unsupported": grounding["unsupported"], "relevance_reason": grounding["relevance_reason"],
        "grounding_error": grounding["error"],
        "passages_count": len(passages),
        "irrelevant_passages": sum(not p["expected"] for p in passages) if expected else None,
        "latency_s": round(latency, 3), "retrieval_latency_s": round(retrieval_latency, 3),
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage) if usage else None,
        "completion_tokens": sum(u["completion_tokens"] for u in usage) if usage else None,
        "cost_usd": estimate_cost(usage),
        "original_query": text, "search_query": search_query, "rewritten": rewritten,
        "sub_queries": sub_queries, "queries_generated": len(sub_queries) or 1,
        "searches": stats.get("searches", 0), "candidates": stats.get("candidates", 0),
        "kept": stats.get("kept", 0), "reranked": bool(retriever.reranker),
        "rewrite_effect": rewrite_effect, "rerank_effect": rerank_effect,
        "decompose_effect": decompose_effect, "keyword_effect": keyword_effect,
        "steps": describe_steps(pipeline, text, search_query, chunks, stats, sub_queries),
        "passages": passages,
    }


def compare_question(pipelines: dict, question: dict, model: str, judge_model: str, throttle: Throttle) -> dict:
    """Both pipelines on one question; `pipelines` maps "before"/"after" to (pipeline, config)."""
    sides = {side: run_side(pipeline, question, side, config, model, judge_model, throttle)
             for side, (pipeline, config) in pipelines.items()}
    return make_row(question, sides)


def make_row(question: dict, sides: dict) -> dict:
    """One saved comparison: the question, its reference, and both sides."""
    return {"id": question["id"], "question": question["question"], "group": question.get("group", "main"),
            "reference_answer": question["reference_answer"], "expected_pages": question.get("expected_pages") or [],
            "generated_at": time.strftime("%Y-%m-%d %H:%M"), **sides}


def resumable_rows(path, model: str, judge_model: str, configs: dict[str, str]) -> dict:
    """Rows of an earlier run of the same models and pipeline configs
    (`configs`: side -> EvalConfig.label), by question id. A run made before
    a technique changed is not reused."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if (saved.get("answer_model"), saved.get("judge_model"), saved.get("configs")) != (model, judge_model, configs):
        return {}
    return {r["id"]: r for r in saved.get("rows", [])}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-case", nargs="+", choices=sorted(USE_CASES), default=list(USE_CASES))
    parser.add_argument("--all-questions", action="store_true", help="Also the related questions, not only the main ones")
    parser.add_argument("--full", action="store_true",
                        help="Every question of the full hand-checked set (<use case>_full.json), saved as "
                             "<use case>_full_compare.json; the results page keeps the main questions")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Pinned answer model (Groq)")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--calls-per-minute", type=int, default=4,
                        help="LLM call rate limit (Groq allows 8k tokens/min on the free tier)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    for key in args.use_case:
        questions_path = USE_CASES[key].replace(".json", "_full.json") if args.full else USE_CASES[key]
        questions_file = load_questions(questions_path)
        questions = [q for q in question_options(questions_file)
                     if args.full or args.all_questions or q["group"] == "main"]
        pipelines = {}
        for side in SIDES:
            config = side_config(side)
            print(f"[{key}] {SIDES[side][0]}: indexing with {config.label} ...", flush=True)
            pipelines[side] = (build_pipeline(questions_file["document"], config)[0], config)
        throttle = Throttle(args.calls_per_minute)
        path = PROJECT_ROOT / "evaluation" / "reports" / f"{key}{'_full' if args.full else ''}_compare.json"
        # Saved after every question, and questions already saved with these
        # models and configs are skipped, so a run stopped by Groq's quota resumes.
        configs = {side: config.label for side, (_, config) in pipelines.items()}
        done = resumable_rows(path, args.model, args.judge_model, configs)
        rows = [done[q["id"]] for q in questions if q["id"] in done]
        for q in questions:
            if q["id"] in done:
                continue
            row = compare_question(pipelines, q, args.model, args.judge_model, throttle)
            rows.append(row)
            path.write_text(json.dumps({"use_case": key, "document": questions_file["document"],
                                        "answer_model": args.model, "judge_model": args.judge_model,
                                        "configs": configs, "rows": rows},
                                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  {q['id']}: Naive {'PASS' if row['before']['passed'] else 'FAIL'}, "
                  f"Advanced {'PASS' if row['after']['passed'] else 'FAIL'}", flush=True)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
