"""Generate an evaluation question set from a document, automatically.

Hand-written questions tend to come from passages the author already read,
which biases a test set toward what the system handles well. This samples
chunks spread evenly across the whole document (fixed seed, so the same
chunks are picked every run) and asks an LLM to write one factual question
answerable from each chunk alone, plus its reference answer. Expected pages
are the chunk's page plus every other page whose text contains all the
numbers in the answer, so restated facts aren't counted as misses.

Works for any document the app can ingest. Example:
    python -m evaluation.generate --document notebooks/GPO-Anual-report.pdf --n 25
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from app.config.settings import settings
from app.rag.generator import AnswerGenerator
from evaluation.harness import PROJECT_ROOT, Throttle, build_pipeline

QUESTION_PROMPT = """Below is an excerpt from a document. Write ONE question that a reader \
of the whole document might ask, which this excerpt answers with a specific fact \
(a number, date, name, amount or short phrase).

Rules:
- The question must make sense on its own, without seeing the excerpt (name the \
organisation, year or program it is about; never say "this excerpt" or "the text").
- The answer must come only from the excerpt, word for word for numbers.
- If the excerpt has no specific fact worth asking about (e.g. a table of contents, \
a caption, boilerplate), reply {{"skip": true}}.

Excerpt (page {page}):
\"\"\"{text}\"\"\"

Reply with JSON only: {{"question": "...", "answer": "..."}} or {{"skip": true}}"""

_MIN_CHUNK_CHARS = 250
_NUMBER_RE = re.compile(r"\d[\d,.]*\d|\d")


def _parse(output: str) -> dict | None:
    match = re.search(r"\{.*\}", output or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    if not data or data.get("skip") or not data.get("question") or not data.get("answer"):
        return None
    return data


_YEAR_RE = re.compile(r"(19|20)\d\d")


def _distinctive_numbers(answer: str) -> list[str]:
    """Numbers specific enough to identify a fact: 3+ digits and not a bare year
    ("20" or "2018" appear on nearly every page; "34,367" doesn't)."""
    return [n for n in _NUMBER_RE.findall(answer)
            if len(re.sub(r"\D", "", n)) >= 3 and not _YEAR_RE.fullmatch(n)]


def _pages_stating(answer: str, page_texts: dict[int, str]) -> list[int]:
    """Pages whose text contains every distinctive number in `answer` (none if it has none)."""
    numbers = _distinctive_numbers(answer)
    if not numbers:
        return []
    return [page for page, text in page_texts.items() if all(n in text for n in numbers)]


def sample_chunks(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """Pick up to `n` substantive chunks spread evenly across the document."""
    usable = [c for c in chunks if len(c["text"]) >= _MIN_CHUNK_CHARS]
    usable.sort(key=lambda c: (c["page"] or 0, c["chunk_id"]))
    if len(usable) <= n:
        return usable
    rng = random.Random(seed)
    stride = len(usable) / n
    # One random chunk from each of n equal slices of the document.
    return [usable[int(i * stride) + rng.randrange(max(1, int(stride)))] for i in range(n)]


def generate_questions(
    document: str, n: int, seed: int, model_name: str | None, calls_per_minute: int | None
) -> list[dict]:
    pipeline, _ = build_pipeline(document)
    chunks = pipeline.vector_store.all_chunks()
    page_texts: dict[int, str] = {}
    for c in chunks:
        if c["page"] is not None:
            page_texts[c["page"]] = page_texts.get(c["page"], "") + "\n" + c["text"]

    generator = AnswerGenerator()
    throttle = Throttle(calls_per_minute)
    questions = []
    # Over-sample: some chunks are skipped as having no concrete fact.
    for chunk in sample_chunks(chunks, n * 2, seed):
        if len(questions) >= n:
            break
        throttle.wait()
        output = generator.complete(
            QUESTION_PROMPT.format(page=chunk["page"], text=chunk["text"]),
            model_name=model_name, temperature=0.0,
        )
        item = _parse(output)
        if item is None:
            continue
        pages = sorted({chunk["page"], *_pages_stating(item["answer"], page_texts)} - {None})
        questions.append({
            "id": f"gen-{len(questions) + 1:02d}",
            "question": item["question"].strip(),
            "reference_answer": item["answer"].strip(),
            "expected_pages": pages,
            "source_chunk": chunk["text"][:300],
        })
    return questions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--document", required=True)
    parser.add_argument("--n", type=int, default=25, help="Number of questions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=settings.gemini_model)
    parser.add_argument("--calls-per-minute", type=int, default=12)
    parser.add_argument("--out", help="Output JSON (default: evaluation/questions/<doc>_generated.json)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    questions = generate_questions(args.document, args.n, args.seed, args.model, args.calls_per_minute)
    out = Path(args.out) if args.out else Path("evaluation/questions") / f"{Path(args.document).stem}_generated.json"
    payload = {
        "document": args.document,
        "notes": f"Generated by evaluation.generate (seed {args.seed}, model {args.model}): one question per "
                 "chunk sampled evenly across the document. Review before relying on it.",
        "questions": questions,
    }
    (PROJECT_ROOT / out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {out}")


if __name__ == "__main__":
    main()
