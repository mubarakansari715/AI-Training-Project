"""Before vs after, on the task spec's sample questions only.

For each use case, the sample questions are asked twice: through the app
without that use case's improvement ("before") and with it ("after").
Each answer is saved with what the "Use case results" page shows side by
side: the judge's verdict, where the answer passage ranked, the steps the
pipeline took, and the passages the model was given.

Examples:
    python -m evaluation.compare                      # all three use cases
    python -m evaluation.compare --use-case gep imac

Writes evaluation/reports/<use case>_compare.json. Answers come only from
the pinned model (no fallback provider), like evaluation.run.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace

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

# What "before" leaves out is each use case's own improvement; "after" is
# the app as it is now.
USE_CASES = {
    "gpo": {
        "questions": "evaluation/questions/gpo.json",
        "before": ("Before: similarity search only", {"reranker_model": ""}),
        "after": ("After: reranked", {}),
    },
    "gep": {
        "questions": "evaluation/questions/gep.json",
        "before": ("Before: no section awareness", {"section_aware": False}),
        "after": ("After: section-aware", {"section_aware": True}),
    },
    "imac": {
        "questions": "evaluation/questions/imac.json",
        "before": ("Before: text only", {"read_visuals": False}),
        "after": ("After: charts, tables and footnotes", {"read_visuals": True}),
    },
}

_ANSWER_ATTEMPTS = 3
_ANSWER_RETRY_SECONDS = 60.0


def describe_steps(pipeline, question: str, search_query: str, chunks: list) -> list[dict]:
    """The steps the standard pipeline took for one question, in order."""
    retriever = pipeline.retriever
    steps = []
    if search_query.strip() != question.strip():
        steps.append({"kind": "Query", "text": f'Rewritten for search: "{search_query}"'})
    named = retriever._section_filter(search_query)[0] if retriever.section_aware else []
    if named:
        sections = "; ".join(" / ".join(group) for group in named)
        steps.append({"kind": "Retrieve", "text": f"Searched only the section(s) the question names: {sections}"})
    count = pipeline.vector_store.count()
    if retriever.reranker:
        search = (f"Similarity search over {count} chunks: shortlist of {retriever.rerank_candidates}, "
                  f"then a cross-encoder reranked it and kept the top {retriever.top_k}")
    else:
        search = f"Similarity search over {count} chunks: top {retriever.top_k}"
    steps.append({"kind": "Retrieve", "text": search})
    if len(named) == 1:
        steps.append({"kind": "Post-process", "text": "Swapped the best hits for their whole pages (up to 2)"})
    visuals = sum("(read from the page image)" in c.text for c in chunks)
    if visuals:
        steps.append({"kind": "Post-process",
                      "text": f"{visuals} passage(s) read from page images (charts, tables, call-outs)"})
    footnotes = sum(c.text.startswith("[Footnote ") for c in chunks)
    if footnotes:
        steps.append({"kind": "Post-process", "text": f"{footnotes} footnote(s) attached to the passages citing them"})
    steps.append({"kind": "Generate",
                  "text": f"Grounded prompt with {len(chunks)} numbered passages: cite them, or say not found"})
    return steps


def ask_side(pipeline, question: dict, label: str, config: EvalConfig, model: str, judge_model: str,
             throttle: Throttle) -> dict:
    """One question through one pipeline: answer, verdict, rank, steps, passages."""
    for attempt in range(1, _ANSWER_ATTEMPTS + 1):
        throttle.wait(calls=2)  # rewrite + answer
        result = pipeline.ask(question["question"], strict_mode=True, model_name=None, temperature=0.0,
                              groq_model_name=model, auto_mode=False)
        if result.answer != FALLBACK_MESSAGE:
            break
        if attempt == _ANSWER_ATTEMPTS:
            raise RuntimeError(f"The answer model failed {_ANSWER_ATTEMPTS} times on {question['id']!r} "
                               "(rate limit or daily quota?); stopping instead of saving a partial comparison.")
        time.sleep(_ANSWER_RETRY_SECONDS)
    # The same search the answer was built from (retrieval is deterministic):
    # `result.sources` only keeps the passages the answer cites.
    chunks = pipeline.retriever.retrieve(result.search_query or question["question"])
    expected = question.get("expected_pages") or []
    throttle.wait()
    grade = grade_answer(pipeline.generator, question["question"], question["reference_answer"], result.answer,
                         judge_model)
    return {
        "label": label,
        "config": config.label,
        "model": model,
        "answer": result.answer,
        "correct": grade["correct"],
        "judge_reason": grade["reason"],
        "rank": first_hit_rank(chunks, expected),
        "steps": describe_steps(pipeline, question["question"], result.search_query or question["question"], chunks),
        "passages": [
            {"rank": i, "page": c.page, "section": c.section, "text": c.text, "expected": c.page in expected}
            for i, c in enumerate(chunks, start=1)
        ],
    }


def compare_use_case(key: str, model: str, judge_model: str, calls_per_minute: int) -> dict:
    spec = USE_CASES[key]
    questions_file = load_questions(spec["questions"])
    samples = [q for q in questions_file["questions"] if q.get("sample")]
    throttle = Throttle(calls_per_minute)
    sides = {}
    for side in ("before", "after"):
        label, overrides = spec[side]
        config = replace(EvalConfig(), **overrides)
        print(f"[{key}] {label}: indexing with {config.label} ...", flush=True)
        pipeline, _ = build_pipeline(questions_file["document"], config)
        sides[side] = [ask_side(pipeline, q, label, config, model, judge_model, throttle) for q in samples]
    return {
        "use_case": key,
        "document": questions_file["document"],
        "answer_model": model,
        "judge_model": judge_model,
        "rows": [
            {"id": q["id"], "question": q["question"], "reference_answer": q["reference_answer"],
             "expected_pages": q.get("expected_pages") or [], "before": before, "after": after}
            for q, before, after in zip(samples, sides["before"], sides["after"])
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-case", nargs="+", choices=sorted(USE_CASES), default=list(USE_CASES))
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Pinned answer model (Groq)")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--calls-per-minute", type=int, default=4,
                        help="LLM call rate limit (Groq allows 8k tokens/min on the free tier)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    for key in args.use_case:
        result = compare_use_case(key, args.model, args.judge_model, args.calls_per_minute)
        path = PROJECT_ROOT / "evaluation" / "reports" / f"{key}_compare.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for row in result["rows"]:
            print(f"  {row['id']}: before {'PASS' if row['before']['correct'] else 'FAIL'}, "
                  f"after {'PASS' if row['after']['correct'] else 'FAIL'}")
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
