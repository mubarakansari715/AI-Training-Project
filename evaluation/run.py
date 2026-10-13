"""Evaluate the RAG pipeline on any document + questions file.

Examples:
    python -m evaluation.run --questions evaluation/questions/gpo.json
    python -m evaluation.run --questions my_questions.json --top-k 8 --chunk-size 600
    python -m evaluation.run --questions my_questions.json --retrieval-only

Every pipeline option defaults to the app's own settings (.env), so with no
flags this measures exactly what ALLY AI does. Reports go to
evaluation/reports/<name>.md and .json.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config.settings import settings
from evaluation.harness import (
    EvalConfig,
    build_pipeline,
    evaluate_answers,
    evaluate_retrieval,
    load_questions,
    summarize_answers,
    summarize_retrieval,
    write_report,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, help="Questions JSON file")
    parser.add_argument("--document", help="Override the document path in the questions file")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--embedding-query-prefix", default=settings.embedding_query_prefix)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--threshold", type=float, default=settings.similarity_threshold)
    parser.add_argument("--model", default=settings.gemini_model,
                        help="Pinned answer model (Gemini) — default: GEMINI_MODEL")
    parser.add_argument("--judge-model", help="Judge model (default: same as --model)")
    parser.add_argument("--calls-per-minute", type=int, default=12,
                        help="LLM call rate limit (Gemini free tier allows 15/min)")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip answering (no LLM calls)")
    parser.add_argument("--out-dir", default="evaluation/reports")
    parser.add_argument("--name", help="Report file name (default: questions file name)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    questions_file = load_questions(args.questions)
    document = args.document or questions_file["document"]
    questions = questions_file["questions"]
    config = EvalConfig(args.chunk_size, args.chunk_overlap, args.embedding_model,
                        args.embedding_query_prefix, args.top_k, args.threshold)

    print(f"Indexing {document} with {config.label} ...")
    pipeline, num_chunks = build_pipeline(document, config)
    print(f"  {num_chunks} chunks")

    retrieval_rows = evaluate_retrieval(pipeline, questions)
    print("Retrieval:", summarize_retrieval(retrieval_rows))

    answer_rows = None
    if not args.retrieval_only:
        print(f"Answering {len(questions)} questions with {args.model} (judge: {args.judge_model or args.model}) ...")
        answer_rows = evaluate_answers(pipeline, questions, model_name=args.model, judge_model=args.judge_model,
                                       calls_per_minute=args.calls_per_minute)
        print("Answers:", summarize_answers(answer_rows))
        for r in answer_rows:
            print(f"  {'PASS' if r['correct'] else 'FAIL'}  {r['id']}: {r['judge_reason']}")

    name = args.name or Path(args.questions).stem
    report = write_report(args.out_dir, name, document, config, retrieval_rows, answer_rows, args.model)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
