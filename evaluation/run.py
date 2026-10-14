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
    parser.add_argument("--reranker", default=settings.reranker_model,
                        help='Cross-encoder reranker model ("" or "none" to disable)')
    parser.add_argument("--rerank-candidates", type=int, default=settings.rerank_candidates)
    parser.add_argument("--section-aware", action=argparse.BooleanOptionalAction, default=settings.section_aware,
                        help="Detect document sections and search named sections only (default: SECTION_AWARE)")
    parser.add_argument("--model", default=settings.groq_model,
                        help='Pinned answer model: a Groq model, "auto", or a gemini-* model (default: GROQ_MODEL)')
    parser.add_argument("--auto-mode", action=argparse.BooleanOptionalAction, default=False,
                        help="Answer in auto (agent) mode (default: off, the standard pipeline)")
    parser.add_argument("--judge-model", default=settings.gemini_model, help="Judge model (default: GEMINI_MODEL)")
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
    reranker = "" if args.reranker.lower() in {"", "none", "off"} else args.reranker
    config = EvalConfig(args.chunk_size, args.chunk_overlap, args.embedding_model,
                        args.embedding_query_prefix, args.top_k, args.threshold,
                        reranker, args.rerank_candidates, args.section_aware)

    print(f"Indexing {document} with {config.label} ...")
    pipeline, num_chunks = build_pipeline(document, config)
    print(f"  {num_chunks} chunks")

    retrieval_rows = evaluate_retrieval(pipeline, questions)
    print("Retrieval:", summarize_retrieval(retrieval_rows))

    answer_rows = None
    if not args.retrieval_only:
        print(f"Answering {len(questions)} questions with {args.model}, auto mode {'on' if args.auto_mode else 'off'} "
              f"(judge: {args.judge_model}) ...")
        gemini = args.model.startswith("gemini")
        answer_rows = evaluate_answers(pipeline, questions, judge_model=args.judge_model,
                                       model_name=args.model if gemini else None,
                                       groq_model_name=None if gemini else args.model,
                                       auto_mode=args.auto_mode, calls_per_minute=args.calls_per_minute)
        print("Answers:", summarize_answers(answer_rows))
        for r in answer_rows:
            print(f"  {'PASS' if r['correct'] else 'FAIL'}  {r['id']}: {r['judge_reason']}")

    name = args.name or Path(args.questions).stem
    report = write_report(args.out_dir, name, document, config, retrieval_rows, answer_rows, args.model)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
