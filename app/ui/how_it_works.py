"""The "How it works" tab of the Use case results page: one screen with the
pipeline, every Advanced RAG field-guide technique (status, measured
result, setting, code) and the free stack. Headline numbers are read from
the saved full comparison reports (python -m evaluation.compare --full)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

_FULL_REPORTS = {
    "GPO": "evaluation/reports/gpo_full_compare.json",
    "World Bank GEP": "evaluation/reports/gep_full_compare.json",
    "iMac": "evaluation/reports/imac_full_compare.json",
}

# The two flows, as steps: drawn as a diagram here, and served as data to the
# React UI (ReactAI/server/api.py), so both show the same pipeline.
INDEX_STEPS = [
    ("PDF", "One document, any length"),
    ("Extract & clean text", "Per page; ligatures expanded, broken words rejoined"),
    ("Find sections", "From PDF bookmarks, or headings by font size"),
    ("Read charts & tables", "Vision model on page images, or tables from the page layout"),
    ("Link footnotes", "Each marker gets its note"),
    ("Chunk", "800 characters with a section label; tables and footnotes whole"),
    ("Store", "Chroma vectors + BM25 keyword index"),
]
ANSWER_STEPS = [
    ("Question", "As typed, or from the chat"),
    ("Rewrite", "Fix typos, resolve follow-ups"),
    ("Split", "Multi-part questions into sub-questions"),
    ("Route", "Search only the sections it names (rules, or LLM)"),
    ("Hybrid search", "Vectors + BM25, merged by rank fusion"),
    ("Rerank & compress", "Cross-encoder keeps the best; far-below-best dropped"),
    ("Expand", "Whole page of the best hits + referenced footnotes"),
    ("Answer", "From numbered passages, with page citations, or 'not found'"),
]


def _chain(prefix: str, steps: list[tuple[str, str]]) -> str:
    nodes = [f'{prefix}{i}["{title}<br/><small>{detail}</small>"]' for i, (title, detail) in enumerate(steps)]
    return "\n        ".join(" --> ".join(nodes[i:i + 2]) for i in range(len(nodes) - 1))


PIPELINE = f"""
flowchart LR
    subgraph Index["Indexing (once per document)"]
        direction TB
        {_chain("I", INDEX_STEPS)}
    end
    subgraph Ask["Answering (every question)"]
        direction TB
        {_chain("A", ANSWER_STEPS)}
    end
    I{len(INDEX_STEPS) - 1} --> A4
"""

# One row per field-guide technique: status is what the app does with it.
TECHNIQUES = [
    ("Baseline", "Extract text, fixed-size chunks, dense search, answer", "Naive RAG side",
     "The comparison baseline", "-", "evaluation/compare.py (SIDES)"),
    ("GPO", "Two-column reading order", "Not needed",
     "pypdf already reads GPO's 16 two-column pages a column at a time", "-", "-"),
    ("GPO", "Sentence chunking with sentence overlap", "Option",
     "No gain on these documents (MRR GPO 0.92 vs 0.94)", "CHUNK_STRATEGY=sentence", "app/rag/chunker.py"),
    ("GPO", "Hybrid search: dense + BM25, reciprocal rank fusion", "On",
     "Right page 88% -> 96%, MRR 0.77 -> 0.91 (25 GPO questions, no reranker)", "HYBRID_SEARCH",
     "app/rag/keyword_search.py"),
    ("GPO", "Rerank the top 20 with a cross-encoder", "On",
     "Right page 92% -> 100% on GPO", "RERANKER_MODEL", "app/rag/reranker.py"),
    ("GPO", "Numbered passages, citations, refuse when not found", "On",
     "Faithfulness 100% on the 9 main questions", "STRICT_DOCUMENT_MODE", "app/rag/prompt.py"),
    ("GEP", "Repair extracted text (ligatures, hyphenation)", "On",
     "~1,560 ligatures fixed; GEP MRR 0.96 -> 1.00", "always", "app/rag/text_cleaner.py"),
    ("GEP", "Chapter/region structure from fonts", "Fallback",
     "Bookmarks score 1.00 vs fonts 0.96; fonts used when a PDF has no bookmarks", "STRUCTURE_SOURCE",
     "app/rag/structure.py"),
    ("GEP", "Heading path on every chunk, embedded with the text", "On",
     "Right section 93% -> 100%", "SECTION_AWARE", "app/rag/chunker.py"),
    ("GEP", "Route questions to section filters (rules or LLM)", "On (rules); LLM option",
     "LLM ties rules (MRR 1.00); filter only when a section is named", "QUERY_ROUTING",
     "app/rag/query_router.py"),
    ("GEP", "Parent-child: search small chunks, answer from the page", "On",
     "Correct answers 80% -> 93% on GEP", "always", "app/rag/retriever.py"),
    ("iMac", "Tables from header bars + word positions", "Fallback",
     "All 3 iMac tables exact without vision; used when vision doesn't read the PDF", "LAYOUT_TABLES",
     "app/rag/layout_tables.py"),
    ("iMac", "Footnote markers linked to their notes", "On",
     "Fixed the footnote question", "always", "app/rag/visual.py"),
    ("iMac", "One chunk per element (table, footnote, chart)", "On",
     "Tables never split from their titles", "always", "app/rag/chunker.py"),
    ("iMac", "Charts described by a vision model, cached", "On",
     "Chart questions 86% -> 100%", "VISION_MODEL", "app/rag/visual.py"),
    ("iMac", "Check the reranker doesn't hurt tables", "Checked",
     "iMac MRR 1.00 with and without it", "-", "-"),
    ("Evaluation", "Hand-checked question sets (guide: 37)", "Done",
     "42 questions: GPO 13, GEP 15, iMac 14", "-", "evaluation/questions/*_full.json"),
    ("Evaluation", "MRR and correct-answer counts, Naive vs Advanced", "Done",
     "See the totals above", "-", "evaluation/run.py, evaluation/compare.py"),
    ("Evaluation", "Differences under ~0.1 are noise", "Applied",
     "Stated next to every result in the write-up", "-", "notebooks/NAIVE_VS_ADVANCED_RAG.md"),
    ("Future work", "Question decomposition", "On",
     "FDsys question: context precision 60% -> 100%", "QUERY_DECOMPOSITION", "app/rag/query_decomposer.py"),
    ("Future work", "Agentic search that searches again", "On (chat Auto mode)",
     "Correct citations 83% -> 100% on GPO", "AGENT_MODE", "app/rag/agent.py"),
    ("Future work", "LLM-as-judge evaluation", "Done",
     "Correctness, faithfulness, relevance, context recall", "-", "evaluation/metrics.py"),
    ("Extra", "Context compression (drop passages far below the best)", "On",
     "~15% fewer passages, no lost answer", "RERANK_RELATIVE_MARGIN", "app/rag/reranker.py"),
    ("Extra", "Cost estimates at paid-tier prices", "On",
     "Shown per answer on the use case tabs", "MODEL_PRICES", "evaluation/metrics.py"),
]

STACK = [
    ("Text extraction", "pypdf, PyMuPDF", "Local"),
    ("Embeddings", "sentence-transformers all-MiniLM-L6-v2", "Local"),
    ("Vector store + keyword search", "ChromaDB + rank_bm25", "Local"),
    ("Reranker", "cross-encoder ms-marco-MiniLM-L-6-v2", "Local"),
    ("Answers, rewriting, routing", "Groq: gpt-oss-20b / gpt-oss-120b", "Free tier"),
    ("Judge and fallback answers", "Gemini 3.5 Flash-Lite", "Free tier"),
    ("Charts and tables from page images", "OpenRouter free vision models", "Free tier"),
    ("App", "Streamlit", "Local"),
]


def overview(read: Callable[[str], dict | None]) -> dict | None:
    """Pass counts per document and overall from the full comparison reports
    (plain data, shared with the React UI); None until all three exist."""
    documents, totals = [], {"questions": 0, "naive": 0, "advanced": 0, "naive_precision": 0.0,
                             "advanced_precision": 0.0, "naive_latency": 0.0, "advanced_latency": 0.0}
    for name, path in _FULL_REPORTS.items():
        report = read(path)
        if not report or not report.get("rows"):
            return None
        results = report["rows"]
        naive = sum(r["before"]["passed"] for r in results)
        advanced = sum(r["after"]["passed"] for r in results)
        documents.append({"document": name, "questions": len(results), "naive": naive, "advanced": advanced,
                          "fixed": sum(r["after"]["passed"] and not r["before"]["passed"] for r in results),
                          "broken": sum(r["before"]["passed"] and not r["after"]["passed"] for r in results)})
        totals["questions"] += len(results)
        totals["naive"] += naive
        totals["advanced"] += advanced
        for side, key in (("before", "naive"), ("after", "advanced")):
            totals[f"{key}_precision"] += sum(r[side]["context_precision"] or 0 for r in results)
            totals[f"{key}_latency"] += sum(r[side]["latency_s"] for r in results)
    n = totals["questions"]
    averages = {key: totals[key] / n for key in totals if key.endswith(("_precision", "_latency"))}
    return {"documents": documents, "questions": n, "naive": totals["naive"], "advanced": totals["advanced"],
            **averages}


def render_how_it_works(read: Callable[[str], dict | None]) -> None:
    st.subheader("How it works")
    st.caption("Every technique from the Advanced RAG field guide, what it does here, and what it measured "
               "on the three documents. Built only with free tools.")

    result = overview(read)
    if result is None:
        st.info("No full comparison saved yet: run `python -m evaluation.compare --full`.", icon=":material/info:")
    else:
        n = result["questions"]
        with st.container(horizontal=True):
            st.metric("Naive RAG passed", f"{result['naive']} / {n}", border=True)
            st.metric("Advanced RAG passed", f"{result['advanced']} / {n}",
                      f"{(result['advanced'] - result['naive']) / n:+.0%}", border=True)
            st.metric("Passages from the answer page", f"{result['advanced_precision']:.0%}",
                      f"{(result['advanced_precision'] - result['naive_precision']) * 100:+.0f} pp vs Naive",
                      border=True)
            st.metric("Average answer time", f"{result['advanced_latency']:.1f} s",
                      f"{result['advanced_latency'] - result['naive_latency']:+.1f} s vs Naive",
                      delta_color="inverse", border=True)
        table = pd.DataFrame(result["documents"]).rename(columns={
            "document": "Document", "questions": "Questions", "naive": "Naive RAG passed",
            "advanced": "Advanced RAG passed", "fixed": "Fixed by Advanced", "broken": "Broken by Advanced"})
        st.dataframe(table, hide_index=True)
        st.caption("Hand-checked questions graded by an AI judge (answer correct and complete, and its page "
                   "retrieved); the same answer model on both sides. Differences under about 10 points are noise.")

    st.markdown("##### The pipeline")
    st.mermaid_chart(PIPELINE)

    st.markdown("##### Field guide techniques")
    techniques = pd.DataFrame(TECHNIQUES, columns=["Area", "Technique", "Status", "Measured on our documents",
                                                   "Setting (.env)", "Code"])
    st.dataframe(
        techniques, hide_index=True,
        column_config={
            "Area": st.column_config.TextColumn(width="small"),
            "Technique": st.column_config.TextColumn(width="large"),
            "Measured on our documents": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption("On = used by Advanced RAG. Fallback = used automatically when the main method can't run. "
               "Option = built and measured, off by default because it didn't beat the default here.")

    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown("##### Free stack")
            st.dataframe(pd.DataFrame(STACK, columns=["Part", "Tool", "Runs"]), hide_index=True)
        with st.container(border=True):
            st.markdown("##### Experiment")
            st.markdown(
                "- **Chunking:** use the controls at the top of this page, then *Apply* on a use case tab.\n"
                "- **Any switch:** change it in `.env` (table above), then restart Streamlit.\n"
                "- **Measure retrieval (no LLM):** `python -m evaluation.run --questions "
                "evaluation/questions/gpo_full.json --retrieval-only`\n"
                "- **Measure answers:** `python -m evaluation.compare --full`\n"
                "- **Full write-up:** `notebooks/NAIVE_VS_ADVANCED_RAG.md`"
            )
