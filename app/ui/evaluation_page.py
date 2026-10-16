"""Use-case results page: Naive RAG vs Advanced RAG on each hands-on use case.

Questions come from the ground-truth files (evaluation/questions/). A
picked question runs through both pipelines only when Apply is clicked
(evaluation.compare); saved runs (evaluation/reports/<use case>_compare.json)
are shown without calling a model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config.settings import settings
from app.rag.citations import cited_pages
from app.ui.how_it_works import render_how_it_works
from evaluation.compare import (
    SIDES,
    PipelineFailed,
    make_row,
    question_options,
    run_side,
    side_config,
)
from evaluation.harness import Throttle, build_pipeline
from evaluation.metrics import metric_rows, summary_points

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
# Pinned for the live comparison, like the saved reports (evaluation.compare
# --model/--judge-model): free-tier models, the judge on another provider so
# grading doesn't use the answer model's daily tokens; calls are paced for
# Groq's free-tier per-minute limit.
_ANSWER_MODEL = "openai/gpt-oss-20b"
_JUDGE_MODEL = "gemini-3.5-flash-lite"
_CALLS_PER_MINUTE = 4
_CHUNK_SIZES = [300, 500, 800]

# One entry per use case: its pass bar from the spec, its questions file and
# its saved Naive vs Advanced runs (evaluation.compare).
USE_CASES = [
    {
        "title": "Use case 1: GPO Annual Report (baseline RAG)",
        "tab": "GPO report",
        "document": "GPO 2018 Annual Report, 82 pages",
        "pass_bar": "The exact fact from the right paragraph, nothing invented.",
        "questions": "evaluation/questions/gpo.json",
        "key": "gpo",
        "compare": "evaluation/reports/gpo_compare.json",
    },
    {
        "title": "Use case 2: World Bank GEP June 2026 (long, structured report)",
        "tab": "World Bank GEP",
        "document": "Global Economic Prospects, 200 pages",
        "pass_bar": "The right chapter and section is found, and no facts are mixed in from another region or chapter.",
        "questions": "evaluation/questions/gep.json",
        "key": "gep",
        "compare": "evaluation/reports/gep_compare.json",
    },
    {
        "title": "Use case 3: Apple iMac Environmental Report (charts, tables, footnotes)",
        "tab": "iMac report",
        "document": "iMac Product Environmental Report, 12 pages",
        "pass_bar": "The right number is pulled from a chart or table, not just a general sentence nearby.",
        "questions": "evaluation/questions/imac.json",
        "key": "imac",
        "compare": "evaluation/reports/imac_compare.json",
    },
]


@st.cache_data(max_entries=20)
def _load_json(path: str, mtime: float) -> dict:
    """Cached per file version (`mtime`), so a re-run report shows up on refresh."""
    return json.loads((ROOT / path).read_text())


def _read(path: str | None) -> dict | None:
    if not path or not (ROOT / path).exists():
        return None
    return _load_json(path, (ROOT / path).stat().st_mtime)


# Badge colour per pipeline step kind (see evaluation.compare.describe_steps).
_STEP_COLORS = {"Query": "violet", "Retrieve": "blue", "Post-process": "green", "Generate": "orange"}
_PASSAGE_PREVIEW_CHARS = 700
_STATUS_ICONS = {
    "improved": ":green[:material/trending_up: improved]",
    "regressed": ":red[:material/trending_down: regressed]",
    "mixed": ":orange[:material/swap_vert: mixed]",
    "same": ":gray[:material/trending_flat: no change]",
    "n/a": ":gray[:material/remove: not measurable]",
    "cost": ":orange[:material/payments: cost of the extra steps]",
}


def _md(text: str) -> str:
    """Model text as Markdown: a "$" would otherwise start a formula."""
    return text.replace("$", "\\$")


def _side_configs(chunking: tuple) -> dict:
    """Both sides' configs at the chunk settings picked at the top of the page:
    size, overlap and unit apply to both sides, the method to Advanced only
    (Naive RAG keeps fixed-size chunks, the field guide's baseline)."""
    size, overlap, strategy, unit = chunking
    configs = {}
    for side in SIDES:
        config = replace(side_config(side), chunk_size=size, chunk_overlap=overlap, chunk_unit=unit)
        configs[side] = config if side == "before" else replace(config, chunk_strategy=strategy)
    return configs


@st.cache_resource(max_entries=4, show_spinner=False)
def _comparison_pipelines(document: str, chunking: tuple) -> dict:
    """Naive and Advanced pipelines over one document at one chunk setting,
    indexed once per server process (a 200-page report takes a few minutes)."""
    return {side: (build_pipeline(document, config)[0], config) for side, config in _side_configs(chunking).items()}


def _chunk_controls() -> tuple:
    """Chunk size, overlap, method and unit for the comparison, at the top of the page."""
    with st.container(border=True):
        st.markdown("**Chunking** (applies to every use case; a new setting re-indexes the document once)")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            choice = st.segmented_control("Chunk size", [*map(str, _CHUNK_SIZES), "Custom"],
                                          default=str(settings.chunk_size) if settings.chunk_size in _CHUNK_SIZES
                                          else "Custom", key="chunk_size_choice")
            size = (st.number_input("Custom size", min_value=50, max_value=4000, value=settings.chunk_size, step=50,
                                    key="chunk_size_custom")
                    if choice in (None, "Custom") else int(choice))
            overlap = st.number_input("Overlap", min_value=0, max_value=max(0, size - 1),
                                      value=min(settings.chunk_overlap, size // 4), step=10, key=f"overlap_{size}")
            unit = st.segmented_control("Unit", ["chars", "tokens"], default=settings.chunk_unit, key="chunk_unit")
            strategy = st.segmented_control("Method (Advanced RAG)", ["recursive", "sentence"],
                                            default=settings.chunk_strategy, key="chunk_strategy")
        if (unit or settings.chunk_unit) == "tokens" and size > 256:
            st.caption(":orange[:material/warning:] The embedding model reads only the first 256 tokens of a chunk.")
    return int(size), int(overlap), strategy or settings.chunk_strategy, unit or settings.chunk_unit


def _question_label(option: dict) -> str:
    group = "Main document question" if option["group"] == "main" else "Related question"
    return f"{group} · {option['question']}"


def _run_comparison(document: str, question: dict, chunking: tuple) -> dict:
    """Both pipelines on the selected question, with progress shown as it runs."""
    with st.status("Running both pipelines on the same question...", expanded=True) as status:
        status.write("Indexing the document for both pipelines (first time only, can take a few minutes)...")
        pipelines = _comparison_pipelines(document, chunking)
        throttle = Throttle(_CALLS_PER_MINUTE)
        sides = {}
        for side, (pipeline, config) in pipelines.items():
            status.write(f"{SIDES[side][0]}: answering, then grading the answer...")
            sides[side] = run_side(pipeline, question, side, config, _ANSWER_MODEL, _JUDGE_MODEL, throttle)
        status.update(label="Done: both pipelines answered", state="complete", expanded=False)
    return make_row(question, sides)


def _render_side(side: dict) -> None:
    """One pipeline's answer, retrieval, citations, steps, passages and scores."""
    st.markdown(f"##### {side['label']}")
    st.caption(side["technique"])
    where = f"answer passage at rank {side['rank']}" if side["rank"] else (
        "answer passage not retrieved" if side["hit"] is False else "document has no answer")
    if side["passed"]:
        st.badge(f"PASS · {where}", icon=":material/check:", color="green")
    else:
        reason = "answer judged incomplete or wrong" if not side["correct"] else "answer page not retrieved"
        st.badge(f"FAIL · {reason} · {where}", icon=":material/close:", color="red")
    st.markdown(_md(side["answer"]))
    cited = sorted(cited_pages(side["answer"]))
    citation = {True: "cites a right page", False: "no right page cited", None: "nothing to cite"}[side["citation_ok"]]
    st.caption(f"Pages cited: {', '.join(map(str, cited)) or 'none'} ({citation}) · Judge: {_md(side['judge_reason'])}")
    if side["unsupported"]:
        st.warning("Claims the judge could not find in the passages: " + "; ".join(map(_md, side["unsupported"])),
                   icon=":material/report:")
    if side["grounding_error"]:
        st.caption(f"Grounding scores unavailable: {side['grounding_error']}")
    with st.expander("Pipeline: what happened, step by step"):
        st.markdown("\n".join(
            f"{i}. :{_STEP_COLORS.get(step['kind'], 'gray')}-badge[{step['kind']}] {_md(step['text'])}"
            for i, step in enumerate(side["steps"], start=1)
        ))
        st.caption(f"Configuration: {side['config']} · answer model {side['model']}")
    with st.expander(f"Passages handed to the model ({len(side['passages'])})"):
        for passage in side["passages"]:
            with st.container(border=True):
                header = f"**[{passage['rank']}] p. {passage['page']}**"
                if passage["section"]:
                    header += f" · {passage['section']}"
                if passage["expected"]:
                    header += " :green-badge[from a page that answers the question]"
                st.markdown(header)
                text = passage["text"]
                st.text(text if len(text) <= _PASSAGE_PREVIEW_CHARS else text[:_PASSAGE_PREVIEW_CHARS] + " ...")


def _render_comparison_result(row: dict, source: str) -> None:
    st.caption(source)
    before, after = st.columns(2, gap="large")
    with before:
        _render_side(row["before"])
    with after:
        _render_side(row["after"])

    st.markdown("##### Before vs after: summary")
    for point in summary_points(row["before"], row["after"]):
        st.markdown(f"**{point['question']}** {_STATUS_ICONS[point['status']]}  \n{_md(point['text'])}")

    rows = metric_rows(row["before"], row["after"])
    with st.expander("Accuracy and evaluation metrics"):
        st.dataframe(pd.DataFrame(rows).drop(columns="How it is measured"), hide_index=True, height="content")
        st.caption("pp = percentage points (80% → 90% is +10 pp, or +12.5% relative).")
        st.markdown("\n".join(f"- **{r['Metric']}:** {_md(r['How it is measured'])}" for r in rows))


def _render_comparison(use_case: dict, chunking: tuple) -> None:
    """Section A-C: pick a question, Apply, see Naive vs Advanced side by side."""
    questions_file = _read(use_case["questions"])
    if not questions_file:
        st.warning(f"No questions file at `{use_case['questions']}`.", icon=":material/pending:")
        return
    options = question_options(questions_file)
    by_id = {o["id"]: o for o in options}
    with st.container(horizontal=True, vertical_alignment="bottom"):
        picked = st.selectbox(
            "Question", list(by_id), index=None, format_func=lambda qid: _question_label(by_id[qid]),
            placeholder="Choose a question: main document questions first, then related ones",
            key=f"question_{use_case['key']}",
        )
        apply = st.button("Apply", type="primary", icon=":material/play_arrow:", disabled=picked is None,
                          key=f"apply_{use_case['key']}")
    if picked is None:
        return
    question = by_id[picked]
    if not question["question"].strip():
        st.warning("This question is empty.", icon=":material/error:")
        return

    results = st.session_state.setdefault("comparisons", {})
    cache_key = f"{use_case['key']}:{picked}:{chunking}"
    if apply:
        try:
            results[cache_key] = _run_comparison(questions_file["document"], question, chunking)
        except PipelineFailed as exc:
            st.error(str(exc), icon=":material/error:")
        except Exception:
            logger.exception("Comparison failed for %s", cache_key)
            st.error("Running the pipelines failed (see the server log). Nothing was saved.", icon=":material/error:")

    if cache_key in results:
        row = results[cache_key]
        _render_comparison_result(row, f"Run in this session at {row['generated_at']}.")
        return
    saved_file = _read(use_case["compare"]) or {}
    labels = {side: config.label for side, config in _side_configs(chunking).items()}
    saved = {r["id"]: r for r in saved_file.get("rows", [])}.get(picked)
    if saved and saved_file.get("configs") not in (None, labels):
        st.caption("No saved run at these chunk settings. Apply runs both pipelines live: about 6-8 LLM calls "
                   "on the Groq free tier.")
    elif saved and "passed" in saved.get("after", {}):
        _render_comparison_result(saved, f"Saved run from {saved.get('generated_at', 'an earlier run')} "
                                         "(`python -m evaluation.compare`). Apply runs it again live.")
    else:
        st.caption("Not run yet. Apply runs both pipelines live: about 6-8 LLM calls on the Groq free tier.")


def _render_use_case(use_case: dict, chunking: tuple) -> None:
    st.subheader(use_case["title"])
    st.caption(f"{use_case['document']} · :material/flag: Pass bar: {use_case['pass_bar']}")

    _render_comparison(use_case, chunking)


def render_evaluation_page() -> None:
    st.title("Use case results")
    st.caption("Naive RAG vs Advanced RAG on the same document and question; an AI judge grades each answer "
               "against a reference written from the PDF.")
    chunking = _chunk_controls()
    overview, *tabs = st.tabs([":material/schema: How it works", *(use_case["tab"] for use_case in USE_CASES)])
    with overview:
        render_how_it_works(_read)
    for tab, use_case in zip(tabs, USE_CASES):
        with tab:
            _render_use_case(use_case, chunking)
