"""Use-case results page: how ALLY AI scores on each hands-on RAG use case.

Reads the reports `python -m evaluation.run` writes (evaluation/reports/)
and the ground-truth questions files (evaluation/questions/), so what is
shown here is exactly what the evaluation measured — nothing is re-run
except the unit tests, on request.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app.rag.citations import cited_pages

ROOT = Path(__file__).resolve().parents[2]

# One entry per use case: its pass bar from the spec, how that bar is checked
# ("bar": "section" or "fact", see _meets_pass_bar), its questions file, the
# final report, and (optionally) the "before" report it improves on.
USE_CASES = [
    {
        "title": "Use case 1: GPO Annual Report (baseline RAG)",
        "document": "GPO 2018 Annual Report, 82 pages",
        "pass_bar": "The exact fact from the right paragraph, nothing invented.",
        "bar": "fact",
        "questions": "evaluation/questions/gpo.json",
        "report": "evaluation/reports/gpo_final.json",
        "baseline": None,
    },
    {
        "title": "Use case 2: World Bank GEP June 2026 (long, structured report)",
        "document": "Global Economic Prospects, 200 pages",
        "pass_bar": "The right chapter and section is found, and no facts are mixed in from another region or chapter.",
        "bar": "section",
        "questions": "evaluation/questions/gep.json",
        "report": "evaluation/reports/gep_final.json",
        "baseline": "evaluation/reports/gep_baseline_standard.json",
    },
    {
        "title": "Use case 3: Apple iMac Environmental Report (charts, tables, footnotes)",
        "document": "iMac Product Environmental Report, 12 pages",
        "pass_bar": "The right number is pulled from a chart or table, not just a general sentence nearby.",
        "bar": "fact",
        "questions": "evaluation/questions/imac.json",
        "report": "evaluation/reports/imac_final.json",
        "baseline": "evaluation/reports/imac_baseline.json",
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


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _meets_pass_bar(bar: str, answer: dict, expected_pages: list[int], retrieved_right_page: bool) -> bool:
    """The spec's pass bar, which is looser than the judge's completeness check.

    "section" (use case 2): the answer cites pages, all inside the expected
    section, so nothing was borrowed from another region or chapter.
    "fact" (use cases 1 and 3): the judged-correct fact, with its page among
    the retrieved passages (so it came from the right place, whatever the
    citation format; page citations have their own column).
    Questions whose answer is not in the document pass when the system declines.
    """
    if not expected_pages:
        return bool(answer["correct"])
    if bar == "section":
        cited = cited_pages(answer["answer"])
        return bool(cited) and cited <= set(expected_pages)
    return bool(answer["correct"]) and retrieved_right_page


def _results_table(questions: dict, report: dict, bar: str) -> pd.DataFrame:
    """One row per question: verdict, citation check, pages retrieved vs expected."""
    retrieval = {row["id"]: row for row in report["retrieval"]["rows"]}
    rows = []
    for answer in report["answers"]["rows"]:
        question = questions.get(answer["id"], {})
        hit = retrieval.get(answer["id"], {})
        rows.append({
            "Sample": bool(question.get("sample")),
            "Question": answer["question"],
            "Meets pass bar": _meets_pass_bar(bar, answer, question.get("expected_pages") or [], bool(hit.get("hit"))),
            "Correct (complete)": answer["correct"],
            "Cites right page": {True: "Yes", False: "No"}.get(answer["citation_ok"], "n/a (no answer expected)"),
            "Right section found": "Yes" if hit.get("hit") else ("n/a" if not question.get("expected_pages") else "No"),
            "Expected pages": ", ".join(map(str, question.get("expected_pages", []))) or "–",
            "Retrieved pages": ", ".join(map(str, sorted(set(hit.get("retrieved_pages", []))))) or "–",
            "Judge's reason": answer["judge_reason"],
        })
    return pd.DataFrame(rows)


def _render_use_case(use_case: dict) -> None:
    st.subheader(use_case["title"])
    st.caption(use_case["document"])
    st.markdown(f":material/flag: **Pass bar:** {use_case['pass_bar']}")

    questions_file = _read(use_case["questions"])
    report = _read(use_case["report"])
    if not report or not questions_file or not report.get("answers"):
        st.warning(f"Not done yet: no evaluation report at `{use_case['report']}`.", icon=":material/pending:")
        return
    questions = {q["id"]: q for q in questions_file["questions"]}
    table = _results_table(questions, report, use_case["bar"])
    samples = table[table["Sample"]]
    summary, retrieval = report["answers"]["summary"], report["retrieval"]["summary"]

    baseline = _read(use_case["baseline"])
    base_answers = (baseline or {}).get("answers") or {}
    base_summary = base_answers.get("summary", {})
    base_retrieval = (baseline or {}).get("retrieval", {}).get("summary", {})

    def delta(now: float | None, before: float | None) -> str | None:
        if now is None or before is None:
            return None
        return f"{(now - before) * 100:+.0f} pts vs before"

    bar_ok, complete = table["Meets pass bar"], table["Correct (complete)"]
    samples_pass = bool(len(samples)) and bool(samples["Meets pass bar"].all())
    with st.container(horizontal=True):
        st.metric("Samples meeting the pass bar", f"{int(samples['Meets pass bar'].sum())} / {len(samples)}", border=True)
        st.metric("All questions meeting the pass bar", f"{int(bar_ok.sum())} / {len(table)}", border=True)
        # Compared only when the baseline asked exactly the same questions.
        same_questions = {r["id"] for r in base_answers.get("rows", [])} == {r["id"] for r in report["answers"]["rows"]}
        st.metric("Correct and complete (strict judge)", f"{int(complete.sum())} / {len(table)}",
                  delta(summary.get("accuracy"), base_summary.get("accuracy")) if same_questions else None, border=True)
        st.metric("Cites the right page", _pct(summary.get("citation_accuracy")),
                  delta(summary.get("citation_accuracy"), base_summary.get("citation_accuracy")), border=True)
        st.metric("Right section retrieved", _pct(retrieval.get("hit_rate")),
                  delta(retrieval.get("hit_rate"), base_retrieval.get("hit_rate")), border=True)

    if samples_pass and bar_ok.all() and complete.all():
        st.success("Passed: every question meets the pass bar, and the strict judge found every answer complete.",
                   icon=":material/check_circle:")
    elif samples_pass and bar_ok.all():
        st.success(f"Passed: every question meets the pass bar. The strict judge found {int((~complete).sum())} "
                   "answer(s) incomplete (see \"Correct (complete)\" and the judge's reason).",
                   icon=":material/check_circle:")
    elif samples_pass:
        st.info("The sample questions meet the pass bar; some extra questions do not (see the table).",
                icon=":material/info:")
    else:
        st.error("Not passed yet: at least one sample question misses the pass bar.", icon=":material/cancel:")

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Sample": st.column_config.CheckboxColumn("Sample", help="One of the 3 questions from the task spec"),
            "Meets pass bar": st.column_config.CheckboxColumn(
                "Meets pass bar", help=f"This use case's pass bar from the spec: {use_case['pass_bar']}"),
            "Correct (complete)": st.column_config.CheckboxColumn(
                "Correct (complete)",
                help="Strict check: an AI judge compares the answer with the reference answer; any missing key fact fails it"),
            "Question": st.column_config.TextColumn(width="large"),
            "Judge's reason": st.column_config.TextColumn(width="large"),
        },
    )

    with st.expander("Full answers"):
        for answer in report["answers"]["rows"]:
            icon = ":material/check_circle:" if answer["correct"] else ":material/cancel:"
            st.markdown(f"{icon} **{answer['question']}**")
            st.markdown(answer["answer"])
            reference = questions.get(answer["id"], {}).get("reference_answer")
            if reference:
                st.caption(f"Reference answer: {reference}")
            st.divider()

    if baseline and base_answers:
        with st.expander("Before vs after, per question"):
            before = {row["id"]: row["correct"] for row in base_answers["rows"]}
            st.dataframe(
                pd.DataFrame([
                    {"Question": row["question"],
                     "Before": {True: "Correct", False: "Wrong"}.get(before.get(row["id"]), "Not asked"),
                     "After": "Correct" if row["correct"] else "Wrong"}
                    for row in report["answers"]["rows"]
                ]),
                hide_index=True,
            )

    config = report["config"]
    st.caption(
        f"Configuration: chunks {config['chunk_size']}/{config['chunk_overlap']}, top {config['top_k']}, "
        f"reranker {config.get('reranker_model') or 'off'}, section-aware {'on' if config.get('section_aware') else 'off'}, "
        f"answer model {report['answer_model']}. Report: `{use_case['report']}`."
    )


def _render_unit_tests() -> None:
    st.subheader("Unit tests")
    st.caption("Runs the project's automated test suite (pytest), about 30 seconds.")
    if st.button("Run unit tests", icon=":material/play_arrow:"):
        with st.spinner("Running pytest..."):
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=ROOT, capture_output=True, text=True,
            )
        summary = next((line for line in reversed(result.stdout.splitlines()) if line.strip()), "")
        if result.returncode == 0:
            st.success(summary, icon=":material/check_circle:")
        else:
            st.error(summary or "pytest failed to run", icon=":material/cancel:")
            st.code(result.stdout[-4000:] + result.stderr[-2000:])


def render_evaluation_page() -> None:
    st.title("Use case results")
    st.write(
        "Each use case is checked with the sample questions from the task spec plus extra, deliberately "
        "confusable questions. An AI judge grades every answer against a reference answer written from the PDF."
    )
    tabs = st.tabs([use_case["title"].split(":")[0] for use_case in USE_CASES] + ["Unit tests"])
    for tab, use_case in zip(tabs, USE_CASES):
        with tab:
            _render_use_case(use_case)
    with tabs[-1]:
        _render_unit_tests()
