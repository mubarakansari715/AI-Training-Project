"""UI states of the Naive vs Advanced comparison on the results page.

The pipelines are replaced by a fake (`MODE` picks what it does), so no
model or document is loaded and no API is called.
"""

from streamlit.testing.v1 import AppTest

from evaluation.compare import PipelineFailed, make_row

MODE = {"run": "ok", "calls": 0}

SCRIPT = """
import app.ui.evaluation_page as page
import tests.test_evaluation_ui as ui

page._run_comparison = ui.fake_run
read = page._read
page._read = lambda path: None if path and path.endswith("_compare.json") else read(path)  # no saved runs
page.render_evaluation_page()
"""


def fake_side(label, passed):
    return {
        "label": label, "technique": "technique", "config": "cfg", "model": "m",
        "answer": "Growth slows to 6.3 percent, costing $5 (r.pdf, page 96).",
        "passed": passed, "correct": passed, "judge_reason": "Matches." if passed else "Wrong figure.",
        "hit": passed, "rank": 1 if passed else None, "citation_ok": passed,
        "faithfulness": 1.0 if passed else 0.5, "answer_relevance": 1.0, "context_recall": 1.0 if passed else 0.0,
        "context_precision": 0.6 if passed else 0.0, "unsupported": [] if passed else ["7 percent growth"],
        "relevance_reason": "", "grounding_error": None, "passages_count": 5, "irrelevant_passages": 2 if passed else 5,
        "latency_s": 3.0 if passed else 2.0, "retrieval_latency_s": 0.4, "prompt_tokens": 2500, "completion_tokens": 150,
        "original_query": "q", "search_query": "rewritten q" if passed else "q", "rewritten": passed,
        "queries_generated": 1, "searches": 1, "candidates": 12, "kept": 5, "reranked": passed,
        "rewrite_effect": {"rank_without": 3, "rank_with": 1, "precision_without": 0.2, "precision_with": 0.6}
        if passed else None,
        "rerank_effect": {"rank_without": 2, "rank_with": 1, "precision_without": 0.4, "precision_with": 0.6}
        if passed else None,
        "steps": [{"kind": "Retrieve", "text": "Similarity search"}, {"kind": "Generate", "text": "Grounded prompt"}],
        "passages": [{"rank": 1, "page": 96, "section": "Ch. 2 › South Asia", "text": "Growth in SAR " * 80,
                      "expected": True}],
    }


def fake_run(document, question, chunking):
    MODE["calls"] += 1
    MODE["chunking"] = chunking
    if MODE["run"] == "fail":
        raise PipelineFailed("Naive RAG: the answer model gave no answer after 3 tries.")
    return make_row(question, {"before": fake_side("Naive RAG", False), "after": fake_side("Advanced RAG", True)})


def _app(mode="ok"):
    MODE.update(run=mode, calls=0)
    return AppTest.from_string(SCRIPT, default_timeout=60).run()


def test_nothing_runs_until_a_question_is_chosen():
    at = _app()
    assert not at.exception
    assert at.button(key="apply_gpo").disabled  # Apply waits for a question
    assert "##### Naive RAG" not in [m.value for m in at.markdown]


def test_only_main_questions_listed_and_apply_appears_without_running():
    at = _app()
    box = at.selectbox(key="question_gpo")
    assert len(box.options) == 3
    assert all(option.startswith("Main document question") for option in box.options)

    at = box.select("sample-1").run()
    assert not at.button(key="apply_gpo").disabled
    assert MODE["calls"] == 0  # selecting alone does not run the pipelines


def test_apply_shows_both_panels_summary_and_metrics_table():
    at = _app()
    at = at.selectbox(key="question_gpo").select("sample-1").run()
    at = at.button(key="apply_gpo").click().run()

    assert not at.exception and MODE["calls"] == 1
    markdown = [m.value for m in at.markdown]
    assert "##### Naive RAG" in markdown and "##### Advanced RAG" in markdown
    assert any("Did query rewriting help?" in m for m in markdown)
    assert any("\\$5" in m for m in markdown)  # dollar signs are not read as formulas
    metrics = next(d.value for d in at.dataframe if "Metric" in d.value.columns)
    assert "Faithfulness to the document" in list(metrics["Metric"])
    assert list(metrics.loc[metrics["Metric"] == "Estimated cost", "Naive RAG"]) == ["N/A"]


def test_pipeline_failure_shows_an_error_instead_of_results():
    at = _app("fail")
    at = at.selectbox(key="question_gpo").select("sample-1").run()
    at = at.button(key="apply_gpo").click().run()

    assert not at.exception
    assert any("no answer after 3 tries" in e.value for e in at.error)
    assert "##### Advanced RAG" not in [m.value for m in at.markdown]


def test_chunk_controls_feed_the_comparison():
    at = _app()
    at.segmented_control(key="chunk_size_choice").set_value("300").run()
    at.segmented_control(key="chunk_strategy").set_value("sentence").run()
    at = at.selectbox(key="question_gpo").select("sample-1").run()
    at = at.button(key="apply_gpo").click().run()

    assert not at.exception
    size, overlap, strategy, unit = MODE["chunking"]
    assert (size, strategy, unit) == (300, "sentence", "chars") and overlap < size


def test_naive_side_keeps_fixed_size_chunks_whatever_the_method():
    from app.ui.evaluation_page import _side_configs

    configs = _side_configs((300, 50, "sentence", "tokens"))

    assert (configs["before"].chunk_size, configs["before"].chunk_unit, configs["before"].chunk_strategy) == (
        300, "tokens", "recursive")
    assert (configs["after"].chunk_size, configs["after"].chunk_strategy) == (300, "sentence")


def test_how_it_works_tab_shows_totals_pipeline_and_every_guide_technique():
    from app.ui.how_it_works import TECHNIQUES

    at = AppTest.from_string("""
import app.ui.how_it_works as how
import tests.test_evaluation_ui as ui
how.render_how_it_works(ui.full_report)
""", default_timeout=60).run()

    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Naive RAG passed"] == "3 / 6" and values["Advanced RAG passed"] == "6 / 6"
    techniques = next(d.value for d in at.dataframe if "Technique" in d.value.columns)
    assert len(techniques) == len(TECHNIQUES) and "Hybrid search" in " ".join(techniques["Technique"])


def full_report(path):
    def row(naive):
        side = {"passed": naive, "context_precision": 0.5, "latency_s": 1.0}
        return {"before": side, "after": dict(side, passed=True, latency_s=2.0)}

    return {"rows": [row(True), row(False)]}
