from types import SimpleNamespace

import pytest

from evaluation.metrics import (
    change,
    context_precision,
    grade_grounding,
    metric_rows,
    parse_grounding,
    estimate_cost,
    summary_points,
)


def test_context_precision_is_the_share_of_passages_from_the_answer_pages():
    passages = [{"page": 96}, {"page": 189}, {"page": 97}, {"page": 4}]
    assert context_precision(passages, [96, 97]) == 0.5
    assert context_precision(passages, []) is None  # no reference pages: not measurable
    assert context_precision([], [96]) == 0.0


@pytest.mark.parametrize("before, after, expected", [
    (0.6, 0.8, (20.0, 33.3)),     # +20 percentage points, +33.3% relative
    (0.0, 0.5, (50.0, None)),     # no relative change from zero
    (0.8, 0.6, (-20.0, -25.0)),
    (None, 0.5, None),
])
def test_change_separates_percentage_points_from_relative_change(before, after, expected):
    assert change(before, after) == expected


def test_parse_grounding_turns_counts_into_scores():
    reply = ('{"claims": 4, "supported": 3, "unsupported": ["growth of 7 percent"], '
             '"relevance": 5, "relevance_reason": "Answers it.", "facts": 2, "covered": 1}')
    g = parse_grounding(reply)
    assert g["faithfulness"] == 0.75 and g["unsupported"] == ["growth of 7 percent"]
    assert g["answer_relevance"] == 1.0  # 1-5 scale mapped to 0-1
    assert g["context_recall"] == 0.5


def test_parse_grounding_without_claims_or_facts_gives_no_score():
    g = parse_grounding('{"claims": 0, "supported": 0, "relevance": 1, "facts": 0, "covered": 0}')
    assert g["faithfulness"] is None and g["context_recall"] is None and g["answer_relevance"] == 0.0


def test_unreadable_grounding_reply_scores_nothing():
    g = parse_grounding("sorry")
    assert g["faithfulness"] is None and g["answer_relevance"] is None and g["error"]


def test_grade_grounding_retries_an_empty_reply(monkeypatch):
    monkeypatch.setattr("evaluation.metrics.time.sleep", lambda s: None)
    replies = ["", '{"claims": 1, "supported": 1, "relevance": 5, "facts": 1, "covered": 1}']
    gen = SimpleNamespace(complete=lambda prompt, **kwargs: replies.pop(0))

    g = grade_grounding(gen, "q?", "ref", "answer", [{"rank": 1, "page": 5, "text": "evidence"}], "groq-model")
    assert g["faithfulness"] == 1.0 and replies == []


def _side(**overrides):
    side = {
        "correct": False, "hit": False, "rank": None, "citation_ok": False,
        "faithfulness": 0.5, "answer_relevance": 0.75, "context_precision": 0.2, "context_recall": 0.0,
        "passages_count": 5, "irrelevant_passages": 4, "latency_s": 2.0, "retrieval_latency_s": 0.1,
        "prompt_tokens": 2000, "completion_tokens": 200, "rewritten": False,
        "original_query": "q?", "search_query": "q?", "searches": 1, "candidates": 5, "kept": 5, "reranked": False,
        "rewrite_effect": None, "rerank_effect": None,
    }
    side.update(overrides)
    return side


def test_metric_rows_report_measured_values_and_mark_unpriced_cost_not_available():
    rows = {r["Metric"]: r for r in metric_rows(
        _side(), _side(correct=True, hit=True, rank=1, faithfulness=1.0, context_precision=0.8, latency_s=3.0,
                       prompt_tokens=3000))}

    assert rows["Faithfulness to the document"]["Change"] == "+50.0 pp (+100.0% relative)"
    assert rows["Context precision"]["Change"] == "+60.0 pp (+300.0% relative)"
    assert rows["Response latency"]["Naive RAG"] == "2.00 s" and rows["Response latency"]["Change"] == "+1.00 s"
    assert rows["Estimated cost"]["Naive RAG"] == "N/A" and "price" in rows["Estimated cost"]["How it is measured"]


def test_summary_reports_regressions_honestly():
    points = dict((p["question"], p) for p in summary_points(
        _side(faithfulness=1.0), _side(faithfulness=0.5, latency_s=4.0)))

    assert points["Did the answer become more faithful to the document?"]["status"] == "regressed"
    assert points["Did latency increase or decrease?"]["status"] == "cost"
    assert points["Overall, compared with Naive RAG?"]["status"] == "regressed"


def test_overall_is_about_the_answer_not_cost_or_a_passage_swap():
    # Same correct answer; one passage swapped by the reranker; slower and more tokens.
    effect = {"rank_without": 1, "rank_with": 1, "precision_without": 0.8, "precision_with": 0.6}
    points = dict((p["question"], p) for p in summary_points(
        _side(correct=True, hit=True, rank=1, faithfulness=1.0, context_precision=0.8, latency_s=0.8),
        _side(correct=True, hit=True, rank=1, faithfulness=1.0, context_precision=0.6, latency_s=2.1,
              prompt_tokens=2141, reranked=True, rerank_effect=effect)))

    assert points["Did Advanced RAG retrieve more relevant information?"]["status"] == "regressed"
    assert points["Did token usage or estimated cost increase or decrease?"]["status"] == "cost"
    overall = points["Overall, compared with Naive RAG?"]
    assert overall["status"] == "same"
    assert "Retrieval got worse" in overall["text"] and "+1.30 s" in overall["text"]


def test_a_fixed_answer_is_an_overall_improvement_even_when_slower():
    points = summary_points(_side(), _side(correct=True, latency_s=5.0))
    assert points[-1]["status"] == "improved"


PRICES = {"big": (0.15, 0.60), "small": (0.075, 0.30)}


def test_estimate_cost_prices_each_call_at_its_own_model():
    usage = [{"model": "big", "prompt_tokens": 2_000_000, "completion_tokens": 1_000_000},
             {"model": "small", "prompt_tokens": 1_000_000, "completion_tokens": 0}]
    assert estimate_cost(usage, PRICES) == pytest.approx(0.30 + 0.60 + 0.075)


def test_estimate_cost_is_unknown_without_calls_or_without_a_price():
    assert estimate_cost([], PRICES) is None
    assert estimate_cost([{"model": "other", "prompt_tokens": 10, "completion_tokens": 1},
                          {"model": "big", "prompt_tokens": 10, "completion_tokens": 1}], PRICES) is None


def test_cost_row_and_summary_show_the_estimate_and_its_change():
    before, after = _side(cost_usd=0.00042), _side(prompt_tokens=3000, cost_usd=0.00057)
    rows = {r["Metric"]: r for r in metric_rows(before, after)}
    points = dict((p["question"], p) for p in summary_points(before, after))

    cost = rows["Estimated cost"]
    assert (cost["Naive RAG"], cost["Advanced RAG"], cost["Change"]) == ("$0.00042", "$0.00057", "+$0.00015")
    tokens = points["Did token usage or estimated cost increase or decrease?"]
    assert "$0.00042 → $0.00057" in tokens["text"] and tokens["status"] == "cost"
    assert "+$0.00015" in points["Overall, compared with Naive RAG?"]["text"]


def test_summary_reports_what_decomposition_and_keyword_search_did():
    effect = {"rank_without": None, "rank_with": 1, "precision_without": 0.2, "precision_with": 0.6}
    points = dict((p["question"], p) for p in summary_points(
        _side(), _side(sub_queries=["a?", "b?"], decompose_effect=effect, keyword_effect=None)))

    decompose = points["Did query decomposition help?"]
    assert decompose["status"] == "improved" and decompose["text"].startswith("Split into 2 sub-questions.")
    assert points["Did hybrid keyword search help?"]["status"] == "n/a"
