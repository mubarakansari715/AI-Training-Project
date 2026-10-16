from types import SimpleNamespace

import pytest

from app.rag.query_decomposer import MAX_SUB_QUESTIONS, decompose_query, needs_decomposition


def _generator(reply, configured=True):
    calls = []

    def complete(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return reply

    return SimpleNamespace(is_configured=lambda: configured, complete=complete, calls=calls)


@pytest.mark.parametrize("question", [
    "Compare the 2026 growth forecasts for South Asia and Sub-Saharan Africa.",
    "What new digital system replaced FDsys, and when was it fully deployed?",
    "What is the difference between idle and sleep power?",
    "Passports vs copies distributed in FY 2018?",
])
def test_comparison_and_multi_part_questions_need_decomposition(question):
    assert needs_decomposition(question)


@pytest.mark.parametrize("question", [
    "How many depository libraries did GPO staff visit in FY 2018?",
    "How many copies and titles did GPO distribute?",  # one fact, even with "and"
    "What is the growth outlook for South Asia in this report?",
    "How much less energy does iMac use compared to the ENERGY STAR requirement?",  # one figure vs a reference
])
def test_single_fact_questions_do_not(question):
    assert not needs_decomposition(question)


def test_splits_into_the_sub_questions_the_model_returns():
    gen = _generator('Sure:\n["What replaced FDsys?", "When was govinfo fully deployed?"]')

    queries = decompose_query(gen, "What replaced FDsys, and when was it fully deployed?", groq_model_name="g")

    assert queries == ["What replaced FDsys?", "When was govinfo fully deployed?"]
    assert gen.calls[0][1]["temperature"] == 0.0 and gen.calls[0][1]["groq_model_name"] == "g"


def test_keeps_at_most_the_limit_and_drops_duplicates():
    gen = _generator('["a?", "a?", "b?", "c?", "d?"]')
    assert decompose_query(gen, "q " * 20) == ["a?", "b?", "c?"][:MAX_SUB_QUESTIONS]


@pytest.mark.parametrize("reply", ["", "not json", '["only one part?"]', '{"a": 1}', '["ok?", 3]',
                                   '["' + "x" * 500 + '", "b?"]'])
def test_unusable_output_falls_back_to_the_original_question(reply):
    assert decompose_query(_generator(reply), "Compare A and B") == ["Compare A and B"]


def test_no_provider_means_no_call():
    gen = _generator("[]", configured=False)
    assert decompose_query(gen, "Compare A and B") == ["Compare A and B"] and gen.calls == []
