from types import SimpleNamespace

import pytest

from app.rag.query_router import MAX_SECTIONS, llm_route

LABELS = ["Ch. 2 Regional Outlooks › South Asia", "Ch. 2 Regional Outlooks › Europe and Central Asia",
          "Ch. 3 Sovereign Debt", "Executive Summary"]


def _generator(reply, configured=True):
    calls = []
    return SimpleNamespace(is_configured=lambda: configured,
                           complete=lambda prompt, **kw: calls.append((prompt, kw)) or reply, calls=calls)


def test_picks_only_exact_labels_from_the_list():
    gen = _generator('Here: ["Ch. 2 Regional Outlooks › South Asia", "India", "Executive Summary"]')

    assert llm_route(gen, "What is India's growth forecast?", LABELS, groq_model_name="fast") == [
        "Ch. 2 Regional Outlooks › South Asia", "Executive Summary"]
    prompt, kwargs = gen.calls[0]
    assert "- Ch. 3 Sovereign Debt" in prompt and kwargs == {"groq_model_name": "fast", "temperature": 0.0}


def test_keeps_at_most_the_limit():
    gen = _generator(str(LABELS).replace("'", '"'))
    assert len(llm_route(gen, "q", LABELS)) == MAX_SECTIONS


@pytest.mark.parametrize("reply", ["", "no idea", "[]", '{"a": 1}', "[1, 2]"])
def test_unusable_or_empty_reply_means_no_filter(reply):
    assert llm_route(_generator(reply), "q", LABELS) == []


def test_no_provider_or_no_sections_means_no_call():
    gen = _generator("[]", configured=False)
    assert llm_route(gen, "q", LABELS) == [] and gen.calls == []
    assert llm_route(_generator("[]"), "q", []) == []
