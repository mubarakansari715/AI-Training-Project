from types import SimpleNamespace

import pytest

from app.rag.query_rewriter import REWRITE_PROMPT, rewrite_query


def _generator(reply: str = "", configured: bool = True):
    calls = []

    def complete(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return reply

    return SimpleNamespace(is_configured=lambda: configured, complete=complete, calls=calls)


def test_returns_cleaned_rewrite_with_zero_temperature():
    gen = _generator("What is my vitamin D level?\n")

    assert rewrite_query(gen, "wat is my vitamn d levl") == "What is my vitamin D level?"
    assert gen.calls[0][1]["temperature"] == 0.0


@pytest.mark.parametrize(
    "reply",
    ['"What is my vitamin D level?"', "Query: What is my vitamin D level?", "\n\n  What is my vitamin D level?  \nextra"],
)
def test_strips_quotes_labels_and_extra_lines(reply):
    assert rewrite_query(_generator(reply), "wat is my vitamn d") == "What is my vitamin D level?"


@pytest.mark.parametrize("reply", ["", "   ", "x" * 500])
def test_falls_back_to_original_on_empty_or_runaway_output(reply):
    assert rewrite_query(_generator(reply), "wat is my vitamn d") == "wat is my vitamn d"


def test_unconfigured_generator_returns_question_without_calling():
    gen = _generator("ignored", configured=False)

    assert rewrite_query(gen, "and my cholesterol?") == "and my cholesterol?"
    assert gen.calls == []


def test_prompt_includes_recent_truncated_history_and_question():
    gen = _generator("What are my cholesterol results?")
    history = [{"role": "user", "content": f"old {i}"} for i in range(5)] + [
        {"role": "assistant", "content": "A" * 1000},
        {"role": "user", "content": "What is my vitamin D?"},
    ]

    rewrite_query(gen, "and my cholesterol?", history, model_name="gemini-x")

    prompt, kwargs = gen.calls[0]
    assert "old 0" not in prompt  # only the last 6 messages
    assert "assistant: " + "A" * 300 + "\n" in prompt  # each truncated
    assert "user: What is my vitamin D?" in prompt
    assert prompt.endswith("Latest message: and my cholesterol?")
    assert kwargs["model_name"] == "gemini-x"


def test_prompt_without_history_says_none():
    gen = _generator("Who built you?")
    rewrite_query(gen, "whu bilt yu")
    assert "Conversation so far:\n(none)" in gen.calls[0][0]
    assert REWRITE_PROMPT.startswith("Prepare the user's latest message")
