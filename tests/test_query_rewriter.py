from types import SimpleNamespace

import pytest

from app.rag.query_rewriter import REWRITE_PROMPT, needs_rewrite, rewrite_query, words


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


_VOCABULARY = set(words("The iMac uses 58 percent less energy than the ENERGY STAR requirements. "
                        "govinfo replaced FDsys. Depository libraries in 25 states."))
_HISTORY = [{"role": "user", "content": "How much energy does the iMac use?"}]


@pytest.mark.parametrize("question", [
    "How much less energy does the iMac use than the ENERGY STAR requirement?",  # singular of a document word
    "What replaced FDsys?",
    "What is the iMac's energy use in 2026?",  # possessive and numbers
])
def test_clear_question_needs_no_rewrite(question):
    assert needs_rewrite(question, None, _VOCABULARY) is False


def test_clear_question_with_history_but_no_follow_up_words_needs_no_rewrite():
    assert needs_rewrite("How many depository libraries are there?", _HISTORY, _VOCABULARY) is False


@pytest.mark.parametrize("question", ["wat is enrgy use of imac", "What does the iMac weigh?"])
def test_unknown_word_needs_rewrite(question):
    # A typo, or a correct word the documents never use: rewriting is the safe side.
    assert needs_rewrite(question, None, _VOCABULARY) is True


@pytest.mark.parametrize("question", ["And in 2027?", "What about govinfo?", "Is it less than that?"])
def test_follow_up_needs_rewrite_only_with_history(question):
    assert needs_rewrite(question, _HISTORY, _VOCABULARY) is True
    assert needs_rewrite(question, None, _VOCABULARY) is False
