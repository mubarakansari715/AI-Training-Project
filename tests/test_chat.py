from app.ui.chat import _find_cached_answer, _normalize_question


def test_normalize_question_collapses_whitespace_and_case():
    assert _normalize_question("  What Is   the Leave Policy?  ") == "what is the leave policy?"


def _turn(role, content):
    return {"role": role, "content": content}


def test_find_cached_answer_returns_prior_reply_for_repeated_question():
    history = [
        _turn("user", "What is the leave policy?"),
        {"role": "assistant", "content": "24 days.", "sources": [], "source": "document"},
        _turn("user", "What is the leave policy?"),  # current turn, no reply yet
    ]

    cached = _find_cached_answer(history, "What is the leave policy?")

    assert cached is not None
    assert cached["content"] == "24 days."


def test_find_cached_answer_is_case_and_whitespace_insensitive():
    history = [
        _turn("user", "What is the leave policy?"),
        {"role": "assistant", "content": "24 days.", "sources": [], "source": "document"},
        _turn("user", "  what   is the leave policy?  "),
    ]

    cached = _find_cached_answer(history, "  what   is the leave policy?  ")

    assert cached is not None
    assert cached["content"] == "24 days."


def test_find_cached_answer_returns_none_for_new_question():
    history = [
        _turn("user", "What is the leave policy?"),
        {"role": "assistant", "content": "24 days.", "sources": [], "source": "document"},
        _turn("user", "What is the sick leave policy?"),
    ]

    assert _find_cached_answer(history, "What is the sick leave policy?") is None


def test_find_cached_answer_returns_none_on_first_ask():
    history = [_turn("user", "What is the leave policy?")]

    assert _find_cached_answer(history, "What is the leave policy?") is None


def test_find_cached_answer_uses_most_recent_matching_reply():
    history = [
        _turn("user", "What is the leave policy?"),
        {"role": "assistant", "content": "First answer.", "sources": [], "source": "document"},
        _turn("user", "What is the leave policy?"),
        {"role": "assistant", "content": "Second answer.", "sources": [], "source": "document"},
        _turn("user", "What is the leave policy?"),
    ]

    cached = _find_cached_answer(history, "What is the leave policy?")

    assert cached["content"] == "Second answer."
