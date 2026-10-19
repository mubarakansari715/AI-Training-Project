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


def test_ready_note_summarises_an_upload():
    from app.ui.chat import _ready_note

    gep = {"stats": {"pages": 200, "passages": 1194, "charts": 101}, "added": 1194}
    assert _ready_note(gep) == "200 pages, 1,194 passages, 101 charts measured"
    notes = {"stats": {"pages": 0, "passages": 3, "charts": 0}, "added": 0}
    assert _ready_note(notes) == "3 passages (already indexed, unchanged)"


def test_milestone_notes_say_text_first_then_charts():
    from app.services.ingest_jobs import FAILED, READY, SEARCHABLE, IngestJob
    from app.ui.chat import _milestone_notes

    job = IngestJob(id="1", name="gep.pdf", status=SEARCHABLE, stats={"pages": 200, "passages": 1194}, added=1194,
                    charts_pending=True)
    (first,) = _milestone_notes(job, None)
    assert first.startswith("📄 **gep.pdf** is ready: 200 pages, 1,194 passages.") and "still being read" in first

    job.status, job.visual_passages = READY, 83
    assert _milestone_notes(job, SEARCHABLE) == ["📊 Charts and tables from **gep.pdf** added (83 passages)."]

    txt = IngestJob(id="2", name="notes.txt", status=READY, stats={"passages": 3}, added=3)
    assert _milestone_notes(txt, None) == ["📄 **notes.txt** is ready: 3 passages."]

    failed = IngestJob(id="3", name="bad.pdf", status=FAILED, error="The PDF may be corrupted.")
    assert _milestone_notes(failed, None) == ["⚠️ **bad.pdf** could not be added: The PDF may be corrupted."]
