from app.rag.text_cleaner import clean_text


def test_collapses_repeated_spaces():
    assert clean_text("Hello    world") == "Hello world"


def test_collapses_repeated_blank_lines():
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    assert clean_text(text) == "Paragraph one.\n\nParagraph two."


def test_preserves_single_paragraph_break():
    text = "Paragraph one.\n\nParagraph two."
    assert clean_text(text) == text


def test_rejoins_hyphenated_word_broken_across_line():
    text = "This is an exam-\nple of a broken word."
    assert clean_text(text) == "This is an example of a broken word."


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("   \n  Hello world  \n   ") == "Hello world"


def test_removes_control_characters():
    text = "Hello\x00 world\x0b!"
    assert clean_text(text) == "Hello world!"


def test_normalizes_windows_and_mac_line_endings():
    assert clean_text("Line one\r\nLine two\rLine three") == "Line one\nLine two\nLine three"


def test_strips_whitespace_only_lines():
    text = "Paragraph one.\n   \n\nParagraph two."
    assert clean_text(text) == "Paragraph one.\n\nParagraph two."


def test_empty_string_returns_empty_string():
    assert clean_text("") == ""


def test_does_not_alter_already_clean_text():
    text = "This is already clean text.\n\nWith one paragraph break."
    assert clean_text(text) == text


def test_does_not_merge_words_that_are_not_hyphen_broken():
    text = "Well-known facts about the annual leave policy."
    assert clean_text(text) == text
