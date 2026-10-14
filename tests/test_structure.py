from app.rag.structure import (
    SEPARATOR,
    clean_title,
    compact,
    detect_sections,
    is_front_matter,
    sections_mentioned,
)

BODY = " Lorem ipsum dolor sit amet. " * 12  # makes a page "non-empty" (>200 chars)


def _report():
    """A small report shaped like the World Bank one: front matter, an
    executive summary, chapters with running headers, a box, regional
    sections and an annex — with one bookmark pointing a page too early."""
    pages = {
        1: "Contents Chapter 1 Global Outlook ...... 1" + BODY,
        2: "Executive Summary The global economy is facing a shock." + BODY,
        3: "CHAPTER 1 GLOBAL OUTLOOK",
        4: "C H A PT ER 1 G L O B AL EC ON O M I C PR O SPEC T S | J U N E 2 0 2 6 3 Global growth slows." + BODY,
        5: "C H A PT ER 1 G L O B AL EC ON O M I C PR O SPEC T S 4 BOX 1.1 How much will AI affect growth?" + BODY,
        6: "C H A PT ER 1 G L O B AL EC ON O M I C PR O SPEC T S 5 Text continues; see box 1.1 for AI." + BODY,
        7: "",  # blank separator page
        8: "CHAPTER 2 REGIONAL OUTLOOKS",
        9: "C H A PT ER 2 G L O B AL 8 TABLE 2.8 Middle East forecasts" + BODY,
        10: "C H A PT ER 2 G L O B AL 9 South Asia Recent developments Growth in SAR is expected to soften." + BODY,
        11: "C H A PT ER 2 G L O B AL 10 Sub-Saharan Africa Recent developments Growth in SSA firmed." + BODY,
        12: "STATISTICAL APPENDIX",
        13: "ST A T I ST I CA L A PPEN DI X 12 Real GDP growth South Asia 6.6 6.8" + BODY,
    }
    outline = [
        ("Contents", 1), ("Executive Summary", 2), ("Ch. 1 Global Outlook", 3),
        ("Box 1.1 How much will AI affect growth?", 5), ("Ch. 2 Regional Outlooks", 8),
        ("-- South Asia", 9),  # bookmark one page early, like the real report
        ("-- Sub-Saharan Africa", 11), ("Statistical Annex", 12),
    ]
    return pages, outline


def test_detects_chapters_sections_boxes_and_front_matter():
    labels = detect_sections(*_report())
    ch1, ch2 = "Ch. 1 Global Outlook", "Ch. 2 Regional Outlooks"
    assert labels[1] == "Contents"
    assert labels[2] == "Executive Summary"
    assert labels[3] == labels[4] == labels[6] == ch1
    assert labels[5] == ch1 + SEPARATOR + "Box 1.1 How much will AI affect growth?"
    assert labels[7] == ch1  # blank page inherits the page before
    assert labels[8] == labels[9] == ch2  # p9 is the table before South Asia starts
    assert labels[10] == ch2 + SEPARATOR + "South Asia"  # bookmark relocated to its heading
    assert labels[11] == ch2 + SEPARATOR + "Sub-Saharan Africa"
    assert labels[12] == labels[13] == "Statistical Annex"


def test_lowercase_box_reference_in_body_text_is_not_a_box():
    labels = detect_sections(*_report())
    assert "Box" not in labels[6]


def test_documents_without_structure_get_no_labels():
    pages = {1: "Just a memo." + BODY, 2: "More text." + BODY}
    assert detect_sections(pages, []) == {}


def test_running_headers_alone_give_chapters():
    pages = {1: "C H A PT ER 3 Some header 87 Debt rose." + BODY}
    assert detect_sections(pages, []) == {1: "Chapter 3"}


def test_helpers():
    assert compact("C H A PT ER 1") == "chapter1"
    assert clean_title("Middle East, North Africa, -- Afghanistan, and Pakistan") == "Middle East, North Africa, Afghanistan, and Pakistan"
    assert is_front_matter("Contents") and is_front_matter("Summary of Contents")
    assert not is_front_matter("Ch. 2 Regional Outlooks › South Asia") and not is_front_matter(None)


LABELS = [
    "Executive Summary",
    "Ch. 1 Global Outlook",
    "Ch. 1 Global Outlook › Box 1.1 How much will AI affect global growth?",
    "Ch. 2 Regional Outlooks › South Asia",
    "Ch. 2 Regional Outlooks › Sub-Saharan Africa",
    "Ch. 3 A Rising Challenge: Sovereign Debt",
    "Ch. 3 A Rising Challenge: Sovereign Debt › Box 3.1 The costs of hidden debt",
]


def test_sections_mentioned_groups_by_what_the_question_names():
    assert sections_mentioned("According to the executive summary, what are the risks?", LABELS) == [["Executive Summary"]]
    assert sections_mentioned("What does Chapter 3 say about debt?", LABELS) == [
        ["Ch. 3 A Rising Challenge: Sovereign Debt",
         "Ch. 3 A Rising Challenge: Sovereign Debt › Box 3.1 The costs of hidden debt"]]
    assert sections_mentioned("Compare South Asia and Sub-Saharan Africa.", LABELS) == [
        ["Ch. 2 Regional Outlooks › South Asia"], ["Ch. 2 Regional Outlooks › Sub-Saharan Africa"]]
    assert sections_mentioned("What does Box 1.1 say about AI?", LABELS) == [
        ["Ch. 1 Global Outlook › Box 1.1 How much will AI affect global growth?"]]


def test_sections_mentioned_ignores_unnamed_and_near_misses():
    assert sections_mentioned("What is the global growth forecast for 2026?", LABELS) == []
    assert sections_mentioned("What does Box 1.10 say?", LABELS) == []
    assert sections_mentioned("Chapter 31 notes", LABELS) == []


def test_world_bank_report_sections_end_to_end():
    """The real 200-page report: bookmarks are off by one for South Asia, the
    running headers use spaced letters, and boxes continue over several pages."""
    from pathlib import Path

    import pytest

    from app.rag.document_loader import parse_document

    pdf = Path(__file__).resolve().parents[1] / "notebooks" / "GEP-Jun-2026.pdf"
    if not pdf.exists():
        pytest.skip("World Bank report not available")
    sections = {p.page: p.section for p in parse_document(pdf.name, pdf.read_bytes())}

    assert sections[19] == "Executive Summary"
    assert sections[95] == "Ch. 2 Regional Outlooks › Middle East, North Africa, Afghanistan, and Pakistan"
    assert sections[96] == sections[98] == "Ch. 2 Regional Outlooks › South Asia"
    assert sections[99] == "Ch. 2 Regional Outlooks › Sub-Saharan Africa"
    assert all("Box 1.1" in sections[p] for p in range(30, 36))
    assert "Box" not in sections[36]
    assert sections[109].startswith("Ch. 3 A Rising Challenge")
    assert sections[189] == "Statistical Annex"
    assert sections[13] == "Contents"
