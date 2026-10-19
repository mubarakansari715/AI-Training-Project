import io

import docx
import pytest

from app.rag.document_loader import DocumentParsingError, parse_document


def _make_minimal_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing `text`, without any
    external PDF-generation library."""
    content = f"BT /F1 24 Tf 10 100 Td ({text}) Tj ET".encode()

    objects = [
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>"
        b"/MediaBox[0 0 200 200]/Contents 5 0 R>>endobj\n",
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        (
            f"5 0 obj<</Length {len(content)}>>\nstream\n".encode()
            + content
            + b"\nendstream\nendobj\n"
        ),
    ]

    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj

    xref_offset = len(pdf)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode()

    pdf += xref
    pdf += (
        f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode()
    return pdf


def _make_docx(paragraphs: list[str]) -> bytes:
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_parse_pdf_extracts_text_and_page_number():
    pdf_bytes = _make_minimal_pdf("Hello World")

    pages = parse_document("sample.pdf", pdf_bytes)

    assert len(pages) == 1
    assert pages[0].source == "sample.pdf"
    assert pages[0].page == 1
    assert "Hello World" in pages[0].text


def test_parse_docx_extracts_text_with_no_page_number():
    docx_bytes = _make_docx(["Annual leave is 24 days per year."])

    pages = parse_document("policy.docx", docx_bytes)

    assert len(pages) == 1
    assert pages[0].source == "policy.docx"
    assert pages[0].page is None
    assert "Annual leave" in pages[0].text


def test_parse_txt_extracts_text():
    pages = parse_document("notes.txt", b"Some plain text content.")

    assert len(pages) == 1
    assert pages[0].page is None
    assert pages[0].text == "Some plain text content."


def test_parse_markdown_extracts_text():
    pages = parse_document("readme.md", b"# Title\n\nSome content.")

    assert len(pages) == 1
    assert "Title" in pages[0].text


def test_parse_empty_file_raises():
    with pytest.raises(DocumentParsingError):
        parse_document("empty.txt", b"")


def test_parse_unsupported_extension_raises():
    with pytest.raises(DocumentParsingError):
        parse_document("archive.zip", b"some bytes")


def test_parse_corrupted_pdf_raises():
    with pytest.raises(DocumentParsingError):
        parse_document("broken.pdf", b"%PDF-1.4 this is not a real pdf")


def test_parse_document_with_no_extractable_text_returns_empty_list():
    pages = parse_document("blank.txt", b"   \n\n   ")

    assert pages == []


def test_original_filename_is_preserved():
    pages = parse_document("employee_handbook.pdf", _make_minimal_pdf("Policy text"))

    assert pages[0].source == "employee_handbook.pdf"


def test_measured_bar_charts_are_indexed_as_their_own_passages():
    from tests.test_chart_measure import _stacked_chart_page

    doc, _ = _stacked_chart_page()
    pages = parse_document("risk.pdf", doc.tobytes())

    [chart] = [p for p in pages if p.element]
    assert chart.page == 1 and chart.text.startswith("[Chart on page 1 (values measured from the drawing)]")
    assert "2015: High risk ≈ 30" in chart.text


def test_a_pdf_over_the_page_limit_is_turned_down(monkeypatch):
    from dataclasses import replace

    import pymupdf
    import pytest

    from app.rag import document_loader
    from app.rag.document_loader import DocumentParsingError, parse_document

    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page().insert_text((50, 60), "Text.", fontsize=10)
    monkeypatch.setattr(document_loader, "settings", replace(document_loader.settings, max_pdf_pages=2))

    with pytest.raises(DocumentParsingError, match="has 3 pages; the limit is 2"):
        parse_document("big.pdf", doc.tobytes())
