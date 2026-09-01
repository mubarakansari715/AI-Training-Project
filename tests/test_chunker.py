from app.rag.chunker import chunk_pages
from app.rag.document_loader import DocumentPage


def test_short_text_produces_a_single_chunk():
    pages = [DocumentPage(text="Short text.", source="notes.txt", page=None)]

    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_long_text_is_split_into_multiple_chunks():
    text = "This is a sentence about company policy. " * 100
    pages = [DocumentPage(text=text, source="policy.pdf", page=3)]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200


def test_metadata_is_preserved_on_every_chunk():
    text = "A" * 500
    pages = [DocumentPage(text=text, source="handbook.pdf", page=12)]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "handbook.pdf"
        assert chunk.page == 12


def test_page_with_no_page_number_preserves_none():
    pages = [DocumentPage(text="Some text.", source="readme.md", page=None)]

    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)

    assert chunks[0].page is None


def test_chunks_from_different_pages_do_not_blend():
    pages = [
        DocumentPage(text="Content only from page one.", source="doc.pdf", page=1),
        DocumentPage(text="Content only from page two.", source="doc.pdf", page=2),
    ]

    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)

    assert len(chunks) == 2
    page_one_chunk = next(c for c in chunks if c.page == 1)
    page_two_chunk = next(c for c in chunks if c.page == 2)
    assert "page two" not in page_one_chunk.text
    assert "page one" not in page_two_chunk.text


def test_chunk_id_is_deterministic():
    pages = [DocumentPage(text="A" * 500, source="handbook.pdf", page=12)]

    chunks_first_run = chunk_pages(pages, chunk_size=200, chunk_overlap=50)
    chunks_second_run = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert [c.chunk_id for c in chunks_first_run] == [
        c.chunk_id for c in chunks_second_run
    ]


def test_chunk_ids_are_unique_within_a_document():
    pages = [DocumentPage(text="B" * 1000, source="handbook.pdf", page=1)]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    pages = [DocumentPage(text=text, source="doc.txt", page=None)]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    first_tail = chunks[0].text[-20:]
    assert first_tail.strip() != ""
    assert any(word in chunks[1].text for word in first_tail.split())


def test_prefers_paragraph_boundary_over_mid_word_split():
    paragraph_one = "First paragraph. " * 10
    paragraph_two = "Second paragraph. " * 10
    text = paragraph_one.strip() + "\n\n" + paragraph_two.strip()
    pages = [DocumentPage(text=text, source="doc.txt", page=None)]

    chunks = chunk_pages(pages, chunk_size=len(paragraph_one) + 5, chunk_overlap=0)

    assert "First paragraph." in chunks[0].text
    assert not chunks[0].text.rstrip().endswith("Second paragraph")


def test_empty_pages_list_returns_no_chunks():
    assert chunk_pages([], chunk_size=800, chunk_overlap=150) == []
