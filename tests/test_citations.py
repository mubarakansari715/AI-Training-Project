import pytest

from app.rag.citations import cited_chunks, cited_pages
from app.rag.retriever import RetrievedChunk


def _chunk(cid, page):
    return RetrievedChunk(text="t", source="r.pdf", page=page, score=0.5, chunk_id=cid)


@pytest.mark.parametrize(
    ("answer", "pages"),
    [("166 libraries (p. 5).", {5}), ("See pages 14 and 15.", {14, 15}), ("(pp. 109–141)", {109, 141}),
     ("report, page 4, 22", {4, 22}), ("No citation here, 2018.", set())],
)
def test_cited_pages(answer, pages):
    assert cited_pages(answer) == pages


def test_cited_chunks_keeps_only_cited_pages():
    chunks = [_chunk("a", 14), _chunk("b", 26), _chunk("c", 15)]
    assert [c.chunk_id for c in cited_chunks("1,737 staff (report.pdf, page 14) and p. 15", chunks)] == ["a", "c"]


def test_refusal_without_citations_has_no_page_sources():
    chunks = [_chunk("a", 12), _chunk("b", 40)]
    assert cited_chunks("I couldn't find the FY 2025 budget in your documents.", chunks) == []


def test_chunks_without_pages_are_kept_when_no_pages_cited():
    chunks = [_chunk("txt", None), _chunk("pdf", 3)]
    assert [c.chunk_id for c in cited_chunks("24 days of leave (handbook.txt).", chunks)] == ["txt"]
