import pytest

from app.rag.chunker import Chunk
from app.rag.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(
        persist_dir=str(tmp_path / "chroma"), collection_name="test_collection"
    )


def _chunk(chunk_id: str, text: str, source: str = "doc.txt", page=None) -> Chunk:
    return Chunk(text=text, source=source, page=page, chunk_id=chunk_id)


def test_add_and_search_returns_the_closest_match(store):
    chunks = [
        _chunk("c1", "Employees get 24 days of annual leave."),
        _chunk("c2", "The office is located in downtown."),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    added = store.add_chunks(chunks, embeddings)
    assert added == 2

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["text"] == "Employees get 24 days of annual leave."


def test_search_on_empty_store_returns_no_results(store):
    assert store.search([1.0, 0.0, 0.0]) == []


def test_metadata_source_and_page_round_trip(store):
    chunks = [_chunk("c1", "Some text.", source="handbook.pdf", page=12)]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0])

    assert results[0]["source"] == "handbook.pdf"
    assert results[0]["page"] == 12


def test_page_none_round_trips_as_none(store):
    chunks = [_chunk("c1", "Some text.", source="notes.txt", page=None)]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0]])

    results = store.search([1.0, 0.0, 0.0])

    assert results[0]["page"] is None


def test_adding_the_same_chunk_id_twice_does_not_duplicate(store):
    chunks = [_chunk("c1", "Duplicate text.")]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0]])
    second_add = store.add_chunks(chunks, [[1.0, 0.0, 0.0]])

    assert second_add == 0
    assert store.count() == 1


def test_clear_removes_all_chunks(store):
    store.add_chunks([_chunk("c1", "Some text.")], [[1.0, 0.0, 0.0]])
    assert store.count() == 1

    store.clear()

    assert store.count() == 0


def test_add_chunks_empty_list_is_a_noop(store):
    assert store.add_chunks([], []) == 0
    assert store.count() == 0
