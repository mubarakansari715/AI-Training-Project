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


def test_all_chunks_returns_every_stored_chunk(tmp_path):
    from app.rag.chunker import Chunk
    from app.rag.vector_store import VectorStore

    store = VectorStore(persist_dir=str(tmp_path / "c"), collection_name="all_chunks_test")
    store.add_chunks(
        [Chunk(text="a", source="d.pdf", page=3, chunk_id="c1"), Chunk(text="b", source="d.txt", page=None, chunk_id="c2")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    chunks = sorted(store.all_chunks(), key=lambda c: c["chunk_id"])

    assert chunks == [
        {"chunk_id": "c1", "text": "a", "source": "d.pdf", "page": 3, "section": None, "pages": []},
        {"chunk_id": "c2", "text": "b", "source": "d.txt", "page": None, "section": None, "pages": []},
    ]


def test_section_metadata_and_where_filter(tmp_path):
    from app.rag.chunker import Chunk
    from app.rag.vector_store import VectorStore

    store = VectorStore(persist_dir=str(tmp_path / "c"), collection_name="sections_test")
    store.add_chunks(
        [Chunk(text="sa", source="r.pdf", page=96, chunk_id="sa", section="Ch. 2 › South Asia"),
         Chunk(text="ssa", source="r.pdf", page=99, chunk_id="ssa", section="Ch. 2 › Sub-Saharan Africa"),
         Chunk(text="memo", source="m.txt", page=None, chunk_id="memo")],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )

    assert store.sections() == ["Ch. 2 › South Asia", "Ch. 2 › Sub-Saharan Africa"]
    only_ssa = store.search([1.0, 0.0], top_k=3, where={"section": "Ch. 2 › Sub-Saharan Africa"})
    assert [m["chunk_id"] for m in only_ssa] == ["ssa"]
    assert only_ssa[0]["section"] == "Ch. 2 › Sub-Saharan Africa"
    unlabelled = store.search([0.8, 0.2], top_k=1)[0]
    assert unlabelled["chunk_id"] == "memo" and unlabelled["section"] is None


def test_chunks_indexed_before_sections_existed_still_match_filters(tmp_path):
    """Older indexes have no 'section' metadata at all; the front-matter
    exclusion filter must not hide them (guards against a ChromaDB change)."""
    from app.rag.chunker import Chunk
    from app.rag.vector_store import VectorStore

    store = VectorStore(persist_dir=str(tmp_path / "c"), collection_name="legacy")
    store._collection.add(ids=["old"], embeddings=[[1.0, 0.0]], documents=["old"], metadatas=[{"source": "old.pdf", "page": 1}])
    store.add_chunks([Chunk(text="toc", source="new.pdf", page=1, chunk_id="toc", section="Contents")], [[0.9, 0.1]])

    found = [m["chunk_id"] for m in store.search([1.0, 0.0], top_k=5, where={"section": {"$nin": ["Contents"]}})]

    assert found == ["old"]


def test_chunk_pages_round_trip_through_search(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="pages_test")
    store.add_chunks([Chunk(text="spans", source="d.pdf", page=4, chunk_id="x", pages=[4, 5])], [[1.0, 0.0]])

    assert store.search([1.0, 0.0], top_k=1)[0]["pages"] == [4, 5]
    assert store.all_chunks()[0]["pages"] == [4, 5]


def test_sources_lists_each_indexed_document_once(tmp_path):
    from app.rag.chunker import Chunk
    from app.rag.vector_store import VectorStore

    store = VectorStore(persist_dir=str(tmp_path), collection_name="sources_test")
    chunks = [Chunk(text=f"t{i}", source=name, page=1, chunk_id=f"{name}-{i}")
              for i, name in enumerate(["b.pdf", "a.pdf", "b.pdf"])]
    store.add_chunks(chunks, [[0.1, 0.2, 0.3]] * 3)

    assert store.sources() == ["a.pdf", "b.pdf"]
