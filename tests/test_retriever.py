import pytest

from app.rag.chunker import Chunk
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class FakeEmbeddingService:
    """Deterministic stand-in for EmbeddingService so retriever tests
    don't depend on the real ML model."""

    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self._vectors_by_text = vectors_by_text

    def embed_query(self, text: str) -> list[float]:
        return self._vectors_by_text[text]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[t] for t in texts]


@pytest.fixture
def populated_store(tmp_path):
    store = VectorStore(
        persist_dir=str(tmp_path / "chroma"), collection_name="retriever_test"
    )
    chunks = [
        Chunk(
            text="Employees get 24 days of annual leave.",
            source="handbook.pdf",
            page=12,
            chunk_id="c1",
        ),
        Chunk(
            text="The office is located downtown.",
            source="handbook.pdf",
            page=3,
            chunk_id="c2",
        ),
    ]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return store


def test_retrieve_returns_most_similar_chunk_first(populated_store):
    embeddings = FakeEmbeddingService(
        {"What is the leave policy?": [1.0, 0.0, 0.0]}
    )
    retriever = Retriever(populated_store, embeddings, top_k=2)

    results = retriever.retrieve("What is the leave policy?")

    assert results[0].source == "handbook.pdf"
    assert results[0].page == 12
    assert "annual leave" in results[0].text


def test_retrieve_empty_question_returns_no_results(populated_store):
    embeddings = FakeEmbeddingService({})
    retriever = Retriever(populated_store, embeddings)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retrieve_respects_top_k(populated_store):
    embeddings = FakeEmbeddingService({"question": [1.0, 0.0, 0.0]})
    retriever = Retriever(populated_store, embeddings, top_k=1)

    results = retriever.retrieve("question")

    assert len(results) == 1


def test_score_threshold_filters_out_dissimilar_chunks(populated_store):
    embeddings = FakeEmbeddingService({"question": [1.0, 0.0, 0.0]})
    retriever = Retriever(populated_store, embeddings, top_k=2, score_threshold=0.99)

    results = retriever.retrieve("question")

    # Only the exact-match vector [1,0,0] clears a 0.99 similarity bar.
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_retrieve_on_empty_store_returns_no_results(tmp_path):
    empty_store = VectorStore(
        persist_dir=str(tmp_path / "chroma_empty"), collection_name="empty_test"
    )
    embeddings = FakeEmbeddingService({"question": [1.0, 0.0, 0.0]})
    retriever = Retriever(empty_store, embeddings)

    assert retriever.retrieve("question") == []
