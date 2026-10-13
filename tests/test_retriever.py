import pytest

from app.rag.chunker import Chunk
from app.rag.retriever import Retriever, is_summary_request
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


@pytest.mark.parametrize(
    "question",
    ["Summarize this video", "What is this report about?", "Give me an overview of the PDF",
     "key points of the uploaded file"],
)
def test_summary_requests_are_detected(question):
    assert is_summary_request(question) is True


@pytest.mark.parametrize(
    "question",
    ["Summarize the French Revolution", "How many leave days do I get?", "What is this about?"],
)
def test_other_questions_are_not_summary_requests(question):
    assert is_summary_request(question) is False


def test_summary_request_skips_score_threshold(populated_store):
    question = "Summarize this document"
    embeddings = FakeEmbeddingService({question: [1.0, 0.0, 0.0]})
    retriever = Retriever(populated_store, embeddings, top_k=1, score_threshold=0.5)

    results = retriever.retrieve(question)

    # c2 is orthogonal (score 0) but still included, and top_k was widened.
    assert [r.chunk_id for r in results] == ["c1", "c2"]


class _ReverseReranker:
    """Fake reranker: reverses candidate order, records what it was given."""

    def __init__(self):
        self.calls = []

    def rerank(self, question, chunks, top_k):
        self.calls.append((question, [c.chunk_id for c in chunks], top_k))
        return list(reversed(chunks))[:top_k]


def test_reranker_reorders_a_wider_candidate_set(populated_store):
    question = "leave"
    embeddings = FakeEmbeddingService({question: [0.9, 0.4, 0.0]})
    reranker = _ReverseReranker()
    retriever = Retriever(populated_store, embeddings, top_k=1, score_threshold=0.0,
                          reranker=reranker, rerank_candidates=2)

    results = retriever.retrieve(question)

    assert reranker.calls == [("leave", ["c1", "c2"], 1)]  # fetched 2 candidates for top_k=1
    assert [r.chunk_id for r in results] == ["c2"]


def test_threshold_is_applied_before_reranking(populated_store):
    question = "leave"
    embeddings = FakeEmbeddingService({question: [1.0, 0.0, 0.0]})  # c2 scores 0
    reranker = _ReverseReranker()
    retriever = Retriever(populated_store, embeddings, top_k=2, score_threshold=0.5,
                          reranker=reranker, rerank_candidates=5)

    retriever.retrieve(question)

    assert reranker.calls[0][1] == ["c1"]  # the below-threshold chunk never reaches the reranker


def test_cross_encoder_reranker_puts_the_answering_chunk_first():
    from app.rag.reranker import Reranker
    from app.rag.retriever import RetrievedChunk

    chunks = [RetrievedChunk(text=t, source="d", page=i, score=0.5, chunk_id=str(i)) for i, t in enumerate([
        "The office is located downtown near the river.",
        "Titles cataloged 34,367. Publications sold 334,798. Passports produced 15,901,600.",
    ])]

    ranked = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2").rerank("How many publications were sold?", chunks, 2)

    assert ranked[0].chunk_id == "1"
