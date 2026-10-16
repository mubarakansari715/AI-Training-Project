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


@pytest.fixture
def sectioned_store(tmp_path):
    """Three sections whose chunks are all equally similar to the query, plus front matter."""
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="sectioned")
    specs = [
        ("toc", "Contents", 13), ("es", "Executive Summary", 19),
        ("sa1", "Ch. 2 Regional Outlooks › South Asia", 96), ("sa2", "Ch. 2 Regional Outlooks › South Asia", 97),
        ("sa3", "Ch. 2 Regional Outlooks › South Asia", 98),
        ("ssa1", "Ch. 2 Regional Outlooks › Sub-Saharan Africa", 99),
        ("ssa2", "Ch. 2 Regional Outlooks › Sub-Saharan Africa", 100),
    ]
    store.add_chunks(
        [Chunk(text=f"text {cid}", source="r.pdf", page=page, chunk_id=cid, section=section) for cid, section, page in specs],
        # The table of contents is the closest match, as titles match every topic.
        [[1.0, 0.0, 0.0]] + [[0.95, 0.05, 0.0]] * 6,
    )
    return store


def _retriever(store, question, top_k=4):
    return Retriever(store, FakeEmbeddingService({question: [1.0, 0.0, 0.0]}), top_k=top_k, score_threshold=0.0)


def test_question_naming_a_section_searches_only_that_section(sectioned_store):
    q = "According to the executive summary, what are the risks?"
    assert [r.chunk_id for r in _retriever(sectioned_store, q).retrieve(q)] == ["es"]


def test_front_matter_is_excluded_when_no_section_is_named(sectioned_store):
    q = "What is the growth outlook?"
    ids = [r.chunk_id for r in _retriever(sectioned_store, q).retrieve(q)]
    assert "toc" not in ids and ids


def test_several_named_sections_each_get_a_share(sectioned_store):
    q = "Compare South Asia and Sub-Saharan Africa."
    results = _retriever(sectioned_store, q, top_k=4).retrieve(q)
    sections = {r.section for r in results}
    assert sections == {"Ch. 2 Regional Outlooks › South Asia", "Ch. 2 Regional Outlooks › Sub-Saharan Africa"}
    # Two sections: each gets a full top_k (South Asia has 3 chunks, Africa 2).
    assert len(results) == 5


def test_retrieved_chunks_carry_their_section(sectioned_store):
    q = "What does the report say about South Asia?"
    assert all(r.section == "Ch. 2 Regional Outlooks › South Asia" for r in _retriever(sectioned_store, q).retrieve(q))


def test_single_named_section_sends_whole_pages_of_its_best_hits(tmp_path):
    """Small-to-big: a section's headline facts often sit on the same page as
    the best hit but in a chunk that scored lower, so the whole page goes in."""
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="pages")
    sa = "Ch. 2 Regional Outlooks › South Asia"
    specs = [  # (chunk_id, page, embedding): only the p. 96 middle chunk matches the query
        ("r.pdf:96:0:a", 96, [0.0, 1.0, 0.0]), ("r.pdf:96:1:b", 96, [1.0, 0.0, 0.0]),
        ("r.pdf:96:2:c", 96, [0.0, 1.0, 0.0]), ("r.pdf:97:0:d", 97, [0.0, 1.0, 0.0]),
    ]
    store.add_chunks(
        [Chunk(text=f"text {cid}", source="r.pdf", page=page, chunk_id=cid, section=sa) for cid, page, _ in specs],
        [emb for _, _, emb in specs],
    )
    q = "What is the growth outlook for South Asia?"
    retriever = Retriever(store, FakeEmbeddingService({q: [1.0, 0.0, 0.0]}), top_k=1, score_threshold=0.5)

    # The whole of p. 96 as one passage in reading order; p. 97 had no hit, so it stays out.
    results = retriever.retrieve(q)
    assert [(r.chunk_id, r.page) for r in results] == [("r.pdf:96:0:a", 96)]
    assert results[0].text == "text r.pdf:96:0:a\ntext r.pdf:96:1:b\ntext r.pdf:96:2:c"


def test_whole_page_passage_drops_chunk_overlap_and_repeated_section_labels(tmp_path):
    """Overlapping fragments made the model treat the continuation of a
    sentence as a repeat and skip it (GEP executive summary risks)."""
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="overlap")
    texts = [
        "[Executive Summary]\nRisks are skewed to the downside. If energy\nsupply disruptions prove more severe than",
        "[Executive Summary]\nsupply disruptions prove more severe than\nassumed, growth could fall to 1.3 percent.",
    ]
    store.add_chunks(
        [Chunk(text=t, source="r.pdf", page=19, chunk_id=f"r.pdf:19:{i}:x{i}", section="Executive Summary")
         for i, t in enumerate(texts)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    q = "According to the executive summary, what are the risks?"
    retriever = Retriever(store, FakeEmbeddingService({q: [1.0, 0.0, 0.0]}), top_k=1, score_threshold=0.5)
    [passage] = retriever.retrieve(q)
    assert passage.text == ("[Executive Summary]\nRisks are skewed to the downside. If energy\n"
                            "supply disruptions prove more severe than\nassumed, growth could fall to 1.3 percent.")


def test_page_expansion_is_off_without_a_named_section(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="plain")
    store.add_chunks(
        [Chunk(text="a", source="r.pdf", page=1, chunk_id="r.pdf:1:0:a"),
         Chunk(text="b", source="r.pdf", page=1, chunk_id="r.pdf:1:1:b")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    q = "question"
    retriever = Retriever(store, FakeEmbeddingService({q: [1.0, 0.0, 0.0]}), top_k=1, score_threshold=0.5)
    assert [r.chunk_id for r in retriever.retrieve(q)] == ["r.pdf:1:0:a"]


def _footnoted_store(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="footnotes")
    specs = [  # (chunk_id, page, text, embedding): only the claim matches the query
        ("i.pdf:4:0:a", 4, "iMac contains more than 15 percent recycled content.[1]", [1.0, 0.0, 0.0]),
        ("i.pdf:4:1:b", 4, "Plastic. At least 50 percent recycled plastic in 13 components.", [0.0, 1.0, 0.0]),
        ("i.pdf:4:2:c", 4, "[Footnote 1, referenced on page 4] Mass of certified recycled material.", [0.0, 0.0, 1.0]),
        ("i.pdf:4:3:d", 4, "[Footnote 2, referenced on page 4] Share of renewable electricity.", [0.0, 0.0, 1.0]),
        ("i.pdf:1:0:e", 1, "[Footnote 1, referenced on page 1] Mass of certified recycled material.", [0.0, 0.0, 1.0]),
    ]
    store.add_chunks(
        [Chunk(text=text, source="i.pdf", page=page, chunk_id=cid) for cid, page, text, _ in specs],
        [emb for *_, emb in specs],
    )
    return store


def test_footnote_referenced_by_a_retrieved_chunk_is_attached(tmp_path):
    q = "What does the 15 percent recycled content claim measure?"
    retriever = Retriever(_footnoted_store(tmp_path), FakeEmbeddingService({q: [1.0, 0.0, 0.0]}),
                          top_k=1, score_threshold=0.5)

    # Footnote 1 of the same page only: not note 2 (unreferenced), not page 1's copy.
    assert [r.chunk_id for r in retriever.retrieve(q)] == ["i.pdf:4:0:a", "i.pdf:4:2:c"]


def test_no_footnote_is_attached_without_a_marker(tmp_path):
    q = "How much recycled plastic?"
    retriever = Retriever(_footnoted_store(tmp_path), FakeEmbeddingService({q: [0.0, 1.0, 0.0]}),
                          top_k=1, score_threshold=0.5)

    assert [r.chunk_id for r in retriever.retrieve(q)] == ["i.pdf:4:1:b"]


def test_a_footnote_is_attached_once_even_when_several_pages_reference_it(tmp_path):
    store = _footnoted_store(tmp_path)
    store.add_chunks(  # page 1 makes the same claim, also marked [1]
        [Chunk(text="More than 15% recycled content [1]", source="i.pdf", page=1, chunk_id="i.pdf:1:1:f")],
        [[0.99, 0.0, 0.14]],
    )
    q = "What does the 15 percent recycled content claim measure?"
    retriever = Retriever(store, FakeEmbeddingService({q: [1.0, 0.0, 0.0]}), top_k=2, score_threshold=0.5)

    ids = [r.chunk_id for r in retriever.retrieve(q)]
    assert ids[:2] == ["i.pdf:4:0:a", "i.pdf:1:1:f"]
    assert sum("Footnote 1," in t for t in (c.text for c in retriever.retrieve(q))) == 1
