import pytest

from app.rag.chunker import Chunk
from app.rag.retriever import RetrievedChunk, Retriever, is_summary_request
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

    def rerank(self, question, chunks, top_k, min_score=None, margin=None):
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
def test_relevance_cutoff_drops_low_scored_chunks_but_keeps_the_best_two():
    from app.rag.reranker import select

    chunks = [RetrievedChunk(text=c, source="s", page=1, score=0.5, chunk_id=c) for c in "abcde"]
    scored = list(zip([3.0, -4.0, 1.0, -2.0, 0.5], chunks))

    assert [c.chunk_id for c in select(scored, 5, None)] == ["a", "c", "e", "d", "b"]
    assert [c.chunk_id for c in select(scored, 5, 0.0)] == ["a", "c", "e"]
    assert [c.chunk_id for c in select(scored, 5, 10.0)] == ["a", "c"]  # best two always stay
    assert [c.chunk_id for c in select(scored, 2, 0.0)] == ["a", "c"]


def test_relative_margin_follows_each_questions_own_score_range():
    from app.rag.reranker import select

    chunks = [RetrievedChunk(text=c, source="s", page=1, score=0.5, chunk_id=c) for c in "abcde"]
    high = list(zip([9.0, 8.5, 8.0, 4.0, 3.0], chunks))  # one question scores everything high
    low = list(zip([-1.0, -1.5, -2.0, -6.0, -7.0], chunks))  # another scores everything low

    assert [c.chunk_id for c in select(high, 5, None, margin=2.0)] == ["a", "b", "c"]
    assert [c.chunk_id for c in select(low, 5, None, margin=2.0)] == ["a", "b", "c"]
    assert [c.chunk_id for c in select(low, 5, 0.0)] == ["a", "b"]  # a fixed cutoff over-drops here
    assert [c.chunk_id for c in select(high, 5, None, margin=0.1)] == ["a", "b"]  # best two always stay


def test_cutoff_is_passed_to_the_reranker_except_for_summaries(populated_store):
    embeddings = FakeEmbeddingService({"leave": [0.9, 0.4, 0.0], "summarize this document": [0.9, 0.4, 0.0]})
    seen = []

    class _Recorder(_ReverseReranker):
        def rerank(self, question, chunks, top_k, min_score=None, margin=None):
            seen.append((min_score, margin))
            return chunks[:1]

    retriever = Retriever(populated_store, embeddings, top_k=2, score_threshold=0.0, reranker=_Recorder(),
                          hybrid_search=False)
    retriever.rerank_min_score, retriever.rerank_margin = 0.5, 3.0
    retriever.retrieve("leave")
    assert retriever.last_stats["compressed"] == 1  # 2 candidates, 1 kept
    retriever.retrieve("summarize this document")

    assert seen == [(0.5, 3.0), (None, None)]


def _exact_term_store(tmp_path):
    """The chunk with the exact figure is far from the question's embedding."""
    store = VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="hybrid_test")
    chunks = [
        Chunk(text="Library services reached many readers this year.", source="r.pdf", page=1, chunk_id="near",
              section="Libraries"),
        Chunk(text="GPO distributed 802,834 copies of 4,211 titles through FDsys.", source="r.pdf", page=2,
              chunk_id="exact", section="Libraries"),
        Chunk(text="FDsys was retired in December 2018.", source="r.pdf", page=3, chunk_id="other",
              section="Technology"),
    ]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return store


def test_hybrid_search_finds_an_exact_figure_that_embedding_search_misses(tmp_path):
    q = "How many copies were 802,834?"
    embeddings = FakeEmbeddingService({q: [1.0, 0.0, 0.0]})
    store = _exact_term_store(tmp_path)

    vector_only = Retriever(store, embeddings, top_k=1, score_threshold=0.5, hybrid_search=False).retrieve(q)
    hybrid = Retriever(store, embeddings, top_k=2, score_threshold=0.5, hybrid_search=True)
    results = hybrid.retrieve(q)

    assert [c.chunk_id for c in vector_only] == ["near"]
    assert "exact" in [c.chunk_id for c in results]
    assert hybrid.last_stats["keyword_only"] == 1
    exact = next(c for c in results if c.chunk_id == "exact")
    assert exact.score == pytest.approx(0.0)  # its real similarity, though the threshold was skipped


def test_hybrid_search_respects_the_section_filter(tmp_path):
    q = "When was FDsys retired in Technology?"
    store = _exact_term_store(tmp_path)
    retriever = Retriever(store, FakeEmbeddingService({q: [1.0, 0.0, 0.0]}), top_k=3, score_threshold=0.5,
                          hybrid_search=True)

    retriever.last_stats = {"searches": 0, "candidates": 0, "keyword_only": 0, "kept": 0, "compressed": 0}
    results = retriever._search(q, [1.0, 0.0, 0.0], 3, {"section": {"$in": ["Technology"]}}, summary=False)

    assert [c.chunk_id for c in results] == ["other"]  # "exact" also says FDsys but is in another section


def test_router_runs_only_when_the_rules_name_no_section_and_is_cached():
    calls = []

    def router(question, labels):
        calls.append(question)
        return ["Europe and Central Asia"]

    store = VectorStore(persist_dir="/tmp/unused", collection_name="unused")
    retriever = Retriever(store, FakeEmbeddingService({}), section_aware=True, hybrid_search=False, router=router)
    retriever._section_labels = lambda: ["South Asia", "Europe and Central Asia"]

    assert retriever.named_sections("What is the growth outlook for South Asia?") == [["South Asia"]]
    assert retriever.named_sections("How fast will Poland grow?") == [["Europe and Central Asia"]]
    assert retriever.named_sections("How fast will Poland grow?") == [["Europe and Central Asia"]]
    assert calls == ["How fast will Poland grow?"]  # rules first; the LLM pick is cached
