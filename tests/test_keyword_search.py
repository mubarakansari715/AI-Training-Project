from app.rag.keyword_search import KeywordIndex, fuse, tokenize


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def count(self):
        return len(self.chunks)

    def all_chunks(self):
        return list(self.chunks)


def _chunk(chunk_id, text, section=None):
    return {"chunk_id": chunk_id, "text": text, "source": "r.pdf", "page": 1, "section": section}


def test_tokenize_keeps_figures_and_drops_stopwords_and_sentence_periods():
    assert tokenize("What is in Box 2.1? GPO sent 802,834 copies in FY 2018.") == [
        "box", "2.1", "gpo", "sent", "802,834", "copies", "fy", "2018"]


def test_search_ranks_the_chunk_with_the_rare_exact_term_first():
    store = FakeStore([_chunk("a", "The report covers libraries."), _chunk("b", "govinfo replaced FDsys."),
                       _chunk("c", "The report covers printing.")])
    hits = KeywordIndex(store).search("What replaced FDsys?", top_k=3)

    assert [hit["chunk_id"] for hit in hits] == ["b"]  # chunks sharing no word are never returned
    assert hits[0]["bm25"] > 0


def test_search_applies_section_filters():
    store = FakeStore([_chunk("a", "South Asia growth 6.3", "South Asia"), _chunk("b", "growth 2.4", "Europe"),
                       _chunk("c", "growth tables", "Contents"), _chunk("d", "inflation", "Europe")])
    index = KeywordIndex(store)

    assert [h["chunk_id"] for h in index.search("growth", 5, {"section": {"$in": ["Europe"]}})] == ["b"]
    assert "c" not in [h["chunk_id"] for h in index.search("growth", 5, {"section": {"$nin": ["Contents"]}})]


def test_index_is_rebuilt_when_the_store_changes():
    store = FakeStore([_chunk("a", "alpha"), _chunk("b", "beta")])
    index = KeywordIndex(store)
    assert index.search("gamma", 5) == []

    store.chunks += [_chunk("c", "gamma"), _chunk("d", "delta")]
    assert [h["chunk_id"] for h in index.search("gamma", 5)] == ["c"]


def test_empty_store_or_stopword_only_query_returns_nothing():
    assert KeywordIndex(FakeStore([])).search("anything", 5) == []
    assert KeywordIndex(FakeStore([_chunk("a", "alpha")])).search("what is the", 5) == []


def test_fuse_prefers_chunks_ranked_well_in_both_lists():
    assert fuse(["a", "b"], ["b", "c"]) == ["b", "a", "c"]
