"""Keyword (BM25) search over the indexed chunks.

Embedding search matches meaning, so it finds the right topic but is weak
on the exact words a question depends on: names and codes ("FDsys",
"Box 2.1") and figures ("802,834"). BM25 scores chunks by those exact
words, weighted by how rare each word is. The retriever runs both searches
and merges the two rankings (see `fuse`), so a chunk found by either one
reaches the reranker.

The index is built in memory from the vector store's chunks and rebuilt
whenever the chunk count changes, so it never needs storing separately.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.rag.vector_store import VectorStore

# Words and numbers, lowercased; a number keeps its thousands separators and
# decimals ("802,834", "2.1"), while a sentence-ending period is dropped.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
# Words too common to tell chunks apart; without this list a question's
# "what", "the" and "of" match nearly every chunk.
_STOPWORDS = frozenset(
    "a an and are as at be been but by did do does for from had has have how in is it its of on or than that "
    "the their there these this those to was were what when where which who whom why will with".split()
)
# Reciprocal rank fusion constant: the standard value, which keeps one list's
# top rank from outweighing a chunk ranked well in both.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _matches(chunk: dict, where: dict | None) -> bool:
    """The subset of ChromaDB `where` filters the retriever uses: section $in / $nin."""
    if not where:
        return True
    condition = where.get("section", {})
    if "$in" in condition:
        return chunk["section"] in condition["$in"]
    if "$nin" in condition:
        return chunk["section"] not in condition["$nin"]
    return True


class KeywordIndex:
    """BM25 over every chunk in a vector store, rebuilt when the store changes."""

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._count = -1
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def search(self, query: str, top_k: int, where: dict | None = None) -> list[dict]:
        """The `top_k` chunks sharing the most (and rarest) words with `query`,
        best first; chunks sharing none are never returned."""
        terms = tokenize(query)
        if not terms or not self._refresh():
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(
            ((score, chunk) for score, chunk in zip(scores, self._chunks) if score > 0 and _matches(chunk, where)),
            key=lambda pair: pair[0], reverse=True,
        )
        return [dict(chunk, bm25=float(score)) for score, chunk in ranked[:top_k]]

    def _refresh(self) -> bool:
        count = self._vector_store.count()
        if count != self._count:
            self._chunks = self._vector_store.all_chunks()
            tokenized = [tokenize(chunk["text"]) for chunk in self._chunks]
            self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None
            self._count = count
        return self._bm25 is not None


def fuse(*rankings: list[str]) -> list[str]:
    """Reciprocal rank fusion of several best-first lists of chunk IDs:
    a chunk ranked high in any list, or well in several, comes first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
