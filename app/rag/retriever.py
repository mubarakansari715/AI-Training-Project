"""Semantic retrieval.

Embeds a user's question and searches the vector store for the most
relevant document chunks, so only that (small) context is ever sent
to the LLM instead of the entire document collection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config.settings import settings
from app.rag.embeddings import EmbeddingService
from app.rag.structure import is_front_matter, sections_mentioned
from app.rag.vector_store import VectorStore

# A request to summarize a whole document/video ("summarize this video",
# "what is the report about?"). Those need broad coverage, not just the
# single closest chunk — and generic wording like "summarize" scores low
# against specific content, so the similarity threshold would drop most
# of the document. Both an intent word and a document word are required,
# so "summarize the French Revolution" is still a normal question.
_SUMMARY_INTENT_RE = re.compile(
    r"\b(summar\w*|overview|tl;?dr|gist|key\s+points|main\s+points|about)\b", re.IGNORECASE
)
_DOCUMENT_WORD_RE = re.compile(
    r"\b(video|audio|recording|transcript|document|doc|file|report|pdf|upload\w*)s?\b", re.IGNORECASE
)
_SUMMARY_TOP_K_MULTIPLIER = 3
_SUMMARY_TOP_K_MAX = 15
# Small-to-big: for a question about one named section, the best-matching
# chunks' whole pages go to the LLM. A section's headline facts (a region's
# forecast, a chapter's key numbers) sit on its opening page but often in a
# chunk that scored lower than its neighbours. Capped to bound the prompt.
_EXPAND_MAX_PAGES = 2


def is_summary_request(question: str) -> bool:
    return bool(_SUMMARY_INTENT_RE.search(question) and _DOCUMENT_WORD_RE.search(question))


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its similarity score and source metadata."""

    text: str
    source: str
    page: int | None
    score: float
    chunk_id: str
    section: str | None = None


class Retriever:
    """Retrieves the most relevant chunks for a question."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int | None = None,
        score_threshold: float | None = None,
        reranker=None,
        rerank_candidates: int | None = None,
        section_aware: bool = True,
    ):
        self._vector_store = vector_store
        self._embeddings = embedding_service
        self.top_k = top_k or settings.top_k
        self.score_threshold = (
            settings.similarity_threshold if score_threshold is None else score_threshold
        )
        # Optional second stage (app.rag.reranker.Reranker): fetch a wider
        # candidate set, then let the reranker pick the best top_k.
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates or settings.rerank_candidates
        self.section_aware = section_aware

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return relevant chunks for `question`, most similar first.

        Summary requests (see `is_summary_request`) fetch more chunks and
        skip the similarity threshold, so the whole document is covered.
        With a reranker, `rerank_candidates` chunks are fetched, filtered
        by the threshold, and reranked down to top_k.
        """
        if not question or not question.strip():
            return []

        summary = is_summary_request(question)
        top_k = min(self.top_k * _SUMMARY_TOP_K_MULTIPLIER, _SUMMARY_TOP_K_MAX) if summary else self.top_k

        query_embedding = self._embeddings.embed_query(question)
        named, where = self._section_filter(question)
        if len(named) > 1:
            # A question naming several sections ("compare South Asia and
            # Sub-Saharan Africa") needs evidence from each: give every named
            # section its own share instead of letting one fill the top-k. A
            # section's headline sentence measured 3rd-4th within its own
            # section, so two sections each get a full top_k; more sections
            # split ~2x top_k between them to keep the prompt bounded.
            share = top_k if len(named) <= 2 else max(2, -(-2 * top_k // len(named)))
            results, seen = [], set()
            for group in named:
                for chunk in self._search(question, query_embedding, share, {"section": {"$in": group}}, summary):
                    if chunk.chunk_id not in seen:
                        seen.add(chunk.chunk_id)
                        results.append(chunk)
            return results
        results = self._search(question, query_embedding, top_k, where, summary)
        if named and not results:  # named section exists but nothing relevant there: search everything
            results = self._search(question, query_embedding, top_k, None, summary)
        elif len(named) == 1:
            results = self._expand_to_pages(results)
        return results

    def _expand_to_pages(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Replace the hits on the top `_EXPAND_MAX_PAGES` pages with those
        whole pages (reading order, same section); other hits stay as they are."""
        pages = []
        for chunk in results:
            key = (chunk.source, chunk.page)
            if chunk.page is not None and key not in pages and len(pages) < _EXPAND_MAX_PAGES:
                pages.append(key)
        expanded, seen = [], set()
        for chunk in results:
            key = (chunk.source, chunk.page)
            if key not in pages:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    expanded.append(chunk)
                continue
            for match in self._vector_store.page_chunks(*key):
                if match["chunk_id"] in seen or match["section"] != chunk.section:
                    continue
                seen.add(match["chunk_id"])
                expanded.append(RetrievedChunk(
                    text=match["text"], source=match["source"], page=match["page"], score=chunk.score,
                    chunk_id=match["chunk_id"], section=match["section"],
                ))
        return expanded

    def _search(
        self, question: str, query_embedding: list[float], top_k: int, where: dict | None, summary: bool
    ) -> list[RetrievedChunk]:
        """One filtered search: fetch candidates, apply the threshold, rerank to top_k."""
        fetch = max(top_k, self.rerank_candidates) if self.reranker else top_k
        matches = self._vector_store.search(query_embedding, top_k=fetch, where=where)

        results = []
        for match in matches:
            # Cosine distance in [0, 2] -> similarity score in [-1, 1].
            score = 1 - match["distance"]
            if score < self.score_threshold and not summary:
                continue
            results.append(
                RetrievedChunk(
                    text=match["text"],
                    source=match["source"],
                    page=match["page"],
                    score=score,
                    chunk_id=match["chunk_id"],
                    section=match.get("section"),
                )
            )
        if self.reranker:
            return self.reranker.rerank(question, results, top_k)
        return results[:top_k]

    def _section_filter(self, question: str) -> tuple[list[list[str]], dict | None]:
        """(named groups, where): search only the sections the question names
        ("according to the executive summary…", "Chapter 3", "Box 1.1",
        "South Asia"); otherwise search everything except front matter
        (table of contents, abbreviations), whose titles match every topic."""
        labels = self._section_labels() if self.section_aware else []
        if not labels:
            return [], None
        named = sections_mentioned(question, labels)
        if named:
            return named, {"section": {"$in": sorted({label for group in named for label in group})}}
        front = [label for label in labels if is_front_matter(label)]
        return [], ({"section": {"$nin": front}} if front else None)

    def _section_labels(self) -> list[str]:
        """Distinct section labels in the index, cached until it changes."""
        count = self._vector_store.count()
        if getattr(self, "_labels_count", None) != count:
            self._labels = self._vector_store.sections()
            self._labels_count = count
        return self._labels
