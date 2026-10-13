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

        fetch = max(top_k, self.rerank_candidates) if self.reranker else top_k
        query_embedding = self._embeddings.embed_query(question)
        matches = self._vector_store.search(query_embedding, top_k=fetch)

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
                )
            )
        if self.reranker:
            return self.reranker.rerank(question, results, top_k)
        return results[:top_k]
