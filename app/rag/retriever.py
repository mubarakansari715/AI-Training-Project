"""Semantic retrieval.

Embeds a user's question and searches the vector store for the most
relevant document chunks, so only that (small) context is ever sent
to the LLM instead of the entire document collection.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore


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
    ):
        self._vector_store = vector_store
        self._embeddings = embedding_service
        self.top_k = top_k or settings.top_k
        self.score_threshold = (
            settings.similarity_threshold if score_threshold is None else score_threshold
        )

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return relevant chunks for `question`, most similar first."""
        if not question or not question.strip():
            return []

        query_embedding = self._embeddings.embed_query(question)
        matches = self._vector_store.search(query_embedding, top_k=self.top_k)

        results = []
        for match in matches:
            # Cosine distance in [0, 2] -> similarity score in [-1, 1].
            score = 1 - match["distance"]
            if score < self.score_threshold:
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
        return results
