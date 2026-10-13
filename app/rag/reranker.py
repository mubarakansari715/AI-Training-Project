"""Second-stage reranking of retrieved chunks with a cross-encoder.

Embedding search compares a question and a chunk as two separate vectors,
which blurs chunks that pack many unrelated facts together (e.g. a
"by the numbers" stats page): the right chunk is found but ranked too low
to make the top-k. A cross-encoder reads the question and each candidate
chunk *together* and scores how well the chunk answers it, so the best
candidates move to the top. It only reorders what the embedding search
already returned (after the similarity threshold), so it never adds
unrelated chunks.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    logger.info("Loading reranker model: %s", model_name)
    return CrossEncoder(model_name)


class Reranker:
    """Reorders candidate chunks by cross-encoder relevance to the question."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if len(chunks) <= 1:
            return chunks[:top_k]
        scores = self._model.predict([(question, chunk.text) for chunk in chunks], show_progress_bar=False)
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
