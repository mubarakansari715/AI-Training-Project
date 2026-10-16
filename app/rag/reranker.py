"""Second-stage reranking of retrieved chunks with a cross-encoder.

Embedding search compares a question and a chunk as two separate vectors,
which blurs chunks that pack many unrelated facts together (e.g. a
"by the numbers" stats page): the right chunk is found but ranked too low
to make the top-k. A cross-encoder reads the question and each candidate
chunk *together* and scores how well the chunk answers it, so the best
candidates move to the top. It only reorders what the embedding search
already returned (after the similarity threshold), so it never adds
unrelated chunks.

Two optional cutoffs compress the context (fewer off-topic passages and
tokens) by dropping kept chunks the cross-encoder scores low:

- `min_score`: below a fixed score. Simple, but cross-encoder scores shift
  from question to question, so one value is too strict for some questions
  and too loose for others.
- `margin`: more than `margin` below the best chunk's score for *this*
  question, so the cutoff follows each question's own score range.

The best `MIN_KEEP` always stay, so a question never loses its top
passages to a strict cutoff.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config.settings import settings
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

MIN_KEEP = 2


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    logger.info("Loading reranker model: %s", model_name)
    device = None if settings.model_device == "auto" else settings.model_device
    return CrossEncoder(model_name, device=device)


class Reranker:
    """Reorders candidate chunks by cross-encoder relevance to the question."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int,
               min_score: float | None = None, margin: float | None = None) -> list[RetrievedChunk]:
        if len(chunks) <= 1:
            return chunks[:top_k]
        scores = self._model.predict([(question, chunk.text) for chunk in chunks], show_progress_bar=False)
        return select(list(zip(scores, chunks)), top_k, min_score, margin)


def select(scored: list[tuple[float, RetrievedChunk]], top_k: int, min_score: float | None,
           margin: float | None = None) -> list[RetrievedChunk]:
    """The top_k chunks by score, minus those under `min_score` or more than
    `margin` below the best score, after the first MIN_KEEP."""
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]
    floor = max(min_score if min_score is not None else float("-inf"),
                ranked[0][0] - margin if ranked and margin is not None else float("-inf"))
    return [chunk for i, (score, chunk) in enumerate(ranked) if i < MIN_KEEP or score >= floor]
