"""Embedding service.

Wraps a sentence-transformers model behind a small service so the
model is loaded once and reused for every chunk/question, instead of
being recreated on every call.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load (and cache) a sentence-transformers model by name."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


class EmbeddingService:
    """Generates vector embeddings using a cached local model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = _load_model(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (e.g. chunks) into vectors."""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (e.g. a user question)."""
        return self.embed_texts([text])[0]
