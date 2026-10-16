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


def _device() -> str | None:
    """settings.model_device, with "auto" meaning let the library decide."""
    return None if settings.model_device == "auto" else settings.model_device


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load (and cache) a sentence-transformers model by name."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name, device=_device())


class EmbeddingService:
    """Generates vector embeddings using a cached local model."""

    def __init__(self, model_name: str | None = None, query_prefix: str | None = None):
        self.model_name = model_name or settings.embedding_model
        # Some retrieval models (e.g. BGE) expect an instruction prepended
        # to queries — but not to the documents being searched.
        self.query_prefix = settings.embedding_query_prefix if query_prefix is None else query_prefix
        self._model = _load_model(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (e.g. chunks) into vectors."""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def count_tokens(self, text: str) -> int:
        """Tokens `text` takes in this model (for token-sized chunks)."""
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (e.g. a user question)."""
        return self.embed_texts([self.query_prefix + text])[0]
