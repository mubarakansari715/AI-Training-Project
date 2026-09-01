"""Centralized app configuration, loaded from environment variables.

Reads `.env` (via python-dotenv) so every setting can be overridden
without touching code. See `.env.example` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # Verify available model names for your API key in Google AI Studio;
    # override via .env if this default is renamed/retired.
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # Low temperature ([0, 2]) favors consistent, literal answers over
    # creative ones — appropriate for a document-grounded QA assistant.
    gemini_temperature: float = _get_float("GEMINI_TEMPERATURE", 0.2)

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)

    top_k: int = _get_int("TOP_K", 5)
    # Minimum cosine similarity ([-1, 1]) for a retrieved chunk to be
    # used as context. Empirically, all-MiniLM-L6-v2 scores unrelated
    # question/chunk pairs well under 0.1 and genuinely relevant ones
    # 0.25+, so 0.2 filters noise without dropping real matches.
    similarity_threshold: float = _get_float("SIMILARITY_THRESHOLD", 0.2)
    strict_document_mode: bool = _get_bool("STRICT_DOCUMENT_MODE", True)


settings = Settings()
