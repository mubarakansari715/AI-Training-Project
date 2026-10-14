"""Centralized app configuration, loaded from environment variables.

Reads `.env` (via python-dotenv) so every setting can be overridden
without touching code. See `.env.example` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The assistant's display name, used in the UI and in the LLM's
# system instructions so it introduces itself consistently.
APP_NAME = "ALLY AI"
# Who built the assistant — its answer to "who created/owns you?".
APP_CREATOR = "Mubarak Ansari"
# Compass logo: browser tab icon, sidebar logo, and assistant avatar.
APP_ICON = str(Path(__file__).resolve().parent.parent / "assets" / "ally_icon.png")


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
    # Primary LLM provider. Get a key at https://console.groq.com/keys.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    # Any chat model your Groq key can use (GET /openai/v1/models), or
    # "auto": simple questions go to the fast model, complex ones
    # (comparisons, explanations, multi-part) to the strong one — see
    # app/rag/model_router.py.
    groq_model: str = os.getenv("GROQ_MODEL", "auto")
    groq_fast_model: str = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    groq_strong_model: str = os.getenv("GROQ_STRONG_MODEL", "openai/gpt-oss-120b")

    # OpenRouter — the previous primary provider, disabled for now (see
    # app/rag/generator.py). Uncomment to bring its settings back.
    # openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    # openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

    # Fallback provider, used only when Groq is unconfigured or a call
    # fails.
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # Verify available model names for your API key in Google AI Studio;
    # override via .env if this default is renamed/retired.
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # Low temperature ([0, 2]) favors consistent, literal answers over
    # creative ones — appropriate for a document-grounded QA assistant.
    gemini_temperature: float = _get_float("GEMINI_TEMPERATURE", 0.2)

    # faster-whisper model for transcribing attached video/audio
    # (tiny/base/small/medium); larger is more accurate but slower.
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Device for the local models (embeddings, reranker). "cpu" by default:
    # on Intel Macs the "mps" GPU backend measured 4-10x slower than CPU for
    # these small models. "auto" lets sentence-transformers choose.
    model_device: str = os.getenv("MODEL_DEVICE", "cpu")
    # Instruction some models expect before each query (e.g. BGE:
    # "Represent this sentence for searching relevant passages: ").
    embedding_query_prefix: str = os.getenv("EMBEDDING_QUERY_PREFIX", "")

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
    # Cross-encoder that reorders the retrieved candidates (empty = off).
    # It fixes chunks that are found but ranked too low, e.g. dense
    # "by the numbers" pages; see app/rag/reranker.py.
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # How many embedding-search candidates the reranker chooses top_k from.
    rerank_candidates: int = _get_int("RERANK_CANDIDATES", 20)
    # Detect chapters/sections in structured PDFs (bookmarks, running
    # headers, boxes), label chunks with them, and let questions that name a
    # section search only there — see app/rag/structure.py.
    section_aware: bool = _get_bool("SECTION_AWARE", True)
    # False (default): answer from documents when they have the answer,
    # otherwise fall back to the model's general knowledge (source is
    # shown in the UI either way). True: only ever answer from documents,
    # refusing when nothing relevant is found.
    strict_document_mode: bool = _get_bool("STRICT_DOCUMENT_MODE", False)
    # Rewrite each question with the LLM before searching (fixes typos,
    # resolves follow-ups like "and my cholesterol?"); costs one short
    # extra LLM call per question while documents are indexed.
    query_rewriting: bool = _get_bool("QUERY_REWRITING", True)
    # Auto (agent) mode: the LLM gets a document-search tool and decides
    # whether, how often and with what wording to search before answering,
    # like ChatGPT/Claude — see app/rag/agent.py. Needs Groq (tool calling).
    agent_mode: bool = _get_bool("AGENT_MODE", True)
    agent_max_searches: int = _get_int("AGENT_MAX_SEARCHES", 3)


settings = Settings()
