"""Centralized app configuration, loaded from environment variables.

Reads `.env` (via python-dotenv) so every setting can be overridden
without touching code. See `.env.example` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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


def _get_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value and value.strip().lower() == "off":
        return None
    return float(value) if value else default


# Paid-tier list prices in USD per 1M tokens (input, output), to estimate what
# a run would cost; the app itself runs on free tiers. Checked 2026-10-15 on
# console.groq.com/docs/models and ai.google.dev/gemini-api/docs/pricing.
_DEFAULT_MODEL_PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


def _get_prices(name: str, default: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """`default`, updated from "model=input/output,model=input/output" (USD per 1M tokens)."""
    prices = dict(default)
    for entry in filter(None, (part.strip() for part in os.getenv(name, "").split(","))):
        model, _, price = entry.rpartition("=")
        input_price, _, output_price = price.partition("/")
        prices[model.strip()] = (float(input_price), float(output_price))
    return prices


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

    # OpenRouter — the previous primary answer provider, disabled for now
    # (see app/rag/generator.py). Its key is used for reading page images
    # (VISION_MODEL, app/rag/visual.py). Uncomment the model to bring it back.
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    # openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    # Vision model (on OpenRouter) that transcribes charts, tables and
    # call-outs from rendered PDF pages; comma-separated models are tried in
    # order. The defaults are free (rate-limited); paid ones such as
    # google/gemini-3.8-flash need OpenRouter credits.
    vision_model: str = os.getenv(
        "VISION_MODEL",
        "dots-studio/dots-3-note-preview:free,nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,google/gemma-4-31b-it:free",
    )
    # Read page images (charts, tables, call-outs) and attach footnotes for
    # PDFs up to this many pages — one vision call per page, cached in
    # VISION_CACHE_DIR. 0 turns it off.
    visual_reading_max_pages: int = _get_int("VISUAL_READING_MAX_PAGES", 30)
    # For longer PDFs, only pages holding a chart that can't be measured from
    # its drawing (a picture) are read, up to this many. 0 turns it off.
    visual_chart_pages: int = _get_int("VISUAL_CHART_PAGES", 40)
    # Tables rebuilt from the page layout (header bars + word positions, no
    # vision model; app/rag/layout_tables.py): "auto" = whenever the vision
    # model doesn't read the PDF (too long, or no OPENROUTER_API_KEY), "on"
    # = always, "off" = never.
    layout_tables: str = os.getenv("LAYOUT_TABLES", "auto")
    # Bar charts drawn as vector shapes are measured from the PDF's drawing
    # (app/rag/chart_measure.py; free, local) and indexed as passages, so a
    # question about a chart's values finds its page.
    index_measured_charts: bool = _get_bool("INDEX_MEASURED_CHARTS", True)
    # Answer chart questions from the chart pages (app/rag/chart_reader.py):
    # their measured values, and the page images shown to a vision model;
    # values judged from the image are marked as estimates. The images need
    # GEMINI_API_KEY. CHART_MODEL defaults to GEMINI_MODEL.
    chart_reading: bool = _get_bool("CHART_READING", True)
    chart_model: str = os.getenv("CHART_MODEL", "") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # Copies of indexed PDFs, so chart pages can be rendered at answer time.
    page_store_dir: str = os.getenv("PAGE_STORE_DIR", "data/documents")
    vision_cache_dir: str = os.getenv("VISION_CACHE_DIR", "data/vision_cache")

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
    # "recursive": paragraph/line/sentence/word boundaries, in that order
    # (may cut a long sentence); "sentence": whole sentences only, overlap =
    # the previous chunk's last sentence(s) — see app/rag/chunker.py.
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    # What CHUNK_SIZE and CHUNK_OVERLAP count: "chars" or "tokens" (the
    # embedding model's tokens, as the field guide sizes its chunks). Keep a
    # token size under the model's limit (all-MiniLM-L6-v2 embeds only the
    # first 256 tokens of a chunk; 800 characters is about 180).
    chunk_unit: str = os.getenv("CHUNK_UNIT", "chars")

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
    # Relevance cutoff after reranking: drop passages the cross-encoder scores
    # below this (its raw score; ms-marco: > 0 relevant, < 0 not), always
    # keeping the best few. Empty = off: always send top_k.
    rerank_min_score: float | None = _get_float("RERANK_MIN_SCORE", None)
    # Relative cutoff (context compression): also drop passages scored more
    # than this below the best passage for the same question. 5 sent ~15%
    # fewer passages on GPO with no lost answer page or failed answer
    # (evaluation/reports/*_compare.json). "off" = no relative cutoff.
    rerank_relative_margin: float | None = _get_float("RERANK_RELATIVE_MARGIN", 5.0)
    # Hybrid search: also rank chunks by exact words (BM25) and merge both
    # rankings, so exact names and figures ("FDsys", "802,834") are found
    # even when embedding search ranks them low — see app/rag/keyword_search.py.
    hybrid_search: bool = _get_bool("HYBRID_SEARCH", True)
    # How many BM25 candidates join the embedding candidates per search.
    keyword_candidates: int = _get_int("KEYWORD_CANDIDATES", 20)
    # Detect chapters/sections in structured PDFs (bookmarks, running
    # headers, boxes), label chunks with them, and let questions that name a
    # section search only there — see app/rag/structure.py.
    section_aware: bool = _get_bool("SECTION_AWARE", True)
    # Where section headings come from: "auto" (PDF bookmarks, or headings
    # found by font size when a PDF has none), "bookmarks" or "fonts".
    structure_source: str = os.getenv("STRUCTURE_SOURCE", "auto")
    # How a question is routed to sections: "rules" (the sections it names)
    # or "llm" (rules first; a question naming none gets one short LLM call
    # that picks sections, e.g. "India" -> South Asia; app/rag/query_router.py).
    query_routing: str = os.getenv("QUERY_ROUTING", "rules")
    # True (default): only ever answer from documents, refusing when
    # nothing relevant is found. False: answer from documents when they
    # have the answer, otherwise fall back to the model's general knowledge
    # (source is shown in the UI either way).
    strict_document_mode: bool = _get_bool("STRICT_DOCUMENT_MODE", True)
    # Rewrite each question with the LLM before searching (fixes typos,
    # resolves follow-ups like "and my cholesterol?"); costs one short
    # extra LLM call per question while documents are indexed.
    query_rewriting: bool = _get_bool("QUERY_REWRITING", True)
    # Split comparison and multi-part questions into sub-questions and
    # search each one (one short extra LLM call, only for such questions);
    # see app/rag/query_decomposer.py.
    query_decomposition: bool = _get_bool("QUERY_DECOMPOSITION", True)
    # Auto (agent) mode: the LLM gets a document-search tool and decides
    # whether, how often and with what wording to search before answering,
    # like ChatGPT/Claude — see app/rag/agent.py. Needs Groq (tool calling).
    agent_mode: bool = _get_bool("AGENT_MODE", True)
    agent_max_searches: int = _get_int("AGENT_MAX_SEARCHES", 3)
    # USD per 1M (input, output) tokens per model, for cost estimates in the
    # evaluation; add or override with MODEL_PRICES (see _get_prices).
    model_prices: dict = field(default_factory=lambda: _get_prices("MODEL_PRICES", _DEFAULT_MODEL_PRICES))


settings = Settings()
