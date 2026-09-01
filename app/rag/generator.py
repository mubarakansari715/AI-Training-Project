"""LLM generation via Google Gemini.

Reads GEMINI_API_KEY from the environment — never hardcoded, never
logged. Failures (missing/invalid key, network errors, rate limits,
etc.) are caught and turned into a friendly message instead of
crashing the app.
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types

from app.config.settings import settings
from app.rag.prompt import build_prompt
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "Unable to generate an answer right now. "
    "Please check your API configuration and try again."
)
MISSING_KEY_MESSAGE = (
    "The Gemini API key is not configured. Set GEMINI_API_KEY in your "
    ".env file to enable answers."
)

# How many prior chat turns (user + assistant messages, combined) to
# send for conversational context. Keeps prompts bounded instead of
# growing unboundedly over a long session.
MAX_HISTORY_MESSAGES = 10


class AnswerGenerator:
    """Generates a document-grounded answer using Gemini."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ):
        self.model_name = model_name or settings.gemini_model
        # `is None` (not truthiness) so an explicit empty string means
        # "no key" instead of silently falling back to the configured one.
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.temperature = settings.gemini_temperature if temperature is None else temperature
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

    def is_configured(self) -> bool:
        return self._client is not None

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict] | None = None,
    ) -> str:
        """Return the answer text, or a friendly fallback message on failure.

        `history` is prior chat turns as [{"role": "user"|"assistant",
        "content": str}, ...], oldest first — used for conversational
        continuity (e.g. remembering something said earlier in the
        session). The most recent `MAX_HISTORY_MESSAGES` are included.
        """
        if not self.is_configured():
            logger.warning("Gemini API key is not configured")
            return MISSING_KEY_MESSAGE

        prompt = build_prompt(question, chunks)
        contents = _build_contents(history or [], prompt)
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(temperature=self.temperature),
            )
            text = (response.text or "").strip()
            return text or FALLBACK_MESSAGE
        except Exception:
            logger.exception("Gemini generation failed")
            return FALLBACK_MESSAGE


def _build_contents(history: list[dict], current_prompt: str) -> list[dict]:
    """Turn chat history + the current RAG prompt into Gemini's multi-turn format."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    contents = [
        {
            "role": "model" if turn["role"] == "assistant" else "user",
            "parts": [{"text": turn["content"]}],
        }
        for turn in recent
    ]
    contents.append({"role": "user", "parts": [{"text": current_prompt}]})
    return contents
