"""LLM generation via OpenRouter, with Google Gemini as a fallback.

Reads OPENROUTER_API_KEY and GEMINI_API_KEY from the environment —
never hardcoded, never logged. OpenRouter is always tried first; Gemini
is only used when OpenRouter is unconfigured or a call to it fails, so
it acts purely as a fallback, not a second choice the user picks.
Failures (missing/invalid keys, network errors, rate limits, etc.) are
caught and turned into a friendly message instead of crashing the app.
"""

from __future__ import annotations

import logging

import requests
from google import genai
from google.genai import types

from app.config.settings import settings
from app.rag.prompt import build_general_knowledge_prompt, build_prompt
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TIMEOUT_SECONDS = 30

FALLBACK_MESSAGE = (
    "Unable to generate an answer right now. "
    "Please check your API configuration and try again."
)
MISSING_KEY_MESSAGE = (
    "No LLM provider is configured. Set OPENROUTER_API_KEY (preferred) or "
    "GEMINI_API_KEY (fallback) in your .env file to enable answers."
)

# How many prior chat turns (user + assistant messages, combined) to
# send for conversational context. Keeps prompts bounded instead of
# growing unboundedly over a long session.
MAX_HISTORY_MESSAGES = 10


class AnswerGenerator:
    """Generates an answer, grounded in document context when there is
    any, falling back to the model's own general knowledge otherwise.

    OpenRouter is the primary provider; Gemini is used only as a
    fallback when OpenRouter is unconfigured or its call fails.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        openrouter_api_key: str | None = None,
        openrouter_model: str | None = None,
    ):
        self.model_name = model_name or settings.gemini_model
        # `is None` (not truthiness) so an explicit empty string means
        # "no key" instead of silently falling back to the configured one.
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.temperature = settings.gemini_temperature if temperature is None else temperature
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

        self.openrouter_api_key = (
            settings.openrouter_api_key if openrouter_api_key is None else openrouter_api_key
        )
        self.openrouter_model = openrouter_model or settings.openrouter_model

    def is_configured(self) -> bool:
        return self._client is not None or bool(self.openrouter_api_key)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict] | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        openrouter_model_name: str | None = None,
    ) -> str:
        """Return the answer text, or a friendly fallback message on failure.

        `history` is prior chat turns as [{"role": "user"|"assistant",
        "content": str}, ...], oldest first — used for conversational
        continuity (e.g. remembering something said earlier in the
        session). The most recent `MAX_HISTORY_MESSAGES` are included.

        `model_name` overrides the configured Gemini model and
        `openrouter_model_name` the configured OpenRouter model, each
        for this call only — used to let the UI switch models without
        rebuilding the (cached) pipeline. `temperature` overrides the
        configured default for whichever provider ends up handling the
        call.

        OpenRouter is tried first. If it isn't configured, or the call
        raises or returns an empty response, Gemini is tried next
        (when configured). Only if neither provider produces an answer
        is `FALLBACK_MESSAGE` returned.

        When `chunks` is non-empty the answer is grounded in that
        document context (and only that context). When it's empty —
        no documents attached, or nothing relevant was retrieved — the
        model answers from its own general knowledge instead, clearly
        told not to claim the answer came from the user's documents.
        Callers should tell the two apart via `bool(chunks)` (e.g. to
        label the answer's source in the UI).
        """
        if not self.is_configured():
            logger.warning("No LLM provider is configured")
            return MISSING_KEY_MESSAGE

        prompt = build_prompt(question, chunks) if chunks else build_general_knowledge_prompt(question)
        history = history or []
        temp = self.temperature if temperature is None else temperature

        if self.openrouter_api_key:
            text = self._generate_with_openrouter(prompt, history, openrouter_model_name, temp)
            if text:
                return text

        if self._client is not None:
            text = self._generate_with_gemini(prompt, history, model_name, temp)
            if text:
                return text

        return FALLBACK_MESSAGE

    def _generate_with_gemini(
        self,
        prompt: str,
        history: list[dict],
        model_name: str | None,
        temperature: float,
    ) -> str:
        model = model_name or self.model_name
        contents = _build_gemini_contents(history, prompt)
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = (response.text or "").strip()
            if not text:
                logger.warning("Gemini returned an empty response")
            return text
        except Exception:
            logger.exception("Gemini generation failed")
            return ""

    def _generate_with_openrouter(
        self,
        prompt: str,
        history: list[dict],
        model_name: str | None,
        temperature: float,
    ) -> str:
        model = model_name or self.openrouter_model
        messages = _build_openrouter_messages(history, prompt)
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=OPENROUTER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if not text:
                logger.warning("OpenRouter returned an empty response")
            return text
        except Exception:
            logger.exception("OpenRouter generation failed")
            return ""


def _build_gemini_contents(history: list[dict], current_prompt: str) -> list[dict]:
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


def _build_openrouter_messages(history: list[dict], current_prompt: str) -> list[dict]:
    """Turn chat history + the current RAG prompt into OpenAI-style chat messages."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    messages = [{"role": turn["role"], "content": turn["content"]} for turn in recent]
    messages.append({"role": "user", "content": current_prompt})
    return messages
