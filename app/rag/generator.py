"""LLM generation via Groq, with Google Gemini as a fallback.

Reads GROQ_API_KEY and GEMINI_API_KEY from the environment — never
hardcoded, never logged. Groq is always tried first; Gemini is only used
when Groq is unconfigured or a call to it fails, so it acts purely as a
fallback, not a second choice the user picks.

OpenRouter was the primary provider before Groq. It is disabled for now
(its settings and URL are commented out below); both use the same
OpenAI-compatible chat API, so re-enabling it means pointing
`_generate_with_groq` back at OPENROUTER_URL with its key and model.
Failures (missing/invalid keys, network errors, rate limits, etc.) are
caught and turned into a friendly message instead of crashing the app.
"""

from __future__ import annotations

import logging
import re
import time

import requests
from google import genai
from google.genai import types

from app.config.settings import settings
from app.rag.model_router import AUTO_MODEL
from app.rag.prompt import build_general_knowledge_prompt, build_prompt
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_SECONDS = 30
# OpenRouter — disabled for now, see the module docstring.
# OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# OPENROUTER_TIMEOUT_SECONDS = 30

FALLBACK_MESSAGE = (
    "Unable to generate an answer right now. "
    "Please check your API configuration and try again."
)
MISSING_KEY_MESSAGE = (
    "No LLM provider is configured. Set GROQ_API_KEY (preferred) or "
    "GEMINI_API_KEY (fallback) in your .env file to enable answers."
)

# How many prior chat turns (user + assistant messages, combined) to
# send for conversational context. Keeps prompts bounded instead of
# growing unboundedly over a long session.
MAX_HISTORY_MESSAGES = 10


class AnswerGenerator:
    """Generates an answer, grounded in document context when there is
    any, falling back to the model's own general knowledge otherwise.

    Groq is the primary provider; Gemini is used only as a fallback
    when Groq is unconfigured or its call fails.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        groq_api_key: str | None = None,
        groq_model: str | None = None,
    ):
        self.model_name = model_name or settings.gemini_model
        # `is None` (not truthiness) so an explicit empty string means
        # "no key" instead of silently falling back to the configured one.
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.temperature = settings.gemini_temperature if temperature is None else temperature
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

        self.groq_api_key = settings.groq_api_key if groq_api_key is None else groq_api_key
        self.groq_model = groq_model or settings.groq_model

    def is_configured(self) -> bool:
        return self._client is not None or bool(self.groq_api_key)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict] | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        groq_model_name: str | None = None,
    ) -> str:
        """Return the answer text, or a friendly fallback message on failure.

        `history` is prior chat turns as [{"role": "user"|"assistant",
        "content": str}, ...], oldest first — used for conversational
        continuity (e.g. remembering something said earlier in the
        session). The most recent `MAX_HISTORY_MESSAGES` are included.

        `model_name` overrides the configured Gemini model and
        `groq_model_name` the configured Groq model, each
        for this call only — used to let the UI switch models without
        rebuilding the (cached) pipeline. `temperature` overrides the
        configured default for whichever provider ends up handling the
        call.

        Groq is tried first, unless a Gemini model was picked
        explicitly (`model_name` set, `groq_model_name` not), in
        which case Gemini goes first. If the first provider isn't
        configured, or its call raises or returns an empty response,
        the other is tried next (when configured). Only if neither
        produces an answer is `FALLBACK_MESSAGE` returned.

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
        text = self.complete(
            prompt,
            history=history,
            model_name=model_name,
            temperature=temperature,
            groq_model_name=groq_model_name,
        )
        return text or FALLBACK_MESSAGE

    def complete(
        self,
        prompt: str,
        history: list[dict] | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        groq_model_name: str | None = None,
    ) -> str:
        """Send a ready-made prompt through the provider fallback chain
        (see `generate`) and return the reply, or "" if every provider
        failed or none is configured. Used directly for helper calls such
        as query rewriting that don't need the RAG prompt."""
        history = history or []
        temp = self.temperature if temperature is None else temperature

        def try_groq() -> str:
            if not self.groq_api_key:
                return ""
            return self._generate_with_groq(prompt, history, groq_model_name, temp)

        def try_gemini() -> str:
            if self._client is None:
                return ""
            return self._generate_with_gemini(prompt, history, model_name, temp)

        # A Gemini model picked explicitly (with no Groq override)
        # means the user wants Gemini, so it goes first; otherwise
        # Groq stays primary.
        prefer_gemini = model_name is not None and groq_model_name is None
        providers = (try_gemini, try_groq) if prefer_gemini else (try_groq, try_gemini)

        for provider in providers:
            text = provider()
            if text:
                return text
        return ""

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

    def _generate_with_groq(
        self,
        prompt: str,
        history: list[dict],
        model_name: str | None,
        temperature: float,
    ) -> str:
        model = self._concrete_groq_model(model_name)
        messages = _build_chat_messages(history, prompt)
        try:
            response = self._post_groq({"model": model, "messages": messages, "temperature": temperature})
            response.raise_for_status()
            data = response.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if not text:
                logger.warning("Groq returned an empty response")
            return text
        except Exception:
            logger.exception("Groq generation failed")
            return ""

    def _concrete_groq_model(self, model_name: str | None) -> str:
        """The model to send to Groq. "auto" should already have been
        resolved per question (see app.rag.model_router); if it reaches
        here — e.g. a helper call like query rewriting — use the fast model."""
        model = model_name or self.groq_model
        return settings.groq_fast_model if model == AUTO_MODEL else model

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model_name: str | None = None,
        temperature: float | None = None,
        tool_choice: str = "auto",
    ) -> dict:
        """One Groq chat turn with tool calling (OpenAI-compatible).

        Returns the assistant message dict (with `content` and/or
        `tool_calls`). With no `tools`, it's a plain turn (the model can't
        call anything — sending tool_choice "none" instead is rejected when
        the model tries anyway). A rate-limit response that asks for a short
        wait is retried once. Raises ToolCallingError on any other failure,
        so the caller can fall back to the non-agent pipeline (which also has
        the Gemini fallback). Groq only: tool calling isn't wired up for Gemini.
        """
        if not self.groq_api_key:
            raise ToolCallingError("Groq is not configured")
        payload = {
            "model": self._concrete_groq_model(model_name),
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        try:
            response = self._post_groq(payload)
            if response.status_code >= 400:
                logger.warning("Groq tool-calling turn rejected: %s %s", response.status_code, response.text[:300])
            response.raise_for_status()
            return response.json()["choices"][0]["message"]
        except Exception as exc:
            logger.exception("Groq tool-calling turn failed")
            raise ToolCallingError(str(exc)) from exc

    def _post_groq(self, payload: dict):
        """POST to Groq; on a rate-limit response asking for a short wait
        (<= MAX_RETRY_WAIT_SECONDS), wait and retry once. Groq's per-minute
        token limit resets within seconds, so this usually succeeds — and
        avoids leaning on the fallback provider, whose quota may be spent."""
        response = None
        for attempt in (1, 2):
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                json=payload,
                timeout=GROQ_TIMEOUT_SECONDS,
            )
            wait = _retry_after_seconds(response)
            if response.status_code == 429 and attempt == 1 and wait is not None and wait <= MAX_RETRY_WAIT_SECONDS:
                logger.info("Groq rate limit; retrying in %.1fs", wait)
                time.sleep(wait)
                continue
            break
        return response


class ToolCallingError(Exception):
    """A tool-calling turn failed (provider error, rate limit, bad response)."""


# Longest wait a rate-limited tool turn will sleep before retrying once.
MAX_RETRY_WAIT_SECONDS = 15.0
_TRY_AGAIN_RE = re.compile(r"try again in ([\d.]+)s")


def _retry_after_seconds(response) -> float | None:
    """How long Groq asks us to wait on a 429 (Retry-After header or message)."""
    if response.status_code != 429:
        return None
    header = response.headers.get("retry-after")
    try:
        if header is not None:
            return float(header)
    except ValueError:
        pass
    match = _TRY_AGAIN_RE.search(response.text or "")
    return float(match.group(1)) + 0.5 if match else None


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


def _build_chat_messages(history: list[dict], current_prompt: str) -> list[dict]:
    """Turn chat history + the current RAG prompt into OpenAI-style chat messages."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    messages = [{"role": turn["role"], "content": turn["content"]} for turn in recent]
    messages.append({"role": "user", "content": current_prompt})
    return messages
