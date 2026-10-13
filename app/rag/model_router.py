"""Automatic model choice for the "auto" Groq model.

Simple lookups ("what is my vitamin D level?") don't need the largest
model; comparisons, explanations, summaries and multi-part questions do.
A quick rule check (no extra LLM call, so no added latency) sends each
question to the fast or the strong model.
"""

from __future__ import annotations

import re

from app.config.settings import settings

AUTO_MODEL = "auto"

_COMPLEX_RE = re.compile(
    r"\b(compar\w*|versus|vs\.?|differ\w*|why|explain\w*|analy[sz]\w*|summar\w*|overview"
    r"|trend\w*|impact|relationship|pros and cons|advantages?|disadvantages?|evaluate|assess)\b",
    re.IGNORECASE,
)
# Long questions are usually multi-part or carry detailed instructions.
_LONG_QUESTION_WORDS = 25


def is_complex(question: str) -> bool:
    return bool(
        _COMPLEX_RE.search(question)
        or len(question.split()) >= _LONG_QUESTION_WORDS
        or question.count("?") > 1
    )


def choose_groq_model(question: str) -> str:
    """The Groq model to use for `question` under "auto"."""
    return settings.groq_strong_model if is_complex(question) else settings.groq_fast_model


def resolve_groq_model(model: str | None, question: str) -> str | None:
    """Turn "auto" into a concrete model for `question`; pass anything else through."""
    return choose_groq_model(question) if model == AUTO_MODEL else model
