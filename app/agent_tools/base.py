"""Shared pieces for the agent tools: the Tool record, errors and input checks.

Every tool validates its own arguments before doing any work (the
"input validation" guardrail), because the arguments come from the model
and can be missing, the wrong type, or out of range.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

# Seconds to wait for any external API before giving up.
HTTP_TIMEOUT_SECONDS = 10


class ToolInputError(ValueError):
    """The model passed arguments the tool can't accept."""


class ToolError(RuntimeError):
    """The tool's work failed (network error, service down, no result)."""


@dataclass(frozen=True)
class Tool:
    """A callable the agent can use: the Python function plus its JSON schema."""

    name: str
    description: str
    parameters: dict
    function: Callable[..., str]

    @property
    def schema(self) -> dict:
        """OpenAI-compatible tool definition, as sent to Groq."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def require_number(name: str, value, max_abs: float) -> float:
    """`value` as a finite number with |value| <= max_abs (numeric strings accepted)."""
    if isinstance(value, bool):
        raise ToolInputError(f"{name} must be a number, not true/false")
    if isinstance(value, str):
        try:
            value = float(value.strip().replace(",", ""))
        except ValueError:
            raise ToolInputError(f"{name} must be a number, got {value!r}") from None
    if not isinstance(value, (int, float)):
        raise ToolInputError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ToolInputError(f"{name} must be a finite number")
    if abs(value) > max_abs:
        raise ToolInputError(f"{name} must be between -{max_abs:g} and {max_abs:g}")
    return value


def require_text(name: str, value, max_length: int, pattern: re.Pattern | None = None) -> str:
    """`value` stripped, non-empty, at most `max_length` characters and matching `pattern`."""
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be text")
    value = " ".join(value.split())
    if not value:
        raise ToolInputError(f"{name} must not be empty")
    if len(value) > max_length:
        raise ToolInputError(f"{name} must be at most {max_length} characters")
    if pattern is not None and not pattern.fullmatch(value):
        raise ToolInputError(f"{name} contains characters that aren't allowed: {value!r}")
    return value
