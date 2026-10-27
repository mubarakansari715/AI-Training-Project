"""multiply(a, b): pure Python, no network — the simplest possible tool."""

from __future__ import annotations

from app.agent_tools.base import Tool, require_number

# Keeps results readable and well inside float precision.
MAX_FACTOR = 1e12


def multiply(a, b) -> str:
    """Return a × b (15 significant digits, so 0.1 × 0.2 shows as 0.02)."""
    a = require_number("a", a, MAX_FACTOR)
    b = require_number("b", b, MAX_FACTOR)
    return f"{a:.15g} × {b:.15g} = {a * b:,.15g}"


MULTIPLY_TOOL = Tool(
    name="multiply",
    description="Multiply two numbers exactly. Use it for any multiplication instead of doing it in your head.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number."},
            "b": {"type": "number", "description": "The second number."},
        },
        "required": ["a", "b"],
    },
    function=multiply,
)
