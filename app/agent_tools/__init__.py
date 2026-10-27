"""The Agent tab's tools: multiply, currency conversion, weather and web search.

`TOOLS` maps each tool's name to its function and JSON schema; `run_tool`
runs one call from the model and always returns text to feed back to it
(the Observe step), including when the arguments were bad or the service
failed, so the model can correct itself instead of the loop crashing.
"""

from __future__ import annotations

import json
import logging

from app.agent_tools.base import Tool, ToolError, ToolInputError
from app.agent_tools.currency import CURRENCY_TOOL
from app.agent_tools.multiply import MULTIPLY_TOOL
from app.agent_tools.weather import WEATHER_TOOL
from app.agent_tools.web_search import WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)

TOOLS: dict[str, Tool] = {tool.name: tool for tool in (MULTIPLY_TOOL, CURRENCY_TOOL, WEATHER_TOOL, WEB_SEARCH_TOOL)}


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    """OpenAI-compatible definitions for `names` (all tools if None)."""
    return [TOOLS[name].schema for name in (names if names is not None else TOOLS)]


def run_tool(name: str, arguments: dict | str | None) -> str:
    """Run tool `name` with the model's `arguments` (dict or JSON string); never raises."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool {name!r}. Available tools: {', '.join(TOOLS)}."
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return f"Error: arguments for {name} are not valid JSON."
    arguments = arguments or {}
    if not isinstance(arguments, dict):
        return f"Error: arguments for {name} must be a JSON object."

    allowed = set(tool.parameters["properties"])
    missing = [p for p in tool.parameters.get("required", []) if p not in arguments]
    unexpected = sorted(set(arguments) - allowed)
    if missing or unexpected:
        problems = ([f"missing {', '.join(missing)}"] if missing else []) + (
            [f"unexpected {', '.join(unexpected)}"] if unexpected else [])
        return f"Error: invalid arguments for {name} ({'; '.join(problems)})."

    try:
        return tool.function(**arguments)
    except ToolInputError as exc:
        return f"Error: invalid input for {name}: {exc}"
    except ToolError as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return f"Error: {name} failed: {exc}"


__all__ = ["TOOLS", "Tool", "ToolError", "ToolInputError", "run_tool", "tool_schemas"]
