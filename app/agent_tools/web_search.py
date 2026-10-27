"""web_search(query) via DuckDuckGo (the `ddgs` package): free, no API key."""

from __future__ import annotations

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from app.agent_tools.base import HTTP_TIMEOUT_SECONDS, Tool, ToolError, require_text

MAX_QUERY_LENGTH = 300
MAX_RESULTS = 5


def web_search(query) -> str:
    """Top web results for `query`: numbered title, link and snippet."""
    query = require_text("query", query, MAX_QUERY_LENGTH)
    try:
        results = DDGS(timeout=HTTP_TIMEOUT_SECONDS).text(query, max_results=MAX_RESULTS)
    except DDGSException as exc:
        # ddgs also raises this when nothing matched.
        if "no results" in str(exc).lower():
            return f"No web results found for {query!r}."
        raise ToolError(f"Web search failed: {exc}") from exc
    if not results:
        return f"No web results found for {query!r}."
    return "\n\n".join(
        f"[{i}] {r.get('title', '').strip()}\n{r.get('href', '')}\n{r.get('body', '').strip()}"
        for i, r in enumerate(results, start=1)
    )


WEB_SEARCH_TOOL = Tool(
    name="web_search",
    description=(
        "Search the web for current or general information. Returns the top results, "
        "each with a title, link and short snippet."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for, as a short natural-language query."},
        },
        "required": ["query"],
    },
    function=web_search,
)
