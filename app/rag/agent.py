"""Auto (agent) mode: the LLM decides whether and how to search.

Instead of always running one search before answering, the model gets a
`search_documents` tool — the way ChatGPT and Claude use tools — and
decides per question:

- no search for greetings or questions it can answer without the documents;
- one search for a single fact;
- several searches, with its own wording, for comparisons and multi-part
  questions ("South Asia vs Sub-Saharan Africa growth"), or to retry with
  different words when the first results don't contain the answer.

Guards keep it predictable: at most `max_searches` searches; in strict mode
the first turn must search (unless it's small talk); facts about the
documents may only come from search results; and any provider error raises
AgentError so the caller can fall back to the standard pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config.settings import APP_CREATOR, APP_NAME
from app.rag.generator import AnswerGenerator, ToolCallingError
from app.rag.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

# Prior chat turns given to the agent for context (same bound as the
# standard pipeline's generator).
_HISTORY_MESSAGES = 10

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the user's uploaded documents. Returns numbered excerpts, each with its "
            "source file and page (or [mm:ss] timestamps for video/audio transcripts)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language question about ONE thing to look up.",
                }
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """You are {app_name}, an assistant that answers questions about the user's \
uploaded documents. You have a tool, search_documents, that searches those documents.

How to work:
- Search whenever the question could be answered from the documents.
- For comparisons or multi-part questions, search once per part, each time with a \
natural-language question about one thing (not a keyword list).
- If the results don't contain the answer, you may search again with different wording. \
You can search at most {max_searches} times in total.
- Don't search for greetings, thanks, or questions about yourself.

Rules for the answer:
- State facts about the documents only if they appear in the search results. Never fill \
gaps from memory — not even expanding an abbreviation the documents don't expand.
- Cite each fact in plain text like "(report.pdf, page 5)" — never bracketed markers.
- {mode_rule}
- Only if the user asks who you are, say you are {app_name}; if asked who built or owns \
{app_name}, say it was built by {app_creator}."""

STRICT_RULE = ("Strict mode: answer only from the search results. If they don't contain the "
               "answer, say the information was not found in the uploaded documents.")
OPEN_RULE = ("If the documents don't contain the answer, you may answer from general "
             "knowledge, but say clearly that this part is not from the documents.")


FINAL_ANSWER_PROMPT = """{question}

You already searched the documents ({searches}); these are the excerpts found:

{excerpts}

Answer the question now, using only these excerpts (follow the rules above). \
If they don't contain the answer, say so."""


class AgentError(Exception):
    """The agent couldn't produce an answer (provider failure, empty reply)."""


@dataclass
class AgentResult:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)


def _format_results(chunks: list[RetrievedChunk], start: int) -> str:
    if not chunks:
        return "No relevant excerpts were found for this search."
    blocks = []
    for i, chunk in enumerate(chunks, start=start):
        location = chunk.source + (f", page {chunk.page}" if chunk.page is not None else "")
        blocks.append(f"[{i}] ({location})\n{chunk.text}")
    return "\n\n".join(blocks)


def _query_from(tool_call: dict, fallback: str) -> str:
    try:
        query = json.loads(tool_call["function"]["arguments"]).get("query", "")
    except (KeyError, TypeError, json.JSONDecodeError):
        query = ""
    return query.strip() or fallback


def run_agent(
    question: str,
    generator: AnswerGenerator,
    retriever: Retriever,
    history: list[dict] | None = None,
    strict: bool = False,
    must_search: bool = False,
    model_name: str | None = None,
    temperature: float | None = None,
    max_searches: int = 3,
) -> AgentResult:
    """Answer `question`, letting the model search the documents as it sees fit."""
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT.format(
            app_name=APP_NAME, app_creator=APP_CREATOR, max_searches=max_searches,
            mode_rule=STRICT_RULE if strict else OPEN_RULE,
        ),
    }]
    messages += [{"role": t["role"], "content": t["content"]} for t in (history or [])[-_HISTORY_MESSAGES:]]
    base = list(messages)  # system + history, without tool turns (for the final answer)
    messages.append({"role": "user", "content": question})

    searches: list[str] = []
    sources: dict[str, RetrievedChunk] = {}
    excerpts: list[str] = []

    # One extra turn beyond max_searches for the final answer.
    for turn in range(max_searches + 1):
        limit_reached = len(searches) >= max_searches
        if limit_reached:
            # Final answer as a plain question with the excerpts inlined and no
            # tools: with earlier tool calls still in the conversation, models
            # keep trying to search again, which the API rejects.
            messages = base + [{"role": "user", "content": FINAL_ANSWER_PROMPT.format(
                question=question, searches=", ".join(f'"{q}"' for q in searches),
                excerpts="\n\n".join(excerpts) or "(none)",
            )}]
            tools, tool_choice = [], "none"
        elif turn == 0 and must_search:
            tools, tool_choice = [SEARCH_TOOL], "required"
        else:
            tools, tool_choice = [SEARCH_TOOL], "auto"
        try:
            message = generator.chat_with_tools(
                messages, tools, model_name=model_name, temperature=temperature,
                tool_choice=tool_choice,
            )
        except ToolCallingError as exc:
            raise AgentError(str(exc)) from exc

        tool_calls = message.get("tool_calls") or []
        if not tool_calls or limit_reached:
            answer = (message.get("content") or "").strip()
            if not answer:
                raise AgentError("The model returned an empty answer")
            return AgentResult(answer=answer, sources=list(sources.values()), searches=searches)

        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
        for call in tool_calls:
            if len(searches) >= max_searches:
                content = "Search limit reached; answer with what you have."
            else:
                query = _query_from(call, question)
                searches.append(query)
                found = retriever.retrieve(query)
                content = _format_results(found, start=len(sources) + 1)
                excerpts.append(content)
                for chunk in found:
                    sources.setdefault(chunk.chunk_id, chunk)
                logger.info("Agent search %d: %r -> %d chunks", len(searches), query, len(found))
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})

    raise AgentError("The agent did not finish within its turn limit")
