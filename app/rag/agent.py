"""Auto (agent) mode: the LLM decides whether and how to search.

Instead of always running one search before answering, the model gets a
`search_documents` tool — the way ChatGPT and Claude use tools — and
decides per question:

- no search for greetings or questions it can answer without the documents;
- one search for a single fact;
- several searches, with its own wording, for comparisons and multi-part
  questions ("South Asia vs Sub-Saharan Africa growth"), or to retry with
  different words when the first results don't contain the answer.

Tools come from a registry: `search_documents` is always there, and
`extra_tools` adds any of the agent tools in app/agent_tools (multiply,
convert_currency, get_weather, web_search) by name. The document chat
passes none, so it only ever searches the documents.

Guards keep it predictable: at most `max_searches` searches and `max_steps`
model turns; in strict mode the first turn must search (unless it's small
talk); facts about the documents may only come from search results; and
any provider error raises AgentError so the caller can fall back to the
standard pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.agent_tools import TOOLS, run_tool
from app.config.settings import APP_CREATOR, APP_NAME
from app.rag.generator import AnswerGenerator, ToolCallingError
from app.rag.prompt import FORMAT_RULE
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
- Excerpts from structured reports start with their section in square brackets \
(e.g. "[Ch. 2 Regional Outlooks › South Asia]"): take facts only from the sections the \
question is about, never mix up sections (e.g. two regions), and name the section in citations.
- {mode_rule}
- {format_rule}
- Only if the user asks who you are, say you are {app_name}; if asked who built or owns \
{app_name}, say it was built by {app_creator}."""

STRICT_RULE = ("Strict mode: answer only from the search results. If they don't contain the "
               "answer, say the information was not found in the uploaded documents.")
OPEN_RULE = ("If the documents don't contain the answer, you may answer from general "
             "knowledge, but say clearly that this part is not from the documents.")

# Appended to the system prompt when extra tools are on.
EXTRA_TOOLS_RULE = """

You also have these tools: {tool_names}. Use them for what they are for (maths, \
currency conversion, weather, the web) instead of answering from memory. Their results \
are not from the documents: say which tool a fact came from, e.g. "(get_weather)" or \
"(web search: example.com)". If a tool returns an error, fix the arguments and try again, \
or explain what went wrong."""


FINAL_ANSWER_PROMPT = """{question}

You already used your tools ({calls}); these are the results:

{results}

Answer the question now, using only these results (follow the rules above). \
If they don't contain the answer, say so."""


class AgentError(Exception):
    """The agent couldn't produce an answer (provider failure, empty reply)."""


@dataclass
class ToolCallRecord:
    """One executed tool call (Act) and what it returned (Observe)."""

    name: str
    arguments: dict
    result: str


@dataclass
class AgentResult:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0  # model turns used


def _format_results(chunks: list[RetrievedChunk], start: int) -> str:
    if not chunks:
        return "No relevant excerpts were found for this search."
    blocks = []
    for i, chunk in enumerate(chunks, start=start):
        location = chunk.source + (f", page {chunk.page}" if chunk.page is not None else "")
        blocks.append(f"[{i}] ({location})\n{chunk.text}")
    return "\n\n".join(blocks)


def _arguments_from(tool_call: dict) -> dict:
    try:
        arguments = json.loads(tool_call["function"]["arguments"] or "{}")
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}
    return arguments if isinstance(arguments, dict) else {}


def _query_from(tool_call: dict, fallback: str) -> str:
    query = _arguments_from(tool_call).get("query", "")
    return (query.strip() if isinstance(query, str) else "") or fallback


def _describe(record: ToolCallRecord) -> str:
    """How a call is listed in the final-answer prompt: a search by its query, others by name and arguments."""
    if record.name == SEARCH_TOOL["function"]["name"]:
        return f'"{record.arguments["query"]}"'
    return f"{record.name} {json.dumps(record.arguments, ensure_ascii=False)}"


def _result_block(record: ToolCallRecord) -> str:
    """A call's result for the final-answer prompt; search excerpts carry their own citations."""
    if record.name == SEARCH_TOOL["function"]["name"]:
        return record.result
    return f"Result of {_describe(record)}:\n{record.result}"


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
    extra_tools: list[str] | None = None,
    max_steps: int | None = None,
) -> AgentResult:
    """Answer `question`, letting the model search the documents (and use any
    `extra_tools`) as it sees fit, in at most `max_steps` model turns
    (default: one per allowed search plus one for the answer)."""
    extra_tools = list(extra_tools or [])
    unknown = [name for name in extra_tools if name not in TOOLS]
    if unknown:
        raise ValueError(f"Unknown agent tools: {', '.join(unknown)}")
    # The registry: tool name → schema; search_documents is handled below,
    # every other tool runs through app.agent_tools.run_tool.
    tool_schemas = [SEARCH_TOOL] + [TOOLS[name].schema for name in extra_tools]
    max_steps = max_steps if max_steps is not None else max_searches + 1

    system_prompt = SYSTEM_PROMPT.format(
        app_name=APP_NAME, app_creator=APP_CREATOR, max_searches=max_searches,
        mode_rule=STRICT_RULE if strict else OPEN_RULE,
        format_rule=" ".join(FORMAT_RULE.split()),
    )
    if extra_tools:
        system_prompt += EXTRA_TOOLS_RULE.format(tool_names=", ".join(extra_tools))
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": t["role"], "content": t["content"]} for t in (history or [])[-_HISTORY_MESSAGES:]]
    base = list(messages)  # system + history, without tool turns (for the final answer)
    messages.append({"role": "user", "content": question})

    searches: list[str] = []
    sources: dict[str, RetrievedChunk] = {}
    records: list[ToolCallRecord] = []

    for turn in range(max_steps):
        # Searches used up, or the last allowed turn: answer now.
        limit_reached = len(searches) >= max_searches or turn == max_steps - 1
        if limit_reached:
            # Final answer as a plain question with the tool results inlined
            # and no tools: with earlier tool calls still in the conversation,
            # models keep trying to call tools again, which the API rejects.
            messages = base + [{"role": "user", "content": FINAL_ANSWER_PROMPT.format(
                question=question, calls=", ".join(_describe(r) for r in records),
                results="\n\n".join(_result_block(r) for r in records) or "(none)",
            )}]
            tools, tool_choice = [], "none"
        elif turn == 0 and must_search:
            tools, tool_choice = tool_schemas, "required"
        else:
            tools, tool_choice = tool_schemas, "auto"
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
            return AgentResult(answer=answer, sources=list(sources.values()), searches=searches,
                               tool_calls=records, steps=turn + 1)

        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
        for call in tool_calls:
            name = (call.get("function") or {}).get("name")
            if name == SEARCH_TOOL["function"]["name"]:
                if len(searches) >= max_searches:
                    content = "Search limit reached; answer with what you have."
                else:
                    query = _query_from(call, question)
                    searches.append(query)
                    found = retriever.retrieve(query)
                    content = _format_results(found, start=len(sources) + 1)
                    records.append(ToolCallRecord(name, {"query": query}, content))
                    for chunk in found:
                        sources.setdefault(chunk.chunk_id, chunk)
                    logger.info("Agent search %d: %r -> %d chunks", len(searches), query, len(found))
            elif name in extra_tools:
                arguments = _arguments_from(call)
                content = run_tool(name, call["function"].get("arguments"))
                records.append(ToolCallRecord(name, arguments, content))
                logger.info("Agent tool %s(%s) -> %r", name, arguments, content[:120])
            else:
                content = f"Error: unknown tool {name!r}. Available tools: " + ", ".join(
                    s["function"]["name"] for s in tool_schemas)
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})

    raise AgentError("The agent did not finish within its turn limit")
