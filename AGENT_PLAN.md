# AI Agents Plan

Plan for adding the "AI agents" topics to ALLY AI. Tick each box as it's done.

## Topics to cover

- Introduction to AI agents and how they differ from chatbots
- Key concepts: LLMs, prompts, tokens, context windows, tools, and agents
- The Think–Act–Observe loop and the four building blocks of an agent: Model, Tools, Memory, and Instructions
- Tool calling, workflows vs. agents, design patterns, multi-agent systems, MCP, and AI frameworks
- Agent guardrails: step limits, input validation, human approval, cost control, and prompt-injection prevention
- Hands-on demo: multiplication, currency conversion, web search, and weather tools; manual tool calling; an agent built with `create_agent` and a `recursion_limit` guardrail

## Already in the project

| Topic | Where |
|---|---|
| Think–Act–Observe loop | `run_agent()` in `app/rag/agent.py` |
| Model | Groq, through `AnswerGenerator.chat_with_tools()` in `app/rag/generator.py` |
| Tools | `search_documents` (the only tool so far) |
| Memory | Last 10 chat messages, plus conversations saved to disk |
| Instructions | `SYSTEM_PROMPT` in `app/rag/agent.py` |
| Manual tool calling | Raw OpenAI-compatible tool calls to Groq, no framework |
| Step-limit guardrail | `AGENT_MAX_SEARCHES` (default 3) |
| Workflow vs. agent | Standard pipeline (rewrite → retrieve → rerank → answer) is a workflow; Auto mode is an agent |

## Engines: manual vs. LangChain

Both engines take a question, let the model call tools (multiply, currency, weather,
web search), and return an answer. The difference is **who writes the
Think → Act → Observe loop**.

### Manual engine: we write the loop

Already done this way in `app/rag/agent.py`. Our code handles every step:

```python
messages = [system_prompt, user_question]
for step in range(max_steps):                       # guardrail: step limit
    reply = groq.chat(messages, tools=TOOL_SCHEMAS)  # THINK: model decides
    if not reply.tool_calls:
        return reply.content                         # done, final answer
    for call in reply.tool_calls:
        result = TOOLS[call.name](**call.args)       # ACT: run the Python function
        messages.append(tool_result(call, result))   # OBSERVE: feed result back
raise StepLimitReached
```

Each tool's JSON schema is written by hand:

```python
{"name": "get_weather", "parameters": {"type": "object",
  "properties": {"city": {"type": "string"}}, "required": ["city"]}}
```

**Role:** shows exactly how an agent works under the hood. No magic and full control, but
more code to write and maintain.

### LangChain engine: the framework runs the loop

We only give it the model, the tools and the instructions:

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""   # schema built from this automatically
    ...

agent = create_agent(
    model=ChatGroq(model="openai/gpt-oss-20b"),
    tools=[multiply, convert_currency, get_weather, web_search],
    system_prompt="You are a helpful assistant...",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"recursion_limit": 10},   # guardrail: stop runaway loops
)
```

- No loop code: LangChain decides when to call the model and when to run tools.
- No hand-written schemas: `@tool` builds them from the type hints and docstring.
- `recursion_limit` is the framework's step limit. When it's hit, LangChain raises
  `GraphRecursionError`, which we catch and show as "stopped by guardrail". Each model call
  and each round of tool calls counts as a step, so a 2-tool question uses about 5.

**Role:** the standard industry way, with less code and built-in extras (middleware,
human-approval hooks, memory). Less visibility inside, plus a new dependency.

### Side by side

| | Manual | LangChain |
|---|---|---|
| Who runs the loop | Our code | `create_agent` |
| Tool schemas | Written by hand (JSON) | Built automatically from `@tool` |
| Step limit | Our `max_steps` counter | `recursion_limit` |
| Code size | More | Less |
| Learning value | See how agents really work | See how frameworks hide that work |
| New dependency | None | `langchain`, `langchain-groq` |

### The `AGENT_ENGINE` setting

Both engines use the same four tools and the same Groq model; the setting only picks which
loop runs.

- `AGENT_ENGINE=manual|langchain` in `.env` sets the default.
- The Manual / LangChain switch in the Agent tab changes it per question.

Ask the same question with each and compare the trace, answer, steps, tokens and time.
That is the hands-on part: "manual tool calling" vs. "`create_agent` with
`recursion_limit`". The document chat (Auto mode) keeps the manual engine either way.

## Decisions still open

- [x] Where the new tools live: a separate **Agent** tab (Option A); the document chat stays as it is
- [ ] OK to add full LangChain (`langchain`, `langchain-groq`) as a dependency, which `create_agent` needs
- [x] Web search provider: DuckDuckGo (`ddgs`, no key), so the demo runs without any API keys

## Phase 1: Concepts notebook

- [ ] Add `notebooks/02_ai_agents.ipynb`: agents vs. chatbots, LLMs, prompts, tokens, context windows, design patterns
- [ ] Link each concept to the matching code in this project (table above)

## Phase 2: The four tools (`app/agent_tools/`)

- [x] `multiply(a, b)`: pure Python
- [x] `convert_currency(amount, from_currency, to_currency)`: Frankfurter API (free, no key)
- [x] `get_weather(city)`: Open-Meteo (free, no key)
- [x] `web_search(query)`: DuckDuckGo via `ddgs`
- [x] JSON schema and input validation for each tool; `run_tool()` turns bad calls and failures into error text for the model
- [x] Tests with network calls mocked (`tests/test_agent_tools.py`)

## Phase 3: Manual tool calling

- [ ] Change `run_agent` from one hard-coded tool to a tool registry (name → function + schema)
- [ ] Register `search_documents` plus the four new tools
- [ ] Keep using the existing `chat_with_tools()` (no new dependencies)
- [ ] Update `tests/test_agent.py`

## Phase 4: `create_agent` with `recursion_limit`

- [ ] Check `langchain` / `langchain-groq` fit the torch 2.2.2 and `numpy<2` pins; add them to `requirements.txt`
- [ ] Add `app/rag/lc_agent.py`: tools via `@tool`, `create_agent(model, tools, system_prompt)`
- [ ] Run with `.invoke(..., config={"recursion_limit": N})`; catch `GraphRecursionError`
- [ ] Add `AGENT_ENGINE=manual|langchain` setting to compare both engines
- [ ] Tests

## Phase 5: Guardrails

- [ ] Step limit: keep max searches; add `recursion_limit` for the LangChain engine
- [x] Input validation: currency codes, number ranges, city name length (done with the tools in Phase 2)
- [ ] Human approval: Streamlit confirm before external calls (e.g. web search)
- [ ] Cost control: token budget per question and a cap on tool calls (usage already recorded in `_record_usage`)
- [ ] Prompt injection: wrap tool results in delimiters as data, system-prompt rule to ignore instructions inside them, tests with injected text

## Phase 6: Agent tab UI (`app/ui/agent_page.py`)

- [x] Design preview: third top tab, engine switch, tool chips, guardrail controls, example Think/Act/Observe trace
- [ ] Wire the settings and chat box to the real engines and tools
- [ ] Show the live trace, steps, tokens and time for each question

## Phase 7: Optional extras

- [ ] Multi-agent: router agent that hands work to a documents agent and a web/tools agent
- [ ] MCP: expose `search_documents` as an MCP server
- [ ] Update `README.md` and the "How it works" page (`app/ui/how_it_works.py`)
