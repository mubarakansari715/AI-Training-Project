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

## Decisions still open

- [ ] Where the new tools live: inside the ALLY AI chat, or in a separate demo/notebook
- [ ] OK to add full LangChain (`langchain`, `langchain-groq`) as a dependency, which `create_agent` needs
- [ ] Web search provider: DuckDuckGo (`ddgs`, no key) or Tavily (`TAVILY_API_KEY`)

## Phase 1: Concepts notebook

- [ ] Add `notebooks/02_ai_agents.ipynb`: agents vs. chatbots, LLMs, prompts, tokens, context windows, design patterns
- [ ] Link each concept to the matching code in this project (table above)

## Phase 2: The four tools (`app/agent_tools/`)

- [ ] `multiply(a, b)`: pure Python
- [ ] `convert_currency(amount, from_currency, to_currency)`: Frankfurter API (free, no key)
- [ ] `get_weather(city)`: Open-Meteo (free, no key)
- [ ] `web_search(query)`: provider from the decision above
- [ ] JSON schema and input validation for each tool
- [ ] Tests with network calls mocked

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
- [ ] Input validation: currency codes, number ranges, city name length
- [ ] Human approval: Streamlit confirm before external calls (e.g. web search)
- [ ] Cost control: token budget per question and a cap on tool calls (usage already recorded in `_record_usage`)
- [ ] Prompt injection: wrap tool results in delimiters as data, system-prompt rule to ignore instructions inside them, tests with injected text

## Phase 6: UI

- [ ] Sidebar switches to turn each tool on/off and to choose the engine
- [ ] Show each tool call in the "Answered in N s" step list

## Phase 7: Optional extras

- [ ] Multi-agent: router agent that hands work to a documents agent and a web/tools agent
- [ ] MCP: expose `search_documents` as an MCP server
- [ ] Update `README.md` and the "How it works" page (`app/ui/how_it_works.py`)
