"""Agent page: ask questions that need tools, answered by LangChain's `create_agent`.

The left column sets up the agent (model, tools, guardrails); the right
column is the chat, where each answer shows its Think → Act → Observe trace
live, then the steps, tokens and time it used. A question that wants a
gated tool (web search, by default) pauses for the user's approval
(human in the loop) and continues when they approve or reject it.

The engine is app.rag.lc_agent; this module is only the UI around it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from app.config.settings import settings
from app.rag.lc_agent import (
    DOCUMENTS_TOOL,
    AgentConfig,
    AgentEngine,
    AgentEngineError,
    AgentRun,
    TraceStep,
)

logger = logging.getLogger(__name__)

TOOL_LABELS = {
    "multiply": ":material/calculate: Multiply",
    "convert_currency": ":material/currency_exchange: Currency",
    "get_weather": ":material/partly_cloudy_day: Weather",
    "web_search": ":material/travel_explore: Web search",
    DOCUMENTS_TOOL: ":material/description: My documents",
}
DEFAULT_TOOLS = ["multiply", "convert_currency", "get_weather", "web_search"]

SUGGESTIONS = {
    ":material/currency_exchange: 250 USD in INR + Mumbai weather":
        "What's 250 USD in INR, and what's the weather in Mumbai right now?",
    ":material/calculate: 1,234.5 × 987": "What is 1,234.5 multiplied by 987?",
    ":material/travel_explore: Latest Streamlit release": "What is the latest Streamlit release? Search the web.",
}

_STEP_ICONS = {
    "think": ":material/psychology:",
    "act": ":material/build:",
    "observe": ":material/visibility:",
    "approval": ":material/how_to_reg:",
    "guardrail": ":material/block:",
}

# Long tool results and reasoning are cut for display (the model saw them in full).
_SHOW_CHARS = 2000


def _init_state() -> None:
    state = st.session_state
    state.setdefault("agent_runs", [])
    # Paused runs (waiting for approval) are kept here between reruns.
    state.setdefault("agent_checkpointer", InMemorySaver())


def _clear() -> None:
    st.session_state.agent_runs = []
    st.session_state.agent_checkpointer = InMemorySaver()
    st.session_state.pop("agent_suggestion", None)  # else the old pick would ask again


def _decide(approved: bool) -> None:
    st.session_state.agent_action = ("resume", approved, st.session_state.get("agent_reject_reason", ""))


# --- Settings -------------------------------------------------------------------------

def _render_settings(locked: bool) -> AgentConfig:
    with st.container(border=True):
        st.markdown("**Model**")
        models = list(dict.fromkeys([settings.agent_model, settings.groq_strong_model]))
        model = st.segmented_control(
            "Model", models, default=models[0], key="agent_model", required=True,
            format_func=lambda m: m.split("/")[-1], label_visibility="collapsed", disabled=locked,
        )
        st.caption("Groq, through LangChain's `create_agent`.")

    with st.container(border=True):
        st.markdown("**Tools**")
        tools = st.pills(
            "Tools", list(TOOL_LABELS), default=DEFAULT_TOOLS, format_func=TOOL_LABELS.get,
            selection_mode="multi", key="agent_tools", label_visibility="collapsed", disabled=locked,
        )
        st.caption("Web search can also open result pages to read them. Each `@tool` gets its "
                   "JSON schema from its type hints and docstring.")

    with st.container(border=True):
        st.markdown("**Guardrails**")
        max_steps = st.slider(
            "Max steps (recursion_limit)", 3, 30, settings.agent_max_steps, key="agent_max_steps",
            disabled=locked, help="Every model call, round of tool calls and approval check is a "
                                  "step. The run stops when the limit is hit.")
        token_budget = st.number_input(
            "Token budget per question", 500, 50000, settings.agent_token_budget, step=500,
            key="agent_token_budget", disabled=locked,
            help="Cost control: no more model calls once this many tokens are used.")
        max_tool_calls = st.number_input(
            "Max tool calls per question", 1, 20, settings.agent_max_tool_calls, key="agent_max_tool_calls",
            disabled=locked, help="Cost control: further calls get an error asking the model to answer.")
        approve = st.toggle(
            "Ask before web search", value=False, key="agent_human_approval", disabled=locked,
            help="Off: the agent decides on its own. On (human in the loop): the run pauses "
                 "until you approve or reject each web search.")
        injection_guard = st.toggle(
            "Treat tool results as data", value=True, key="agent_injection_guard", disabled=locked,
            help="Prompt-injection guard: tool results are wrapped in <tool_result> tags and "
                 "instructions inside them are ignored.")

    st.button("Clear conversation", icon=":material/delete_sweep:", on_click=_clear,
              width="stretch", disabled=not st.session_state.agent_runs)

    return AgentConfig(
        tools=tuple(tools), model=model or "", max_steps=int(max_steps), token_budget=int(token_budget),
        max_tool_calls=int(max_tool_calls), approve_tools=("web_search",) if approve else (),
        injection_guard=injection_guard,
    )


# --- Trace and answers ------------------------------------------------------------------

def _render_step(step: TraceStep, waiting: bool = False) -> None:
    if step.kind == "answer":
        return  # shown below the trace
    failed = step.kind == "guardrail" or step.title.endswith("· error") or step.title == "Rejected"
    state = "running" if waiting and step.title.startswith("Waiting") else ("error" if failed else "complete")
    label = f"{_STEP_ICONS.get(step.kind, '')} **{step.kind.capitalize()}** · {step.title}"
    with st.status(label, type="step", state=state):
        detail = step.detail if len(step.detail) <= _SHOW_CHARS else step.detail[:_SHOW_CHARS] + " …"
        if step.kind == "act" or (step.kind == "approval" and step.title.startswith("Waiting")):
            st.code(detail, language="json", wrap_lines=True)
        elif step.kind == "observe":
            st.code(detail, language=None, wrap_lines=True)
        elif detail:
            st.caption(detail)


def _trace_label(run: AgentRun) -> str:
    if run.status == "waiting":
        return "Waiting for your approval"
    if run.status == "error":
        return "Agent failed"
    return f"Think → Act → Observe · {run.steps} steps, {run.seconds:.1f} s"


def _trace_state(run: AgentRun) -> str:
    return {"waiting": "running", "error": "error", "stopped": "error"}.get(run.status, "complete")


def _render_outcome(run: AgentRun) -> None:
    if run.answer:
        st.markdown(run.answer)
    if run.status == "stopped":
        st.warning(run.stop_reason or "Stopped by a guardrail.", title="Stopped by guardrail",
                   icon=":material/block:")
    elif run.status == "error":
        st.error(run.stop_reason or "The agent failed.", icon=":material/error:")
    elif run.status == "waiting":
        _render_approval(run)
    if run.steps or run.tokens:
        budget = run.config.token_budget
        with st.container(horizontal=True):
            st.badge(f"{run.tool_calls} tool calls", icon=":material/build:", color="blue")
            st.badge(f"Steps {run.steps}/{run.config.max_steps}", icon=":material/footprint:", color="orange")
            st.badge(f"{run.tokens:,} / {budget:,} tokens" if budget else f"{run.tokens:,} tokens",
                     icon=":material/token:", color="green")
            st.badge(f"{run.seconds:.1f} s", icon=":material/timer:", color="gray")


def _render_approval(run: AgentRun) -> None:
    with st.container(border=True):
        st.markdown(":material/how_to_reg: **The agent wants to run:**")
        for request in run.pending:
            st.markdown(f"`{request['name']}`")
            st.code(str(request.get("args", {})), language="json", wrap_lines=True)
        st.text_input("Reason (optional, sent to the agent if you reject)", key="agent_reject_reason")
        with st.container(horizontal=True):
            st.button("Approve", icon=":material/check:", type="primary", on_click=_decide, args=(True,))
            st.button("Reject", icon=":material/close:", on_click=_decide, args=(False,))


def _render_run(run: AgentRun) -> None:
    with st.chat_message("user"):
        st.markdown(run.question)
    with st.chat_message("assistant"):
        if run.trace:
            with st.status(_trace_label(run), type="compact", state=_trace_state(run),
                           expanded=run.status == "waiting"):
                for step in run.trace:
                    _render_step(step, waiting=run.status == "waiting")
        _render_outcome(run)


def _render_live(run: AgentRun, steps: Iterator[TraceStep], show_question: bool) -> None:
    """Show `steps` as they arrive; the full message is redrawn on the next rerun."""
    if show_question:
        with st.chat_message("user"):
            st.markdown(run.question)
    with st.chat_message("assistant"):
        with st.status(":shimmer[Thinking]", type="compact", expanded=True) as status:
            for step in list(run.trace):
                _render_step(step)
            for step in steps:
                _render_step(step)
            status.update(label=_trace_label(run), state=_trace_state(run))


# --- Running the engine ------------------------------------------------------------------

def _engine(config: AgentConfig, get_retriever: Callable | None) -> AgentEngine:
    retriever = None
    if DOCUMENTS_TOOL in config.tools:
        if get_retriever is None:
            raise AgentEngineError("The document index isn't available on this page.")
        retriever = get_retriever()
    return AgentEngine(config, retriever=retriever, checkpointer=st.session_state.agent_checkpointer)


def _history(runs: list[AgentRun]) -> list[dict]:
    """Memory: earlier questions and their answers."""
    messages = []
    for run in runs:
        if run.answer:
            messages += [{"role": "user", "content": run.question}, {"role": "assistant", "content": run.answer}]
    return messages


def _fail(run: AgentRun, message: str) -> Iterator[TraceStep]:
    run.status, run.stop_reason = "error", message
    return iter(())


def _start(run: AgentRun, earlier: list[AgentRun], get_retriever: Callable | None) -> Iterator[TraceStep]:
    try:
        engine = _engine(run.config, get_retriever)
    except Exception as exc:  # no key, no tools, index failed to load
        logger.warning("Agent could not start: %s", exc)
        return _fail(run, str(exc))
    return engine.start(run, _history(earlier))


def _resume(run: AgentRun, approved: bool, reason: str, get_retriever: Callable | None) -> Iterator[TraceStep]:
    try:
        engine = _engine(run.config, get_retriever)
    except Exception as exc:
        return _fail(run, str(exc))
    return engine.resume(run, approved, reason)


# --- Page ----------------------------------------------------------------------------------

def _render_concepts() -> None:
    with st.expander("How an agent works", icon=":material/school:"):
        cols = st.columns(4)
        blocks = [
            (":material/neurology:", "Model", "The LLM that thinks and picks tools (Groq via ChatGroq)."),
            (":material/build:", "Tools", "`@tool` functions: multiply, currency, weather, web, documents."),
            (":material/history:", "Memory", "Earlier questions and answers on this page."),
            (":material/rule:", "Instructions", "The system prompt: role, rules, limits."),
        ]
        for col, (icon, name, text) in zip(cols, blocks):
            with col.container(border=True, height="stretch"):
                st.markdown(f"{icon} **{name}**")
                st.caption(text)
        st.markdown(
            "**Loop:** *Think* (decide what to do) → *Act* (call a tool) → *Observe* "
            "(read the result) → repeat until it can answer, or a guardrail stops it. "
            "LangChain's `create_agent` runs this loop; the guardrails are its "
            "`recursion_limit` and middleware (human approval, token budget, tool-call cap, "
            "tool results as data)."
        )


def render_agent_page(get_retriever: Callable | None = None) -> None:
    """The Agent tab. `get_retriever` returns the document retriever (for "My documents")."""
    _init_state()
    runs: list[AgentRun] = st.session_state.agent_runs
    waiting = any(run.status == "waiting" for run in runs)

    st.title("Agent")
    st.caption("Ask anything that needs tools: maths, currency, weather, the web or your documents.")
    if not settings.groq_api_key:
        st.error("Set GROQ_API_KEY in your .env file: the agent's model runs on Groq.", icon=":material/key:")
    _render_concepts()

    settings_col, chat_col = st.columns([1, 2], gap="large")
    with settings_col:
        config = _render_settings(locked=waiting)

    prompt = st.chat_input("Waiting for your approval above…" if waiting else "Ask the agent…",
                           key="agent_input", disabled=waiting, submit_mode="disable")
    action = st.session_state.pop("agent_action", None)

    with chat_col:
        if not runs and not prompt:
            picked = st.pills("Try asking", list(SUGGESTIONS), key="agent_suggestion",
                              label_visibility="collapsed")
            if picked:
                prompt = SUGGESTIONS[picked]

        resuming = action is not None and waiting
        for run in runs[:-1] if resuming else runs:
            _render_run(run)

        if prompt:
            run = AgentRun(question=prompt.strip(), config=config)
            earlier = list(runs)
            runs.append(run)
            _render_live(run, _start(run, earlier, get_retriever), show_question=True)
            st.rerun()
        elif resuming:
            run = runs[-1]
            _, approved, reason = action
            st.session_state.agent_reject_reason = ""
            _render_live(run, _resume(run, approved, reason, get_retriever), show_question=True)
            st.rerun()
