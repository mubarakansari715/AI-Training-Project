"""Agent page (design preview): tools, guardrails and a Think–Act–Observe trace.

Layout only — no tool or model is called yet. The trace below is a fixed
example so the page can be reviewed before the tools (AGENT_PLAN.md, Phase 2)
and engines (Phases 3–4) are built.
"""

from __future__ import annotations

import streamlit as st

TOOLS = {
    "multiply": ":material/calculate: Multiply",
    "currency": ":material/currency_exchange: Currency",
    "weather": ":material/partly_cloudy_day: Weather",
    "web_search": ":material/travel_explore: Web search",
    "documents": ":material/description: My documents",
}

SAMPLE_QUESTION = "What's 250 USD in INR, and what's the weather in Mumbai right now?"

# (kind, title, detail) — kind is think, act, observe or answer.
SAMPLE_TRACE = [
    ("think", "Plan", "Two separate facts: a currency conversion and the current weather. "
                      "I'll call convert_currency, then get_weather."),
    ("act", "convert_currency", '{"amount": 250, "from_currency": "USD", "to_currency": "INR"}'),
    ("observe", "Result", "250 USD = 20,912.50 INR (rate 83.65, Frankfurter, today)"),
    ("act", "get_weather", '{"city": "Mumbai"}'),
    ("observe", "Result", "31 °C, humidity 78%, light rain (Open-Meteo)"),
    ("think", "Check", "I have both facts. No more tools needed."),
    ("answer", "Answer", "250 USD is about **₹20,912.50** at today's rate of 83.65. "
                         "In Mumbai it's **31 °C** with light rain and 78% humidity."),
]

_STEP_ICONS = {
    "think": ":material/psychology:",
    "act": ":material/build:",
    "observe": ":material/visibility:",
    "answer": ":material/check_circle:",
}


def _render_settings() -> None:
    with st.container(border=True):
        st.markdown("**Engine**")
        st.segmented_control(
            "Engine", ["Manual", "LangChain"], default="Manual",
            key="agent_engine", label_visibility="collapsed",
        )
        st.caption("Manual: our own tool-calling loop. LangChain: `create_agent`.")

    with st.container(border=True):
        st.markdown("**Tools**")
        st.pills(
            "Tools", list(TOOLS), default=["multiply", "currency", "weather", "web_search"],
            format_func=TOOLS.get, selection_mode="multi",
            key="agent_tools", label_visibility="collapsed",
        )

    with st.container(border=True):
        st.markdown("**Guardrails**")
        st.slider("Max steps", 1, 15, 8, key="agent_max_steps",
                  help="Step limit (recursion_limit for LangChain).")
        st.number_input("Token budget per question", 500, 20000, 4000, step=500,
                        key="agent_token_budget")
        st.toggle("Ask before web search", value=True, key="agent_human_approval",
                  help="Human approval: confirm each external search before it runs.")
        st.toggle("Treat tool results as data", value=True, key="agent_injection_guard",
                  help="Prompt-injection guard: instructions inside tool results are ignored.")


def _render_trace(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status(f"Think → Act → Observe · {len(SAMPLE_TRACE) - 1} steps",
                       state="complete", expanded=True):
            for kind, title, detail in SAMPLE_TRACE[:-1]:
                label = f"{_STEP_ICONS[kind]} **{kind.capitalize()}** · {title}"
                st.markdown(label)
                if kind == "act":
                    st.code(detail, language="json")
                else:
                    st.caption(detail)
        st.markdown(SAMPLE_TRACE[-1][2])
        with st.container(horizontal=True):
            st.badge("2 tools", icon=":material/build:", color="blue")
            st.badge(f"Steps {len(SAMPLE_TRACE) - 1}/{st.session_state.agent_max_steps}",
                     icon=":material/footprint:", color="orange")
            st.badge(f"1,820 / {st.session_state.agent_token_budget:,} tokens",
                     icon=":material/token:", color="green")
            st.badge("2.4 s", icon=":material/timer:", color="gray")


def _render_concepts() -> None:
    with st.expander("How an agent works", icon=":material/school:"):
        cols = st.columns(4)
        blocks = [
            (":material/neurology:", "Model", "The LLM that thinks and picks tools (Groq)."),
            (":material/build:", "Tools", "Functions it can call: multiply, currency, weather, search."),
            (":material/history:", "Memory", "Earlier turns of the conversation."),
            (":material/rule:", "Instructions", "The system prompt: role, rules, limits."),
        ]
        for col, (icon, name, text) in zip(cols, blocks):
            with col.container(border=True):
                st.markdown(f"{icon} **{name}**")
                st.caption(text)
        st.markdown(
            "**Loop:** *Think* (decide what to do) → *Act* (call a tool) → *Observe* "
            "(read the result) → repeat until it can answer, or a guardrail stops it."
        )


def render_agent_page() -> None:
    st.title("Agent")
    st.caption("Ask anything that needs tools: maths, currency, weather or the web.")
    st.info("Design preview: the tools aren't connected yet, so every question shows "
            "the same example trace.", icon=":material/construction:")

    _render_concepts()

    settings_col, chat_col = st.columns([1, 2], gap="large")
    with settings_col:
        _render_settings()
    with chat_col:
        question = st.session_state.get("agent_last_question", SAMPLE_QUESTION)
        _render_trace(question)

    if asked := st.chat_input("Ask the agent…", key="agent_input"):
        st.session_state.agent_last_question = asked
        st.rerun()
