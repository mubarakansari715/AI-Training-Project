"""The Agent tab (app/ui/agent_page.py), run headless with a scripted model."""

import dataclasses

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from app.config.settings import settings
from app.rag import lc_agent
from app.ui import agent_page


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def page():
    from app.ui.agent_page import render_agent_page

    render_agent_page()


@pytest.fixture
def script(monkeypatch):
    """Replies the model will give, across every run of the page."""
    replies = []
    shared = iter(replies)  # one stream of replies even though each rerun builds a new model
    monkeypatch.setattr(lc_agent, "make_model", lambda config: ScriptedModel(messages=shared))
    monkeypatch.setattr(lc_agent, "run_tool", lambda name, args: f"{name} result")
    use_key(monkeypatch, "test-key")
    return replies


def use_key(monkeypatch, key):
    patched = dataclasses.replace(settings, groq_api_key=key)
    monkeypatch.setattr(lc_agent, "settings", patched)
    monkeypatch.setattr(agent_page, "settings", patched)


def call(name, args, call_id="c1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}],
                     usage_metadata={"input_tokens": 90, "output_tokens": 10, "total_tokens": 100})


def test_first_render_shows_settings_and_suggestions(script):
    at = AppTest.from_function(page, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Agent"
    assert at.toggle(key="agent_human_approval").value is False  # the agent decides on its own
    assert at.slider(key="agent_max_steps").value == settings.agent_max_steps


def test_question_shows_trace_answer_and_badges(script):
    script += [call("multiply", {"a": 6, "b": 7}), AIMessage("It is **42** (multiply).")]
    at = AppTest.from_function(page, default_timeout=30).run()
    at.chat_input(key="agent_input").set_value("6 times 7?").run()

    assert not at.exception
    assert any("It is **42**" in m.value for m in at.markdown)
    run = at.session_state.agent_runs[0]
    # model, tools, model (no approval step: asking first is off by default)
    assert run.status == "done" and run.tool_calls == 1 and run.steps == 3


def test_web_search_asks_for_approval_then_answers(script):
    script += [call("web_search", {"query": "news"}), AIMessage("Here is the news (web search).")]
    at = AppTest.from_function(page, default_timeout=30).run()
    at.toggle(key="agent_human_approval").set_value(True).run()
    at.chat_input(key="agent_input").set_value("Any news?").run()

    assert at.session_state.agent_runs[0].status == "waiting"
    approve = next(b for b in at.button if b.label == "Approve")
    approve.click().run()

    assert not at.exception
    run = at.session_state.agent_runs[0]
    assert run.status == "done" and run.answer == "Here is the news (web search)."


def test_rejecting_web_search(script):
    script += [call("web_search", {"query": "news"}), AIMessage("OK, I won't search.")]
    at = AppTest.from_function(page, default_timeout=30).run()
    at.toggle(key="agent_human_approval").set_value(True).run()
    at.chat_input(key="agent_input").set_value("Any news?").run()
    next(b for b in at.button if b.label == "Reject").click().run()

    run = at.session_state.agent_runs[0]
    assert run.status == "done"
    assert [s.title for s in run.trace if s.kind == "approval"][-1] == "Rejected"


def test_missing_key_is_reported(monkeypatch):
    use_key(monkeypatch, "")
    at = AppTest.from_function(page, default_timeout=30).run()
    at.chat_input(key="agent_input").set_value("hi").run()

    assert any("GROQ_API_KEY" in e.value for e in at.error)
    assert at.session_state.agent_runs[0].status == "error"
