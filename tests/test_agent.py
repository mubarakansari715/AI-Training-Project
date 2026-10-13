import json

import pytest

from app.rag.agent import AgentError, run_agent
from app.rag.generator import ToolCallingError
from app.rag.retriever import RetrievedChunk


def _chunk(cid, page, text="fact"):
    return RetrievedChunk(text=text, source="report.pdf", page=page, score=0.5, chunk_id=cid)


def _call(query, call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": "search_documents", "arguments": json.dumps({"query": query})}}


class ScriptedGenerator:
    """Fake Groq: replays scripted assistant messages, records each turn."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.turns = []

    def chat_with_tools(self, messages, tools, model_name=None, temperature=None, tool_choice="auto"):
        self.turns.append({"messages": [dict(m) for m in messages], "tool_choice": tool_choice, "model": model_name,
                           "tools": list(tools)})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.results.get(query, [])


def test_answers_without_searching_when_the_model_decides_not_to():
    gen = ScriptedGenerator([{"content": "Hello! How can I help?"}])
    result = run_agent("hi", gen, FakeRetriever({}))

    assert result.answer == "Hello! How can I help?"
    assert result.searches == [] and result.sources == []
    assert gen.turns[0]["tool_choice"] == "auto"


def test_multi_search_comparison_collects_deduplicated_sources():
    gen = ScriptedGenerator([
        {"content": None, "tool_calls": [_call("How many passports did GPO produce?", "a"),
                                         _call("How many staff did GPO have?", "b")]},
        {"content": "15,901,600 passports (report.pdf, page 14) and 1,737 staff (report.pdf, page 14)."},
    ])
    shared = _chunk("p14", 14, "GPO staff 1,737. Passports produced 15,901,600.")
    retriever = FakeRetriever({"How many passports did GPO produce?": [shared],
                               "How many staff did GPO have?": [shared, _chunk("p15", 15)]})

    result = run_agent("Compare passports and staff", gen, retriever, model_name="openai/gpt-oss-120b")

    assert result.searches == ["How many passports did GPO produce?", "How many staff did GPO have?"]
    assert [c.chunk_id for c in result.sources] == ["p14", "p15"]
    tool_messages = [m for m in gen.turns[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert "(report.pdf, page 14)" in tool_messages[0]["content"]
    assert gen.turns[0]["model"] == "openai/gpt-oss-120b"


def test_search_limit_forces_a_final_answer():
    gen = ScriptedGenerator([
        {"content": None, "tool_calls": [_call("q1")]},
        {"content": None, "tool_calls": [_call("q2")]},
        {"content": "Final answer."},
    ])
    result = run_agent("question", gen, FakeRetriever({}), max_searches=2)

    assert result.searches == ["q1", "q2"]
    assert [t["tool_choice"] for t in gen.turns] == ["auto", "auto", "none"]
    assert result.answer == "Final answer."


def test_extra_calls_beyond_the_limit_are_not_executed():
    gen = ScriptedGenerator([
        {"content": None, "tool_calls": [_call("q1", "a"), _call("q2", "b"), _call("q3", "c")]},
        {"content": "Done."},
    ])
    retriever = FakeRetriever({})
    run_agent("question", gen, retriever, max_searches=2)

    assert retriever.queries == ["q1", "q2"]
    final_prompt = gen.turns[1]["messages"][-1]["content"]
    assert '"q1", "q2"' in final_prompt and "q3" not in final_prompt


def test_strict_must_search_forces_a_tool_call_first():
    gen = ScriptedGenerator([{"content": None, "tool_calls": [_call("vitamin D")]}, {"content": "18.4 ng/mL."}])
    run_agent("what is my vitamin D?", gen, FakeRetriever({}), strict=True, must_search=True)

    assert gen.turns[0]["tool_choice"] == "required"
    assert "Strict mode" in gen.turns[0]["messages"][0]["content"]


def test_bad_tool_arguments_fall_back_to_the_question():
    bad = {"id": "x", "function": {"name": "search_documents", "arguments": "{not json"}}
    gen = ScriptedGenerator([{"content": None, "tool_calls": [bad]}, {"content": "ok"}])
    retriever = FakeRetriever({})
    run_agent("original question", gen, retriever)

    assert retriever.queries == ["original question"]


def test_history_is_included_and_bounded():
    history = [{"role": "user", "content": f"m{i}"} for i in range(15)]
    gen = ScriptedGenerator([{"content": "ok"}])
    run_agent("and my cholesterol?", gen, FakeRetriever({}), history=history)

    contents = [m["content"] for m in gen.turns[0]["messages"]]
    assert "m4" not in contents and "m5" in contents  # last 10 only
    assert contents[-1] == "and my cholesterol?"


@pytest.mark.parametrize("reply", [ToolCallingError("429 rate limited"), {"content": "   "}])
def test_provider_failure_or_empty_answer_raises_agent_error(reply):
    with pytest.raises(AgentError):
        run_agent("question", ScriptedGenerator([reply]), FakeRetriever({}))


def test_final_turn_after_search_limit_sends_no_tools():
    gen = ScriptedGenerator([{"content": None, "tool_calls": [_call("q1")]}, {"content": "Not in the documents."}])
    run_agent("question", gen, FakeRetriever({}), max_searches=1)

    final = gen.turns[-1]
    assert final["tools"] == []
    # Plain conversation (no tool turns) with the found excerpts inlined:
    # leftover tool calls prime the model to search again, which the API rejects.
    assert [m["role"] for m in final["messages"]] == ["system", "user"]
    assert '"q1"' in final["messages"][-1]["content"]
    assert final["messages"][-1]["content"].startswith("question")
