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


# --- Extra tools from app/agent_tools ---

def _tool_call(name, arguments, call_id="t1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _tool_names(turn):
    return [t["function"]["name"] for t in turn["tools"]]


def test_document_chat_only_offers_search_documents():
    gen = ScriptedGenerator([{"content": "ok"}])
    run_agent("question", gen, FakeRetriever({}))

    assert _tool_names(gen.turns[0]) == ["search_documents"]
    assert "You also have these tools" not in gen.turns[0]["messages"][0]["content"]


def test_extra_tools_are_offered_run_and_recorded():
    gen = ScriptedGenerator([
        {"content": None, "tool_calls": [_tool_call("multiply", {"a": 6, "b": 7}, "m"), _call("tax rate", "s")]},
        {"content": "6 × 7 = 42 (multiply)."},
    ])
    result = run_agent("6 times 7, and the tax rate?", gen, FakeRetriever({"tax rate": [_chunk("p3", 3)]}),
                       extra_tools=["multiply", "get_weather"])

    assert _tool_names(gen.turns[0]) == ["search_documents", "multiply", "get_weather"]
    assert "You also have these tools: multiply, get_weather" in gen.turns[0]["messages"][0]["content"]
    tool_messages = [m for m in gen.turns[1]["messages"] if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_messages][0] == ("m", "6 × 7 = 42")
    assert [(r.name, r.arguments) for r in result.tool_calls] == [("multiply", {"a": 6, "b": 7}),
                                                                   ("search_documents", {"query": "tax rate"})]
    assert result.searches == ["tax rate"] and result.steps == 2


def test_tool_errors_are_fed_back_so_the_model_can_retry():
    gen = ScriptedGenerator([
        {"content": None, "tool_calls": [_tool_call("multiply", {"a": "six", "b": 7})]},
        {"content": None, "tool_calls": [_tool_call("multiply", {"a": 6, "b": 7})]},
        {"content": "42"},
    ])
    result = run_agent("6 times 7", gen, FakeRetriever({}), extra_tools=["multiply"])

    first_result = [m for m in gen.turns[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert first_result.startswith("Error: invalid input for multiply")
    assert [r.result for r in result.tool_calls] == [first_result, "6 × 7 = 42"]
    assert result.answer == "42"


def test_calls_to_tools_that_are_not_enabled_are_refused(monkeypatch):
    monkeypatch.setattr("app.rag.agent.run_tool", lambda *a: pytest.fail("disabled tool was run"))
    gen = ScriptedGenerator([{"content": None, "tool_calls": [_tool_call("get_weather", {"city": "Mumbai"})]},
                             {"content": "I can't check the weather."}])
    result = run_agent("weather in Mumbai?", gen, FakeRetriever({}))

    tool_message = [m for m in gen.turns[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert tool_message.startswith("Error: unknown tool 'get_weather'")
    assert result.tool_calls == []


def test_unknown_extra_tool_names_are_rejected_up_front():
    with pytest.raises(ValueError, match="divide"):
        run_agent("q", ScriptedGenerator([]), FakeRetriever({}), extra_tools=["multiply", "divide"])


def test_step_limit_forces_a_final_answer_with_tool_results_inlined():
    gen = ScriptedGenerator([
        {"content": None, "tool_calls": [_tool_call("multiply", {"a": 2, "b": 3})]},
        {"content": None, "tool_calls": [_tool_call("multiply", {"a": 6, "b": 4})]},
        {"content": "24."},
    ])
    result = run_agent("2 × 3 × 4", gen, FakeRetriever({}), extra_tools=["multiply"], max_steps=3)

    final = gen.turns[-1]
    assert final["tools"] == [] and [m["role"] for m in final["messages"]] == ["system", "user"]
    prompt = final["messages"][-1]["content"]
    assert 'multiply {"a": 2, "b": 3}' in prompt and "6 × 4 = 24" in prompt
    assert result.steps == 3 and result.answer == "24."
