from unittest.mock import MagicMock

import pytest

from app.rag.generator import FALLBACK_MESSAGE, MISSING_KEY_MESSAGE, AnswerGenerator
from app.rag.retriever import RetrievedChunk


def _chunk(text="Employees get 24 days of leave.", source="handbook.pdf"):
    return RetrievedChunk(text=text, source=source, page=1, score=0.9, chunk_id="c1")


def _capturing_groq_post(captured, content="answer"):
    """A fake requests.post that records its kwargs and returns `content`."""

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        return response

    return fake_post


def _capturing_gemini_client(captured):
    """A fake genai.Client whose generate_content records its kwargs."""

    def fake_generate_content(model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        response = MagicMock()
        response.text = "answer"
        return response

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = fake_generate_content
    return fake_client


def test_generate_without_api_key_returns_friendly_message():
    generator = AnswerGenerator(api_key="", groq_api_key="")

    assert generator.is_configured() is False
    assert generator.generate("A question?", []) == MISSING_KEY_MESSAGE


# --- Groq: primary provider ---


def test_generate_returns_answer_text_on_success(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post",
        _capturing_groq_post(captured, content="  Employees get 24 days of annual leave.  "),
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    answer = generator.generate("What is the leave policy?", [])

    assert answer == "Employees get 24 days of annual leave."
    assert captured["headers"]["Authorization"] == "Bearer fake-or-key"


def test_generate_returns_fallback_message_on_api_error(monkeypatch):
    monkeypatch.setattr(
        "app.rag.generator.requests.post", MagicMock(side_effect=RuntimeError("API down"))
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    answer = generator.generate("A question?", [])

    assert answer == FALLBACK_MESSAGE


def test_generate_returns_fallback_message_on_empty_response(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post",
        _capturing_groq_post(captured, content=""),
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    answer = generator.generate("A question?", [])

    assert answer == FALLBACK_MESSAGE


def test_generate_uses_configured_groq_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(
        groq_api_key="fake-or-key", groq_model="meta-llama/llama-3.1-8b-instruct:free", api_key=""
    )
    generator.generate("A question?", [])

    assert captured["json"]["model"] == "meta-llama/llama-3.1-8b-instruct:free"


def test_generate_groq_model_name_overrides_configured_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(
        groq_api_key="fake-or-key", groq_model="groq/auto", api_key=""
    )
    generator.generate("A question?", [], groq_model_name="mistralai/mistral-7b-instruct:free")

    assert captured["json"]["model"] == "mistralai/mistral-7b-instruct:free"
    assert generator.groq_model == "groq/auto"  # unchanged for next calls


def test_prompt_is_built_from_question_and_chunks(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    generator.generate("What is the leave policy?", [])

    # Last message is the current RAG-augmented prompt.
    last_turn = captured["json"]["messages"][-1]
    assert last_turn["role"] == "user"
    assert "What is the leave policy?" in last_turn["content"]


def test_generate_without_history_sends_only_current_turn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    generator.generate("A question?", [])

    assert len(captured["json"]["messages"]) == 1


def test_generate_with_history_includes_prior_turns_in_order(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    history = [
        {"role": "user", "content": "My name is Mubarak."},
        {"role": "assistant", "content": "Nice to meet you, Mubarak!"},
    ]

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    answer = generator.generate("What is my name?", [], history=history)

    assert answer == "answer"
    messages = captured["json"]["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "My name is Mubarak."}
    assert messages[1] == {"role": "assistant", "content": "Nice to meet you, Mubarak!"}
    assert messages[2]["role"] == "user"
    assert "What is my name?" in messages[2]["content"]


def test_generate_caps_history_to_recent_messages(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(30)
    ]

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    generator.generate("Final question?", [], history=history)

    # 10 history messages capped + 1 current turn.
    messages = captured["json"]["messages"]
    assert len(messages) == 11
    assert messages[0]["content"] == "message 20"


def test_generate_uses_configured_default_temperature(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    generator.generate("A question?", [])

    assert captured["json"]["temperature"] == generator.temperature


def test_generate_uses_explicit_temperature_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="", temperature=0.9)
    generator.generate("A question?", [])

    assert generator.temperature == 0.9
    assert captured["json"]["temperature"] == 0.9


def test_generate_uses_grounded_prompt_when_chunks_present(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    generator.generate("What is the leave policy?", [_chunk()])

    prompt = captured["json"]["messages"][-1]["content"]
    assert "SOURCE INFORMATION" in prompt
    assert "handbook.pdf" in prompt


def test_generate_uses_general_knowledge_prompt_when_no_chunks(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.requests.post", _capturing_groq_post(captured)
    )

    generator = AnswerGenerator(groq_api_key="fake-or-key", api_key="")
    generator.generate("What is the capital of France?", [])

    prompt = captured["json"]["messages"][-1]["content"].lower()
    assert "general knowledge" in prompt
    assert "source information" not in prompt


# --- Gemini: fallback provider ---


def test_generate_falls_back_to_gemini_when_groq_unconfigured(monkeypatch):
    fake_response = MagicMock()
    fake_response.text = "Answer from Gemini."
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: fake_client
    )

    generator = AnswerGenerator(api_key="fake-key", groq_api_key="")

    assert generator.is_configured() is True
    answer = generator.generate("A question?", [])

    assert answer == "Answer from Gemini."


def test_generate_falls_back_to_gemini_when_groq_call_fails(monkeypatch):
    monkeypatch.setattr(
        "app.rag.generator.requests.post", MagicMock(side_effect=RuntimeError("network down"))
    )

    fake_response = MagicMock()
    fake_response.text = "Answer from Gemini."
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: fake_client
    )

    generator = AnswerGenerator(api_key="fake-key", groq_api_key="fake-or-key")
    answer = generator.generate("A question?", [])

    assert answer == "Answer from Gemini."


def test_generate_prefers_gemini_when_gemini_model_picked(monkeypatch):
    post = MagicMock(side_effect=AssertionError("Groq should not be called"))
    monkeypatch.setattr("app.rag.generator.requests.post", post)

    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: _capturing_gemini_client(captured)
    )

    generator = AnswerGenerator(api_key="fake-key", groq_api_key="fake-or-key")
    answer = generator.generate("A question?", [], model_name="gemini-2.5-flash")

    assert answer == "answer"
    assert captured["model"] == "gemini-2.5-flash"
    post.assert_not_called()


def test_generate_falls_back_to_groq_when_picked_gemini_fails(monkeypatch):
    monkeypatch.setattr(
        "app.rag.generator.requests.post",
        _capturing_groq_post({}, content="Answer from Groq."),
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("API down")
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: fake_client
    )

    generator = AnswerGenerator(api_key="fake-key", groq_api_key="fake-or-key")
    answer = generator.generate("A question?", [], model_name="gemini-2.5-flash")

    assert answer == "Answer from Groq."


def test_generate_returns_fallback_message_when_both_providers_fail(monkeypatch):
    monkeypatch.setattr(
        "app.rag.generator.requests.post", MagicMock(side_effect=RuntimeError("network down"))
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("API down")
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: fake_client
    )

    generator = AnswerGenerator(api_key="fake-key", groq_api_key="fake-or-key")
    answer = generator.generate("A question?", [])

    assert answer == FALLBACK_MESSAGE


def test_gemini_fallback_maps_assistant_role_and_builds_contents(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client",
        lambda api_key: _capturing_gemini_client(captured),
    )

    history = [
        {"role": "user", "content": "My name is Mubarak."},
        {"role": "assistant", "content": "Nice to meet you, Mubarak!"},
    ]

    generator = AnswerGenerator(api_key="fake-key", groq_api_key="")
    answer = generator.generate("What is my name?", [], history=history)

    assert answer == "answer"
    contents = captured["contents"]
    assert len(contents) == 3
    assert contents[0] == {"role": "user", "parts": [{"text": "My name is Mubarak."}]}
    # Assistant turns map to Gemini's "model" role.
    assert contents[1] == {"role": "model", "parts": [{"text": "Nice to meet you, Mubarak!"}]}
    assert contents[2]["role"] == "user"
    assert "What is my name?" in contents[2]["parts"][0]["text"]


class _Resp:
    def __init__(self, status, body=None, text="", headers=None):
        self.status_code, self._body, self.text, self.headers = status, body or {}, text, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_chat_with_tools_retries_once_on_short_rate_limit(monkeypatch):
    replies = [_Resp(429, text="Rate limit reached ... Please try again in 1.5s."),
               _Resp(200, {"choices": [{"message": {"content": "ok"}}]})]
    sent = []
    monkeypatch.setattr("app.rag.generator.requests.post", lambda url, **kw: (sent.append(kw["json"]), replies.pop(0))[1])
    slept = []
    monkeypatch.setattr("app.rag.generator.time.sleep", slept.append)

    message = AnswerGenerator(api_key="", groq_api_key="k").chat_with_tools([{"role": "user", "content": "hi"}], [])

    assert message == {"content": "ok"}
    assert slept == [2.0]
    assert "tools" not in sent[0] and "tool_choice" not in sent[0]  # no tools -> plain turn


def test_chat_with_tools_raises_when_rate_limit_wait_is_too_long(monkeypatch):
    from app.rag.generator import ToolCallingError

    monkeypatch.setattr("app.rag.generator.requests.post",
                        lambda url, **kw: _Resp(429, text="Please try again in 45s."))
    with pytest.raises(ToolCallingError):
        AnswerGenerator(api_key="", groq_api_key="k").chat_with_tools([{"role": "user", "content": "hi"}], [])

