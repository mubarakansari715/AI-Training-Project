from unittest.mock import MagicMock

from app.rag.generator import MISSING_KEY_MESSAGE, FALLBACK_MESSAGE, AnswerGenerator


def test_generate_without_api_key_returns_friendly_message():
    generator = AnswerGenerator(api_key="")

    assert generator.is_configured() is False
    assert generator.generate("A question?", []) == MISSING_KEY_MESSAGE


def test_generate_returns_model_text_on_success(monkeypatch):
    fake_response = MagicMock()
    fake_response.text = "  Employees get 24 days of annual leave.  "

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: fake_client
    )

    generator = AnswerGenerator(api_key="fake-key")
    answer = generator.generate("What is the leave policy?", [])

    assert answer == "Employees get 24 days of annual leave."
    fake_client.models.generate_content.assert_called_once()


def test_generate_returns_fallback_message_on_api_error(monkeypatch):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("API down")
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: fake_client
    )

    generator = AnswerGenerator(api_key="fake-key")
    answer = generator.generate("A question?", [])

    assert answer == FALLBACK_MESSAGE


def test_generate_returns_fallback_message_on_empty_response(monkeypatch):
    fake_response = MagicMock()
    fake_response.text = ""

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: fake_client
    )

    generator = AnswerGenerator(api_key="fake-key")
    answer = generator.generate("A question?", [])

    assert answer == FALLBACK_MESSAGE


def _capturing_client(captured):
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


def test_prompt_is_built_from_question_and_chunks(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: _capturing_client(captured)
    )

    generator = AnswerGenerator(api_key="fake-key")
    generator.generate("What is the leave policy?", [])

    # Last content turn is the current RAG-augmented prompt.
    last_turn = captured["contents"][-1]
    assert last_turn["role"] == "user"
    assert "What is the leave policy?" in last_turn["parts"][0]["text"]


def test_generate_without_history_sends_only_current_turn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: _capturing_client(captured)
    )

    generator = AnswerGenerator(api_key="fake-key")
    generator.generate("A question?", [])

    assert len(captured["contents"]) == 1


def test_generate_with_history_includes_prior_turns_in_order(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: _capturing_client(captured)
    )

    history = [
        {"role": "user", "content": "My name is Mubarak."},
        {"role": "assistant", "content": "Nice to meet you, Mubarak!"},
    ]

    generator = AnswerGenerator(api_key="fake-key")
    answer = generator.generate("What is my name?", [], history=history)

    assert answer == "answer"
    contents = captured["contents"]
    assert len(contents) == 3
    assert contents[0] == {
        "role": "user",
        "parts": [{"text": "My name is Mubarak."}],
    }
    # Assistant turns map to Gemini's "model" role.
    assert contents[1] == {
        "role": "model",
        "parts": [{"text": "Nice to meet you, Mubarak!"}],
    }
    assert contents[2]["role"] == "user"
    assert "What is my name?" in contents[2]["parts"][0]["text"]


def test_generate_caps_history_to_recent_messages(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: _capturing_client(captured)
    )

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(30)
    ]

    generator = AnswerGenerator(api_key="fake-key")
    generator.generate("Final question?", [], history=history)

    # 10 history messages capped + 1 current turn.
    assert len(captured["contents"]) == 11
    assert captured["contents"][0]["parts"][0]["text"] == "message 20"


def test_generate_uses_configured_default_temperature(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: _capturing_client(captured)
    )

    generator = AnswerGenerator(api_key="fake-key")
    generator.generate("A question?", [])

    assert captured["config"].temperature == generator.temperature


def test_generate_uses_explicit_temperature_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.rag.generator.genai.Client", lambda api_key: _capturing_client(captured)
    )

    generator = AnswerGenerator(api_key="fake-key", temperature=0.9)
    generator.generate("A question?", [])

    assert generator.temperature == 0.9
    assert captured["config"].temperature == 0.9
