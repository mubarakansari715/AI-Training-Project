import pytest

from app.rag.small_talk import is_small_talk


@pytest.mark.parametrize(
    "text",
    [
        "hi",
        "Hi!",
        "hello",
        "Hello there?",
        "hey",
        "good morning",
        "Good Evening!",
        "how are you",
        "how's it going?",
        "what's up",
        "thanks",
        "thank you",
        "ty",
        "bye",
        "goodbye",
        "see you",
        "ok",
        "okay",
        "cool",
        "  hi  ",
    ],
)
def test_recognizes_small_talk(text):
    assert is_small_talk(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "What is the annual leave policy?",
        "How many remote work days are allowed?",
        "hi, what is the leave policy?",
        "Tell me about the sick leave process",
        "hi there, can you summarize page 3",
        "",
        "   ",
    ],
)
def test_does_not_flag_real_questions_as_small_talk(text):
    assert is_small_talk(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Hi my name is ali",
        "hi my name is Ali",
        "My name is Ali",
        "my name is Ali Khan",
        "i'm Ali",
        "I am Ali",
        "hey, i am Sam",
    ],
)
def test_recognizes_self_introductions(text):
    assert is_small_talk(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "what is my name ?",
        "what is my name?",
        "what's my name",
        "What is my name?",
        "do you remember what I said earlier?",
        "what did I say about the budget",
        "what did you tell me earlier",
        "can you repeat that?",
        "repeat what i said",
        "what were we talking about?",
    ],
)
def test_recognizes_conversational_recall_questions(text):
    assert is_small_talk(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I am not sure about the vacation policy",
        "I am looking for information about sick leave",
        "I'm trying to find the remote work section",
        "call me later about the contract",
        "What is the capital of France?",
        "What did the report say about revenue?",
    ],
)
def test_does_not_over_match_real_sentences(text):
    assert is_small_talk(text) is False
