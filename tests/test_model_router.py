import pytest

from app.config.settings import settings
from app.rag.model_router import (
    AUTO_MODEL,
    choose_groq_model,
    is_complex,
    resolve_groq_model,
)


@pytest.mark.parametrize(
    "question",
    [
        "Compare passports produced and staff count in FY 2018",
        "What is the difference between FDsys and govinfo?",
        "Why did revenue increase?",
        "Explain the lipid results",
        "Summarize this report",
        "What is the growth outlook for South Asia vs Sub-Saharan Africa?",
        "What was revenue? And how many staff?",
        " ".join(["word"] * 26),
    ],
)
def test_complex_questions(question):
    assert is_complex(question)
    assert choose_groq_model(question) == settings.groq_strong_model


@pytest.mark.parametrize("question", ["What is my vitamin D level?", "How many passports did GPO produce?", "hi"])
def test_simple_questions(question):
    assert not is_complex(question)
    assert choose_groq_model(question) == settings.groq_fast_model


def test_resolve_only_changes_auto():
    assert resolve_groq_model(AUTO_MODEL, "hi") == settings.groq_fast_model
    assert resolve_groq_model("qwen/qwen3.8-27b", "Compare a and b") == "qwen/qwen3.8-27b"
    assert resolve_groq_model(None, "hi") is None
