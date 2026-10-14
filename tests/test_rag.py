"""Integration tests for the complete RAG pipeline.

The embedding model and vector store are real (fast/local, no network
needed after the model is cached). Only the Gemini API call is
mocked, so these tests never require a real API key.
"""

from unittest.mock import MagicMock

import pytest

from app.rag.generator import AnswerGenerator
from app.services.rag_service import (
    NOT_FOUND_MESSAGE,
    SOURCE_DOCUMENT,
    SOURCE_GENERAL_KNOWLEDGE,
    SOURCE_SMALL_TALK,
    RAGPipeline,
)


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


@pytest.fixture
def pipeline(tmp_path):
    pipeline = RAGPipeline(
        persist_dir=str(tmp_path / "chroma"), collection_name="rag_pipeline_test"
    )
    # The mocked LLM returns the same text for every call, so it would also
    # be used as the "rewritten" query; rewriting has its own tests below.
    pipeline.rewrite_queries = False
    return pipeline


def _mock_gemini(pipeline, monkeypatch, answer_text: str):
    fake_response = MagicMock()
    fake_response.text = answer_text
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr("app.rag.generator.genai.Client", lambda api_key: fake_client)
    # The pipeline's generator was already constructed (with no API key,
    # since tests run without GEMINI_API_KEY set) before this patch — swap
    # in a fresh one now that genai.Client is mocked. Groq is
    # explicitly disabled so the (primary) fallback path doesn't shadow
    # the Gemini call these tests are exercising.
    pipeline.generator = AnswerGenerator(api_key="fake-key", groq_api_key="")
    return fake_client


def test_index_files_adds_chunks(pipeline):
    file = FakeUploadedFile(
        "handbook.txt", b"Employees are entitled to 24 days of annual leave per year."
    )

    num_chunks, errors = pipeline.index_files([file])

    assert num_chunks >= 1
    assert errors == {}
    assert pipeline.vector_store.count() >= 1
    assert pipeline.document_count() == 1


def test_ask_without_any_documents_falls_back_to_general_knowledge_by_default(pipeline, monkeypatch):
    # Non-strict is the default: no documents at all shouldn't refuse
    # outright, it should answer from the model's general knowledge.
    _mock_gemini(pipeline, monkeypatch, "Paris is the capital of France.")

    result = pipeline.ask("What is the capital of France?")

    assert result.answer == "Paris is the capital of France."
    assert result.sources == []
    assert result.found_context is False
    assert result.source == SOURCE_GENERAL_KNOWLEDGE


def test_greeting_bypasses_not_found_even_with_no_documents(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "Hello! How can I help you with your documents?")

    result = pipeline.ask("hi", strict_mode=True)

    assert result.answer == "Hello! How can I help you with your documents?"
    assert result.source == SOURCE_SMALL_TALK


def test_greeting_bypasses_strict_mode_even_with_irrelevant_documents(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "Hey there!")

    file = FakeUploadedFile("handbook.txt", b"Completely unrelated office snack policy.")
    pipeline.index_files([file])
    pipeline.retriever.score_threshold = 0.99  # force "no relevant chunks"

    result = pipeline.ask("hello", strict_mode=True)

    assert result.answer == "Hey there!"


def test_real_question_still_blocked_under_strict_mode_with_no_documents(pipeline):
    # Sanity check: only small talk gets the bypass, not real questions.
    result = pipeline.ask("What is the annual leave policy?", strict_mode=True)

    assert result.answer == NOT_FOUND_MESSAGE


def test_ask_with_empty_question_asks_for_input(pipeline):
    file = FakeUploadedFile("handbook.txt", b"Some content about leave policy.")
    pipeline.index_files([file])

    result = pipeline.ask("   ")

    assert "enter a question" in result.answer.lower()


def test_full_pipeline_returns_answer_with_sources(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "Employees get 24 days of annual leave per year.")

    file = FakeUploadedFile(
        "handbook.txt",
        b"Employees are entitled to 24 days of annual leave per year. "
        b"Sick leave is handled separately under a different policy.",
    )
    pipeline.index_files([file])

    result = pipeline.ask("What is the annual leave policy?")

    assert result.answer == "Employees get 24 days of annual leave per year."
    assert result.found_context is True
    assert len(result.sources) > 0
    assert result.sources[0].source == "handbook.txt"
    assert result.source == SOURCE_DOCUMENT


def test_strict_mode_blocks_llm_call_when_no_relevant_context(pipeline, monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr("app.rag.generator.genai.Client", lambda api_key: fake_client)

    file = FakeUploadedFile("handbook.txt", b"Completely unrelated office snack policy.")
    pipeline.index_files([file])
    pipeline.retriever.score_threshold = 0.99  # force "no relevant chunks"

    result = pipeline.ask("What is the annual leave policy?", strict_mode=True)

    assert result.answer == NOT_FOUND_MESSAGE
    fake_client.models.generate_content.assert_not_called()


def test_non_strict_mode_calls_llm_even_without_relevant_context(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "I don't have specific info, but generally...")

    file = FakeUploadedFile("handbook.txt", b"Completely unrelated office snack policy.")
    pipeline.index_files([file])
    pipeline.retriever.score_threshold = 0.99

    result = pipeline.ask("What is the annual leave policy?", strict_mode=False)

    assert result.answer == "I don't have specific info, but generally..."
    assert result.source == SOURCE_GENERAL_KNOWLEDGE


def test_clear_index_removes_all_documents(pipeline):
    file = FakeUploadedFile("handbook.txt", b"Some content about leave policy.")
    pipeline.index_files([file])
    assert pipeline.vector_store.count() > 0

    pipeline.clear_index()

    assert pipeline.vector_store.count() == 0
    assert pipeline.document_count() == 0


def test_reuploading_the_same_document_does_not_duplicate_chunks(pipeline):
    file = FakeUploadedFile("handbook.txt", b"Employees get 24 days of annual leave.")

    first_count, _ = pipeline.index_files([file])
    second_count, _ = pipeline.index_files([file])

    assert first_count > 0
    assert second_count == 0
    assert pipeline.vector_store.count() == first_count


def test_small_talk_does_not_attach_document_sources(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "I am ALLY AI.")

    file = FakeUploadedFile("handbook.txt", b"Who are you? Employees get 24 days of leave.")
    pipeline.index_files([file])

    result = pipeline.ask("who are you?")

    assert result.answer == "I am ALLY AI."
    assert result.sources == []
    assert result.source == SOURCE_SMALL_TALK


def test_retrieval_uses_rewritten_query(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "You get 24 days.")
    pipeline.index_files([FakeUploadedFile("handbook.txt", b"Employees are entitled to 24 days of annual leave per year.")])
    pipeline.rewrite_queries = True
    calls = []

    def fake_rewrite(generator, question, history, **kwargs):
        calls.append((question, history))
        return "How many days of annual leave are employees entitled to?"

    monkeypatch.setattr("app.services.rag_service.rewrite_query", fake_rewrite)
    history = [{"role": "user", "content": "hi"}]

    result = pipeline.ask("hw mny leev dayz?", strict_mode=True, history=history)

    assert calls == [("hw mny leev dayz?", history)]
    assert result.search_query == "How many days of annual leave are employees entitled to?"
    assert result.source == SOURCE_DOCUMENT
    assert result.sources


def test_rewrite_is_skipped_when_no_documents_are_indexed(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "Paris.")
    pipeline.rewrite_queries = True
    monkeypatch.setattr(
        "app.services.rag_service.rewrite_query",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not rewrite")),
    )

    result = pipeline.ask("What is the capital of France?")

    assert result.search_query == "What is the capital of France?"


def test_typo_question_recognised_as_small_talk_after_rewrite(pipeline, monkeypatch):
    _mock_gemini(pipeline, monkeypatch, "I was built by the team.")
    pipeline.index_files([FakeUploadedFile("handbook.txt", b"Who built the office? Contractors.")])
    pipeline.rewrite_queries = True
    monkeypatch.setattr("app.services.rag_service.rewrite_query", lambda *a, **k: "Who built you?")

    result = pipeline.ask("whu bilt yu", strict_mode=True)

    assert result.source == SOURCE_SMALL_TALK
    assert result.sources == []



def test_pipeline_settings_can_be_overridden_per_instance(tmp_path):
    pipeline = RAGPipeline(
        persist_dir=str(tmp_path / "chroma"), collection_name="override_test",
        chunk_size=200, chunk_overlap=20, top_k=2, score_threshold=0.5,
    )
    long_text = ("Employees get 24 days of annual leave. " * 30).encode()

    num_chunks, _ = pipeline.index_files([FakeUploadedFile("handbook.txt", long_text)])

    assert num_chunks > 1  # 200-char chunks split this text, the 800 default wouldn't twice over
    assert pipeline.retriever.top_k == 2
    assert pipeline.retriever.score_threshold == 0.5


def _agent_pipeline(pipeline, monkeypatch, run_agent_impl):
    pipeline.index_files([FakeUploadedFile("handbook.txt", b"Employees are entitled to 24 days of annual leave per year.")])
    pipeline.generator.groq_api_key = "fake-groq"
    monkeypatch.setattr("app.services.rag_service.run_agent", run_agent_impl)
    return pipeline


def test_auto_mode_uses_the_agent_and_reports_its_searches(pipeline, monkeypatch):
    from app.rag.agent import AgentResult

    captured = {}

    def fake_agent(question, generator, retriever, history, **kwargs):
        captured.update(kwargs)
        chunks = retriever.retrieve("annual leave days")
        return AgentResult(answer="24 days (handbook.txt).", sources=chunks, searches=["annual leave days"])

    _agent_pipeline(pipeline, monkeypatch, fake_agent)
    result = pipeline.ask("How many leave days?", auto_mode=True, groq_model_name="auto")

    assert result.answer == "24 days (handbook.txt)."
    assert result.searches == ["annual leave days"]
    assert result.source == SOURCE_DOCUMENT
    assert result.model == captured["model_name"]  # "auto" resolved to a concrete model
    assert result.model != "auto"


def test_auto_mode_falls_back_to_standard_pipeline_on_agent_error(pipeline, monkeypatch):
    from app.rag.agent import AgentError

    def failing_agent(*args, **kwargs):
        raise AgentError("rate limited")

    _agent_pipeline(pipeline, monkeypatch, failing_agent)
    pipeline.rewrite_queries = False
    _mock_gemini(pipeline, monkeypatch, "You get 24 days.")
    pipeline.generator.groq_api_key = "fake-groq"
    monkeypatch.setattr(pipeline.generator, "_generate_with_groq", lambda *a, **k: "")  # Groq down -> Gemini

    result = pipeline.ask("How many leave days?", auto_mode=True)

    assert result.answer == "You get 24 days."
    assert result.searches == []
    assert result.source == SOURCE_DOCUMENT


def test_auto_mode_strict_guard_refuses_when_agent_found_nothing(pipeline, monkeypatch):
    from app.rag.agent import AgentResult

    _agent_pipeline(pipeline, monkeypatch, lambda *a, **k: AgentResult(answer="Probably 30 days.", sources=[], searches=["x"]))

    result = pipeline.ask("How many sick days?", auto_mode=True, strict_mode=True)

    assert result.answer == NOT_FOUND_MESSAGE


def test_auto_mode_skipped_when_gemini_model_picked(pipeline, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("agent should not run")

    _agent_pipeline(pipeline, monkeypatch, must_not_run)
    pipeline.rewrite_queries = False
    _mock_gemini(pipeline, monkeypatch, "You get 24 days.")
    pipeline.generator.groq_api_key = "fake-groq"

    result = pipeline.ask("How many leave days?", auto_mode=True, model_name="gemini-3.5-flash-lite")

    assert result.answer == "You get 24 days."
    assert result.model is None


def test_not_found_answer_is_not_labelled_as_from_documents(pipeline, monkeypatch):
    from app.rag.chunker import Chunk

    _mock_gemini(pipeline, monkeypatch, "The FY 2025 budget was not found in the uploaded documents.")
    chunk = Chunk(text="Appropriations for FY 2018 were $117 million.", source="report.pdf", page=12, chunk_id="c12")
    pipeline.vector_store.add_chunks([chunk], pipeline.embedding_service.embed_texts([chunk.text]))
    pipeline.retriever.score_threshold = -1  # make sure the unrelated chunk is retrieved

    result = pipeline.ask("What was the budget for FY 2025?", strict_mode=True)

    assert result.sources == []
    assert result.source is None


def test_section_aware_toggle_disables_labels_and_filtering(monkeypatch):
    """Fast check of the SECTION_AWARE switch (no real model or PDF needed)."""
    from app.rag import document_loader
    from app.rag.retriever import Retriever

    monkeypatch.setattr(document_loader, "detect_sections", lambda texts, outline: dict.fromkeys(texts, "Executive Summary"))
    data = _minimal_pdf("Global growth slows.")
    assert document_loader.parse_document("r.pdf", data)[0].section == "Executive Summary"
    assert document_loader.parse_document("r.pdf", data, detect_structure=False)[0].section is None

    class Store:
        def count(self):
            return 1

        def sections(self):
            return ["Executive Summary"]

    off = Retriever(Store(), None, section_aware=False)
    on = Retriever(Store(), None, section_aware=True)
    assert off._section_filter("According to the executive summary?") == ([], None)
    assert on._section_filter("According to the executive summary?")[0] == [["Executive Summary"]]


def _minimal_pdf(text: str) -> bytes:
    from tests.test_document_loader import _make_minimal_pdf

    return _make_minimal_pdf(text)

