from app.rag.prompt import (
    build_context,
    build_general_knowledge_prompt,
    build_prompt,
    build_source_list,
)
from app.rag.retriever import RetrievedChunk


def _chunk(text="Employees get 24 days of leave.", source="handbook.pdf", page=12, score=0.9):
    return RetrievedChunk(text=text, source=source, page=page, score=score, chunk_id="c1")


def test_build_context_includes_chunk_text_and_source():
    context = build_context([_chunk()])

    assert "Employees get 24 days of leave." in context
    assert "handbook.pdf" in context
    assert "page 12" in context


def test_build_context_with_no_chunks_says_so():
    context = build_context([])

    assert "no relevant context" in context.lower()


def test_build_source_list_omits_page_when_none():
    chunk = _chunk(source="notes.txt", page=None)

    source_list = build_source_list([chunk])

    assert "notes.txt" in source_list
    assert "page" not in source_list


def test_build_prompt_has_all_required_sections():
    prompt = build_prompt("What is the leave policy?", [_chunk()])

    assert "SYSTEM INSTRUCTIONS" in prompt
    assert "CONTEXT" in prompt
    assert "SOURCE INFORMATION" in prompt
    assert "USER QUESTION" in prompt
    assert "What is the leave policy?" in prompt


def test_build_prompt_instructs_no_outside_knowledge():
    # Normalize whitespace so line-wrapping in the instructions text
    # doesn't break a substring check across a newline.
    prompt = " ".join(build_prompt("Anything?", []).lower().split())

    assert "only the provided context" in prompt
    assert "not found in the uploaded documents" in prompt


def test_build_prompt_allows_conversational_continuity():
    prompt = build_prompt("Anything?", []).lower()

    assert "conversation" in prompt


def test_build_general_knowledge_prompt_has_required_sections():
    prompt = build_general_knowledge_prompt("What is the capital of France?")

    assert "SYSTEM INSTRUCTIONS" in prompt
    assert "USER QUESTION" in prompt
    assert "What is the capital of France?" in prompt


def test_build_general_knowledge_prompt_allows_outside_knowledge():
    prompt = build_general_knowledge_prompt("Anything?").lower()

    assert "general knowledge" in prompt
    assert "documents" in prompt  # tells the model to disclose it isn't one
