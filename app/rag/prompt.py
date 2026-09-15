"""RAG prompt template.

Centralizes the instructions given to the LLM in one place, instead
of hardcoding prompt text throughout the app, so the "don't
hallucinate, cite sources" behavior is easy to find and tune.
"""

from __future__ import annotations

from app.rag.retriever import RetrievedChunk

SYSTEM_INSTRUCTIONS = """You are a document question-answering assistant.

For any question about the uploaded documents, answer using only the
provided context below. Do not use outside knowledge and do not
invent facts about the documents. If the answer isn't in the provided
context, clearly say that the information was not found in the
uploaded documents. When possible, mention which source document (and
page, if given) the answer came from.

You may also use the earlier turns of this conversation for
continuity (e.g. remembering the user's name or referring back to
something they said) — that rule only restricts document facts, not
the conversation itself."""


def _format_location(chunk: RetrievedChunk) -> str:
    return chunk.source + (f", page {chunk.page}" if chunk.page is not None else "")


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block with sources."""
    if not chunks:
        return "(no relevant context was found)"

    blocks = [
        f"[{i}] Source: {_format_location(chunk)}\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_source_list(chunks: list[RetrievedChunk]) -> str:
    """Format the distinct sources referenced by the retrieved chunks."""
    if not chunks:
        return "(none)"
    return "\n".join(f"- {_format_location(chunk)}" for chunk in chunks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the full document-grounded prompt sent to the LLM."""
    return (
        f"SYSTEM INSTRUCTIONS\n{SYSTEM_INSTRUCTIONS}\n\n"
        f"CONTEXT\n{build_context(chunks)}\n\n"
        f"SOURCE INFORMATION\n{build_source_list(chunks)}\n\n"
        f"USER QUESTION\n{question}"
    )


GENERAL_KNOWLEDGE_INSTRUCTIONS = """You are a helpful assistant. No relevant
content was found in the user's uploaded documents (or none are attached
yet), so answer the question using your own general knowledge instead.

Do not claim the answer came from the user's documents — it didn't. You
may also use the earlier turns of this conversation for continuity (e.g.
remembering the user's name or referring back to something they said)."""


def build_general_knowledge_prompt(question: str) -> str:
    """Assemble a prompt for answering from the model's own knowledge,
    used when no document context is available (see `build_prompt` for
    the document-grounded counterpart)."""
    return f"SYSTEM INSTRUCTIONS\n{GENERAL_KNOWLEDGE_INSTRUCTIONS}\n\nUSER QUESTION\n{question}"
