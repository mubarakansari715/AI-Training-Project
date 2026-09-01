"""Chat UI.

Real, document-grounded chat: retrieves relevant chunks through the
RAG pipeline, sends only that context (plus recent conversation
history, for continuity) to Gemini, and shows the answer with its
sources (and, optionally, the raw retrieved chunks for transparency).
Documents are uploaded via the chat input's attach icon (ChatGPT/
Gemini-style) rather than a separate sidebar widget. Each exchange is
persisted via ConversationStore so it survives an app restart and
shows up in the sidebar's conversation list.
"""

from __future__ import annotations

import streamlit as st

from app.rag.retriever import RetrievedChunk
from app.services.conversation_service import ConversationStore
from app.services.rag_service import RAGPipeline
from app.ui.components import SUPPORTED_DOCUMENT_TYPES


def _format_location(source: str, page: int | None) -> str:
    return f"{source}, page {page}" if page is not None else source


def _render_sources(sources: list[RetrievedChunk]) -> None:
    if not sources:
        return
    with st.expander("📚 Sources", expanded=False):
        seen = set()
        for chunk in sources:
            location = _format_location(chunk.source, chunk.page)
            if location in seen:
                continue
            seen.add(location)
            st.markdown(f"📄 {location}")


def _render_debug(sources: list[RetrievedChunk]) -> None:
    if not sources:
        return
    with st.expander("🔍 Retrieval details", expanded=False):
        for i, chunk in enumerate(sources, start=1):
            location = _format_location(chunk.source, chunk.page)
            st.markdown(f"**Chunk {i}** — {location} — score: {chunk.score:.2f}")
            st.caption(chunk.text)
            if i < len(sources):
                st.divider()


def _index_attached_files(pipeline: RAGPipeline, files: list) -> str | None:
    """Index newly attached files and return a status note, or None."""
    with st.spinner(f"Indexing {len(files)} document(s)..."):
        try:
            num_chunks, errors = pipeline.index_files(files)
        except Exception:
            num_chunks, errors = 0, {
                "_index": "Something went wrong while processing your documents."
            }

    if "document_names" not in st.session_state:
        st.session_state.document_names = []
    for file in files:
        if file.name not in st.session_state.document_names:
            st.session_state.document_names.append(file.name)

    parts = []
    if num_chunks:
        parts.append(f"📄 Indexed {num_chunks} chunk(s) from {len(files)} document(s).")
    for filename, message in errors.items():
        parts.append(f"⚠️ {filename}: {message}")
    return "\n\n".join(parts) if parts else None


def render_chat(
    pipeline: RAGPipeline,
    conversation_store: ConversationStore,
    conversation_id: str,
    show_debug: bool,
    strict_mode: bool,
) -> None:
    """Render chat history and input box, storing messages in session state
    and persisting each exchange to `conversation_id` via `conversation_store`.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        avatar = ":material/smart_toy:" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            for filename in message.get("attached_files", []):
                st.caption(f"📎 {filename}")
            if message["content"]:
                st.write(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))
                if message.get("show_debug"):
                    _render_debug(message.get("sources", []))

    submission = st.chat_input(
        "Ask a question or attach documents...",
        accept_file="multiple",
        file_type=SUPPORTED_DOCUMENT_TYPES,
        submit_mode="disable",
    )
    if not submission:
        return

    question = (submission.text or "").strip()
    files = submission.files or []
    if not question and not files:
        return

    # Snapshot prior turns (skipping empty-text ones, e.g. attachment-only
    # turns) before appending the new message, so the generator gets
    # history that doesn't duplicate the current turn.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history
        if m["content"]
    ]

    with st.chat_message("user"):
        for file in files:
            st.caption(f"📎 {file.name}")
        if question:
            st.write(question)
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
            "attached_files": [file.name for file in files],
        }
    )

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        index_note = _index_attached_files(pipeline, files) if files else None
        sources = []

        if question:
            with st.spinner("Thinking..."):
                try:
                    result = pipeline.ask(question, strict_mode=strict_mode, history=history)
                    answer, sources = result.answer, result.sources
                except Exception:
                    answer, sources = (
                        "Something went wrong while generating the answer. Please try again.",
                        [],
                    )
            display_answer = f"{index_note}\n\n{answer}" if index_note else answer
        else:
            display_answer = index_note or "No files were indexed."

        st.write(display_answer)
        _render_sources(sources)
        if show_debug:
            _render_debug(sources)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": display_answer,
            "sources": sources,
            "show_debug": show_debug,
        }
    )
    conversation_store.save(conversation_id, st.session_state.chat_history)
    st.rerun()
