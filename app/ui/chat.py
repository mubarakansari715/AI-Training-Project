"""Chat UI.

Retrieves relevant chunks through the RAG pipeline and sends that
context (plus recent conversation history, for continuity) to the
LLM. When nothing relevant is found, the answer falls back to the
model's general knowledge instead of refusing — either way, the
source is labeled under the answer. Re-asking a question already
answered earlier in this conversation returns that same stored answer
instead of a fresh (and possibly different) LLM call. Documents are
added through the Attach button above the chat box, which opens a dialog
showing each file's indexing progress. Each exchange is persisted via
ConversationStore so it survives an app restart and shows up in the
sidebar's conversation list.
"""

from __future__ import annotations

import logging
import re

import streamlit as st

from app.config.settings import APP_ICON
from app.rag.document_loader import DocumentParsingError
from app.rag.media_transcriber import transcribe_speech
from app.rag.retriever import RetrievedChunk
from app.services.conversation_service import ConversationStore
from app.services.rag_service import (
    SOURCE_DOCUMENT,
    SOURCE_GENERAL_KNOWLEDGE,
    RAGPipeline,
)
from app.ui.components import SUPPORTED_DOCUMENT_TYPES
from app.ui.sidebar import render_model_picker

logger = logging.getLogger(__name__)

# Neutral icon so the user avatar follows the theme instead of
# Streamlit's default red.
USER_AVATAR = ":material/person:"

# Small-talk answers aren't labeled — only these two are worth calling out.
_SOURCE_LABELS = {
    SOURCE_DOCUMENT: "📄 Source: your uploaded documents",
    SOURCE_GENERAL_KNOWLEDGE: "🌐 Source: general AI knowledge (not from your documents)",
}


def _format_location(source: str, page: int | None) -> str:
    return f"{source}, page {page}" if page is not None else source




def _render_source_label(source: str | None) -> None:
    label = _SOURCE_LABELS.get(source or "")
    if label:
        st.caption(label)


def _render_agent_trace(searches: list[str], model: str | None, chart_pages: list[str] | None = None) -> None:
    """Auto mode transparency: what the agent searched for, which chart
    pages the answer was read from, and which model answered — like
    ChatGPT's "Searched …" line."""
    if searches:
        st.caption("🔎 Searched: " + " · ".join(f"“{q}”" for q in searches))
    if chart_pages:
        st.caption("📊 Chart read: " + " · ".join(chart_pages))
    if model:
        st.caption(f"⚡ {model.split('/')[-1]}")


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _md(text: str) -> str:
    """Message text as Markdown: a "$" (as in "$375.7 million") would otherwise
    start a math formula and garble the line, or a table of amounts."""
    return text.replace("$", "\\$")


def _find_cached_answer(chat_history: list[dict], question: str) -> dict | None:
    """Return a previous answer to this same question in this conversation,
    if any, so asking it again gives back the same answer instead of a
    fresh (and possibly different) LLM call.

    `chat_history` must already include the current question as its
    last entry (with no reply yet) — the search starts just before it,
    looking for the most recent earlier user turn with the same
    (normalized) text, and returns its immediate assistant reply.
    """
    target = _normalize_question(question)
    if not target:
        return None
    for i in range(len(chat_history) - 2, -1, -1):
        message = chat_history[i]
        if message["role"] != "user" or _normalize_question(message["content"]) != target:
            continue
        reply = chat_history[i + 1]
        if reply["role"] == "assistant":
            return reply
    return None


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


def _transcribe_voice(audio) -> str | None:
    """Turn a recorded voice message into text; None (after telling the
    user why) if it failed or no speech was heard."""
    with st.spinner("Transcribing your voice message..."):
        try:
            spoken = transcribe_speech(audio.getvalue())
        except DocumentParsingError:
            st.error("Sorry, I couldn't process that recording. Please try again.")
            return None
    if not spoken:
        st.warning("I couldn't hear any speech in that recording. Please try again.")
        return None
    return spoken


def _index_with_progress(pipeline: RAGPipeline, files: list) -> list[dict]:
    """Index each file in turn, with a live progress bar and stage label;
    return one result per file: {"name", "error", "stats", "added"}."""
    results = []
    for file in files:
        with st.container(border=True):
            st.markdown(f"**{file.name}**")
            bar = st.progress(0.0, text="Starting...")
            try:
                added, errors = pipeline.index_files(
                    [file], progress=lambda label, fraction: bar.progress(fraction, text=label))
            except Exception:
                logger.exception("Indexing %s failed", file.name)
                added, errors = 0, {"_index": "Something went wrong while processing this document."}
        error = errors.get(file.name) or errors.get("_index")
        results.append({"name": file.name, "error": error, "stats": pipeline.last_indexed.get(file.name),
                        "added": added})
    return results


def _ready_note(result: dict) -> str:
    """ "200 pages, 1,194 passages, 101 charts measured" for a finished upload."""
    stats = result["stats"] or {}
    parts = []
    if stats.get("pages"):
        parts.append(f"{stats['pages']:,} page{'s' if stats['pages'] != 1 else ''}")
    if stats.get("passages"):
        parts.append(f"{stats['passages']:,} passages")
    if stats.get("charts"):
        parts.append(f"{stats['charts']:,} chart{'s' if stats['charts'] != 1 else ''} measured")
    note = ", ".join(parts)
    if not result["added"]:
        note += " (already indexed, unchanged)" if note else "already indexed, unchanged"
    return note


@st.dialog("Add documents", width="medium", dismissible=False)
def _attach_dialog(pipeline: RAGPipeline, conversation_store: ConversationStore, conversation_id: str) -> None:
    """Pick files, watch them being indexed, then go back to the chat.

    Not dismissible, so a long upload can't be cut off by a stray click:
    it closes through Cancel or Start asking (a full rerun)."""
    results = st.session_state.get("attach_results")
    if results is None:
        picker = st.empty()  # cleared once the upload starts, leaving only its progress
        with picker.container():
            files = st.file_uploader(
                "Choose documents",
                type=SUPPORTED_DOCUMENT_TYPES,
                accept_multiple_files=True,
                key=f"attach_uploader_{st.session_state.get('attach_round', 0)}",
                help="PDF, Word, text, Markdown, video or audio. A long PDF takes a few minutes.",
            )
            with st.container(horizontal=True, horizontal_alignment="right"):
                cancel = st.button("Cancel", key="attach_cancel")
                upload = st.button("Upload", type="primary", icon=":material/upload:", disabled=not files,
                                   key="attach_upload")
        if cancel:
            st.rerun()
        if upload:
            picker.empty()
            st.session_state.attach_results = _index_with_progress(pipeline, files)
            st.rerun(scope="fragment")
        return

    for result in results:
        if result["error"]:
            st.error(f"**{result['name']}**: {result['error']}", icon=":material/error:")
        else:
            st.success(f"**{result['name']}** is ready: {_ready_note(result)}", icon=":material/check_circle:")
    ready = [r for r in results if not r["error"]]
    with st.container(horizontal=True, horizontal_alignment="right"):
        more = st.button("Add more", icon=":material/add:", key="attach_more")
        done = st.button("Start asking" if ready else "Close", type="primary", key="attach_done")
    if more:
        st.session_state.attach_results = None
        st.session_state.attach_round = st.session_state.get("attach_round", 0) + 1
        st.rerun(scope="fragment")
    if done:
        if ready:
            _announce_documents(ready, conversation_store, conversation_id)
        st.session_state.attach_results = None
        st.rerun()


def _announce_documents(ready: list[dict], conversation_store: ConversationStore, conversation_id: str) -> None:
    """Note the new documents in the chat and the sidebar list."""
    names = st.session_state.setdefault("document_names", [])
    lines = []
    for result in ready:
        if result["name"] not in names:
            names.append(result["name"])
        lines.append(f"📄 **{result['name']}** is ready: {_ready_note(result)}.")
    lines.append("Ask me anything about " + ("it." if len(ready) == 1 else "them."))
    st.session_state.chat_history.append({"role": "assistant", "content": "\n\n".join(lines), "sources": []})
    conversation_store.save(conversation_id, st.session_state.chat_history)


def render_chat(
    pipeline: RAGPipeline,
    conversation_store: ConversationStore,
    conversation_id: str,
    show_debug: bool,
    strict_mode: bool,
    temperature: float | None = None,
    auto_mode: bool | None = None,
) -> None:
    """Render chat history and input box, storing messages in session state
    and persisting each exchange to `conversation_id` via `conversation_store`.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        avatar = APP_ICON if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            for filename in message.get("attached_files", []):
                st.caption(f"📎 {filename}")
            if message.get("voice"):
                st.caption("🎤 Voice message")
            if message["content"]:
                st.markdown(_md(message["content"]))
            if message["role"] == "assistant":
                _render_source_label(message.get("source"))
                _render_agent_trace(message.get("searches", []), message.get("model"), message.get("chart_pages"))
                _render_sources(message.get("sources", []))
                if message.get("show_debug"):
                    _render_debug(message.get("sources", []))

    with st.bottom:
        with st.container(horizontal=True, vertical_alignment="center"):
            attach = st.button("Attach", icon=":material/attach_file:", key="attach_open",
                               help="Add documents to ask about")
            st.space("stretch")
            model_name, groq_model_name = render_model_picker()

        submission = st.chat_input(
            "Ask a question about your documents, or record your voice...",
            accept_audio=True,
            submit_mode="disable",
        )
    if attach:
        st.session_state.attach_results = None  # a fresh dialog each time
        _attach_dialog(pipeline, conversation_store, conversation_id)
    if not submission:
        return

    question = (submission.text or "").strip()
    audio = submission.audio

    if audio is not None:
        spoken = _transcribe_voice(audio)
        if spoken is None:
            return
        question = f"{question} {spoken}".strip()

    if not question:
        return

    # Snapshot prior turns (skipping empty-text ones) before appending the
    # new message, so the generator gets history that doesn't duplicate the
    # current turn.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history
        if m["content"]
    ]

    with st.chat_message("user", avatar=USER_AVATAR):
        if audio is not None:
            st.caption("🎤 Voice message")
        st.write(question)
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
            "voice": audio is not None,
        }
    )

    with st.chat_message("assistant", avatar=APP_ICON):
        sources = []
        source_label = None
        searches, answer_model, chart_pages = [], None, []

        cached = _find_cached_answer(st.session_state.chat_history, question)
        if cached is not None:
            answer, sources, source_label = (
                cached["content"],
                cached.get("sources", []),
                cached.get("source"),
            )
            searches, answer_model = cached.get("searches", []), cached.get("model")
            chart_pages = cached.get("chart_pages", [])
        else:
            with st.spinner("Thinking..."):
                try:
                    result = pipeline.ask(
                        question,
                        strict_mode=strict_mode,
                        history=history,
                        model_name=model_name,
                        groq_model_name=groq_model_name,
                        temperature=temperature,
                        auto_mode=auto_mode,
                    )
                    answer, sources, source_label = result.answer, result.sources, result.source
                    # Agent searches, or the sub-questions of a decomposed question.
                    searches, answer_model = result.searches or result.sub_queries, result.model
                    chart_pages = result.chart_pages
                except Exception:
                    answer, sources, source_label = (
                        "Something went wrong while generating the answer. Please try again.",
                        [],
                        None,
                    )

        st.markdown(_md(answer))
        _render_source_label(source_label)
        _render_agent_trace(searches, answer_model, chart_pages)
        _render_sources(sources)
        if show_debug:
            _render_debug(sources)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "source": source_label,
            "show_debug": show_debug,
            "searches": searches,
            "model": answer_model,
            "chart_pages": chart_pages,
        }
    )
    conversation_store.save(conversation_id, st.session_state.chat_history)
    st.rerun()
