"""Chat UI.

Retrieves relevant chunks through the RAG pipeline and sends that
context (plus recent conversation history, for continuity) to the
LLM. When nothing relevant is found, the answer falls back to the
model's general knowledge instead of refusing — either way, the
source is labeled under the answer. Re-asking a question already
answered earlier in this conversation returns that same stored answer
instead of a fresh (and possibly different) LLM call. Documents are
added through the Attach button above the chat box; they are indexed in
the background (app.services.ingest_jobs), with their progress in the
sidebar, so the chat stays usable meanwhile. Each exchange is persisted via
ConversationStore so it survives an app restart and shows up in the
sidebar's conversation list.
"""

from __future__ import annotations

import logging
import re
import time

import streamlit as st

from app.config.settings import APP_ICON
from app.rag.document_loader import DocumentParsingError
from app.rag.media_transcriber import transcribe_speech
from app.rag.retriever import RetrievedChunk
from app.services.conversation_service import ConversationStore
from app.services.ingest_jobs import FAILED, READY, SEARCHABLE, IngestJob, IngestQueue
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

# Chat bubbles: the user's messages on the right in a tinted bubble, ALLY's
# answers on the left, both in a centred column of comfortable width. Styles
# only (st.html sends them to the event container, taking no space).
_CHAT_STYLE = """<style>
.stMainBlockContainer, [data-testid="stBottomBlockContainer"] { max-width: 920px; }
.stMainBlockContainer { padding-top: 4.75rem; padding-bottom: 2rem; }
[data-testid="stSidebarUserContent"] { padding-top: 0; }
[data-testid="stSidebarUserContent"] h3:first-of-type { padding-top: 0.25rem; }
[data-testid="stSidebar"] hr { margin: 0.75rem 0; }
[data-testid="stChatMessage"] { background: transparent; padding: 0.4rem 0; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 75%; margin: 0 0 0 auto; padding: 0.55rem 1rem;
  border-radius: 18px 18px 4px 18px; background: #e6f2ec; border: 1px solid #d3e7dc;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) p { margin: 0; line-height: 1.5; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
</style>"""

# Starter questions offered before the first question: for the reports this
# project is evaluated on, and general ones for any other document.
_SUGGESTIONS = {
    "GEP": [":material/trending_up: What is the global growth forecast for 2026?",
            ":material/bar_chart: From the chart, how did moderate debt-distress risk change from 2015 to 2026?",
            ":material/public: What is the growth outlook for South Asia?",
            ":material/warning: What are the main downside risks to the outlook?"],
    "GPO": [":material/local_library: How many depository libraries did GPO visit in FY 2018?",
            ":material/sync_alt: What system replaced FDsys, and when?",
            ":material/payments: What were GPO's total revenues in FY 2018?"],
    "iMac": [":material/eco: What is the iMac's total carbon footprint?",
             ":material/bolt: How much less energy does iMac use than the ENERGY STAR requirement?",
             ":material/recycling: What is the iMac stand made of?"],
}
_GENERAL_SUGGESTIONS = [":material/summarize: Summarize this document in five points",
                        ":material/pin: What are the key numbers in this document?",
                        ":material/bar_chart: What do the charts in this document show?"]


def _suggestions(document_names: list[str]) -> list[str]:
    for key, questions in _SUGGESTIONS.items():
        if any(key.lower() in name.lower() for name in document_names):
            return questions
    return _GENERAL_SUGGESTIONS


def _question_from_suggestion(label: str) -> str:
    """The question without its ":material/...:" icon."""
    return re.sub(r"^:material/[a-z_]+:\s*", "", label)


def _format_location(source: str, page: int | None) -> str:
    return f"{source}, page {page}" if page is not None else source


def _render_answer_meta(source: str | None, searches: list[str], model: str | None,
                        chart_pages: list[str] | None = None, seconds: float | None = None) -> None:
    """One row of tags under an answer: where it came from, what was searched,
    which chart pages were read, which model answered and how long it took
    (like ChatGPT's "Searched …" line)."""
    with st.container(horizontal=True, gap="small"):
        if source == SOURCE_DOCUMENT:
            st.badge("From your documents", icon=":material/description:", color="green")
        elif source == SOURCE_GENERAL_KNOWLEDGE:
            st.badge("General knowledge, not from your documents", icon=":material/public:", color="blue")
        if searches:
            query = searches[0] if len(searches[0]) <= 40 else searches[0][:38].rstrip() + "…"
            label = f"Searched “{query}”" if len(searches) == 1 else f"Searched {len(searches)} times"
            st.badge(label, icon=":material/search:", color="gray", help="\n\n".join(f"“{q}”" for q in searches))
        if chart_pages:
            pages = " · ".join(re.sub(r"^.*, page ", "p. ", page) for page in chart_pages)
            st.badge(f"Chart read: {pages}", icon=":material/bar_chart:", color="violet",
                     help="\n\n".join(chart_pages))
        if model:
            st.badge(model.split("/")[-1], icon=":material/bolt:", color="gray")
        if seconds:
            st.badge(f"{seconds:.1f} s", icon=":material/timer:", color="gray")


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


def _render_passages(sources: list[RetrievedChunk], show_debug: bool) -> None:
    """Sources and (when on) retrieval details side by side, so an answer
    isn't followed by two full-width boxes."""
    if not sources:
        return
    with st.container(horizontal=True, gap="small"):
        _render_sources(sources)
        if show_debug:
            _render_debug(sources)


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


@st.dialog("Add documents", width="medium")
def _attach_dialog(ingest: IngestQueue) -> None:
    """Pick files and queue them for indexing; the dialog closes at once and
    their progress shows in the sidebar (render_ingest_status)."""
    files = st.file_uploader(
        "Choose documents",
        type=SUPPORTED_DOCUMENT_TYPES,
        accept_multiple_files=True,
        key=f"attach_uploader_{st.session_state.get('attach_round', 0)}",
        help="PDF, Word, text, Markdown, video or audio. Its text is searchable in a minute or so; "
             "a PDF's charts and tables are added after that.",
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        cancel = st.button("Cancel", key="attach_cancel")
        upload = st.button("Upload", type="primary", icon=":material/upload:", disabled=not files,
                           key="attach_upload")
    if upload:
        announced = st.session_state.setdefault("ingest_jobs", {})
        for file in files:
            announced[ingest.submit(file.name, file.getvalue())] = None
        st.session_state.attach_round = st.session_state.get("attach_round", 0) + 1
    if cancel or upload:
        st.rerun()


def render_ingest_status(ingest: IngestQueue, conversation_store: ConversationStore, conversation_id: str) -> None:
    """This session's uploads still being indexed, refreshed every two
    seconds while there are any; each milestone (text searchable, charts
    added, failed) is noted in the chat once."""
    if not st.session_state.get("ingest_jobs"):
        return
    st.fragment(run_every=2.0)(_ingest_status)(ingest, conversation_store, conversation_id)


def _ingest_status(ingest: IngestQueue, conversation_store: ConversationStore, conversation_id: str) -> None:
    announced = st.session_state.ingest_jobs  # job id -> last milestone noted in the chat
    notes = []
    for job in ingest.jobs(announced):
        if job.active:
            with st.container(border=True):
                st.markdown(f"**{job.name}**")
                st.progress(job.fraction, text=job.label)
        milestone = {SEARCHABLE: SEARCHABLE, READY: READY, FAILED: FAILED}.get(job.status)
        if milestone is None or announced[job.id] == milestone:
            continue
        notes += _milestone_notes(job, announced[job.id])
        announced[job.id] = milestone
        if job.status in (SEARCHABLE, READY) and job.name not in st.session_state.setdefault("document_names", []):
            st.session_state.document_names.append(job.name)
    for job_id in [j.id for j in ingest.jobs(announced) if not j.active]:
        del announced[job_id]  # noted; the document list shows it from now on
    if notes:
        st.session_state.chat_history.append({"role": "assistant", "content": "\n\n".join(notes), "sources": []})
        conversation_store.save(conversation_id, st.session_state.chat_history)
        st.rerun()  # the whole page, so the note and the document list appear


def _milestone_notes(job: IngestJob, noted: str | None) -> list[str]:
    """The chat note(s) for a job reaching its current status, given the
    milestone already `noted` (None, or SEARCHABLE)."""
    if job.status == FAILED:
        return [f"⚠️ **{job.name}** could not be added: {job.error}"]
    notes = []
    if noted is None:
        note = f"📄 **{job.name}** is ready: {_ready_note({'stats': job.stats, 'added': job.added})}."
        if job.charts_pending:
            note += " Ask away — its charts and tables are still being read."
        notes.append(note)
    if job.status == READY and noted == SEARCHABLE:
        if job.warning:
            notes.append(f"⚠️ **{job.name}**: {job.warning}")
        elif job.visual_passages:
            notes.append(f"📊 Charts and tables from **{job.name}** added "
                         f"({job.visual_passages:,} passage{'s' if job.visual_passages != 1 else ''}).")
    return notes


def render_chat(
    pipeline: RAGPipeline,
    ingest: IngestQueue,
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
    st.html(_CHAT_STYLE)

    for message in st.session_state.chat_history:
        avatar = APP_ICON if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            for filename in message.get("attached_files", []):
                st.caption(f"📎 {filename}")
            if message.get("voice"):
                st.caption("🎤 Voice message")
            if message["content"]:
                st.markdown(_md(message["content"]))
            if message["role"] == "assistant" and (message.get("source") or message.get("model")):
                _render_answer_meta(message.get("source"), message.get("searches", []), message.get("model"),
                                    message.get("chart_pages"), message.get("seconds"))
            if message["role"] == "assistant":
                _render_passages(message.get("sources", []), message.get("show_debug"))

    suggested = None
    documents = st.session_state.get("document_names") or []
    if documents and not any(m["role"] == "user" for m in st.session_state.chat_history):
        picked = st.pills("Try asking", _suggestions(documents), label_visibility="collapsed",
                          key=f"suggestion_{st.session_state.get('suggestion_round', 0)}")
        if picked:
            suggested = _question_from_suggestion(picked)
            st.session_state.suggestion_round = st.session_state.get("suggestion_round", 0) + 1

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
        _attach_dialog(ingest)
    if not submission and not suggested:
        return

    question = suggested or (submission.text or "").strip()
    audio = None if suggested else submission.audio

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
        searches, answer_model, chart_pages, seconds = [], None, [], None

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
            started = time.perf_counter()
            with st.status("Searching your documents...", expanded=False) as status:
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
                seconds = time.perf_counter() - started
                for query in searches:
                    st.write(f":material/search: Searched “{query}”")
                for page in chart_pages:
                    st.write(f":material/bar_chart: Read the chart on {page}")
                if sources:
                    pages = sorted({_format_location(c.source, c.page) for c in sources})
                    st.write(f":material/description: Used {len(pages)} page(s): " + "; ".join(pages))
                status.update(label=f"Answered in {seconds:.1f} s", state="complete")

        st.markdown(_md(answer))
        _render_answer_meta(source_label, searches, answer_model, chart_pages, seconds)
        _render_passages(sources, show_debug)

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
            "seconds": seconds,
        }
    )
    conversation_store.save(conversation_id, st.session_state.chat_history)
    st.rerun()
