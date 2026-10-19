"""Application entry point.

Full RAG pipeline: upload (via the chat input's attach icon) -> parse
-> clean -> chunk -> embed -> store -> question -> retrieve -> build
context -> Groq/Gemini -> answer + sources. Conversations are
persisted to disk (ConversationStore) so past chats survive an app
restart and can be reopened from the sidebar.
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is importable so submodules can use
# absolute imports like `from app.ui.sidebar import render_sidebar`,
# matching how the test suite imports the same package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config.settings import APP_ICON, APP_NAME
from app.services.conversation_service import ConversationStore
from app.services.ingest_jobs import IngestQueue, ProcessRunner
from app.services.rag_service import RAGPipeline
from app.ui.chat import render_chat, render_ingest_status
from app.ui.evaluation_page import render_evaluation_page
from app.ui.sidebar import render_sidebar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_pipeline() -> RAGPipeline:
    """Build the RAG pipeline once and reuse it across reruns/sessions.

    Loading the embedding model and opening the persistent ChromaDB
    collection are both relatively expensive, so this is cached as a
    shared resource instead of being recreated on every rerun.
    """
    return RAGPipeline()


@st.cache_resource
def get_ingest_queue() -> IngestQueue:
    """The background indexing queue, shared by every session (one worker
    writing to the shared pipeline, reading in a worker process;
    app.services.ingest_jobs)."""
    return IngestQueue(get_pipeline(), run=ProcessRunner())


@st.cache_resource
def get_conversation_store() -> ConversationStore:
    """Shared handle to the on-disk conversation history."""
    return ConversationStore()


def switch_conversation(conversation_id: str, store: ConversationStore) -> None:
    """Make `conversation_id` the active conversation and load its messages."""
    st.session_state.conversation_id = conversation_id
    st.session_state.chat_history = store.load(conversation_id)


def handle_conversation_actions(store: ConversationStore, sidebar) -> None:
    """Apply New chat / switch / delete actions requested from the sidebar."""
    if sidebar.new_chat:
        switch_conversation(store.new_id(), store)
        st.rerun()

    if sidebar.selected_conversation_id:
        switch_conversation(sidebar.selected_conversation_id, store)
        st.rerun()

    if sidebar.delete_conversation_id:
        store.delete(sidebar.delete_conversation_id)
        # Deleting the active conversation leaves nothing to show, so
        # start a fresh one instead of pointing at a now-missing file.
        if sidebar.delete_conversation_id == st.session_state.get("conversation_id"):
            switch_conversation(store.new_id(), store)
        st.rerun()


def handle_document_management(pipeline: RAGPipeline, sidebar) -> None:
    """Apply the clear-index action requested from the sidebar."""
    if sidebar.clear_index:
        try:
            pipeline.clear_index()
            st.session_state.document_names = []
            st.success("Document index cleared.")
        except Exception:
            logger.exception("Failed to clear index")
            st.error("Could not clear the document index. Please try again.")


def chat_page() -> None:
    try:
        pipeline = get_pipeline()
    except Exception:
        logger.exception("Failed to initialize the RAG pipeline")
        st.error(
            "Failed to initialize the RAG engine. Check your setup "
            "(dependencies, disk permissions for the vector store) and try again."
        )
        st.stop()

    conversation_store = get_conversation_store()
    ingest = get_ingest_queue()

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = conversation_store.new_id()
        st.session_state.chat_history = []

    if "document_names" not in st.session_state:
        # Documents indexed in earlier sessions are still searchable.
        try:
            st.session_state.document_names = pipeline.vector_store.sources()
        except Exception:
            logger.exception("Could not list the indexed documents")
            st.session_state.document_names = []

    sidebar = render_sidebar(
        conversations=conversation_store.list_conversations(),
        active_conversation_id=st.session_state.conversation_id,
        document_status=lambda: render_ingest_status(ingest, conversation_store, st.session_state.conversation_id),
        indexing=ingest.busy(),
    )

    handle_conversation_actions(conversation_store, sidebar)
    handle_document_management(pipeline, sidebar)

    if not st.session_state.get("chat_history"):
        st.title(f"Welcome to {APP_NAME}")
        st.write(
            "Add documents with the **Attach** button above the chat box, "
            "then ask questions about their content."
        )

        if not pipeline.generator.is_configured():
            st.info(
                "No LLM provider configured — set GROQ_API_KEY "
                "(preferred) or GEMINI_API_KEY (fallback) in your .env file "
                "to get real answers. You can still upload and index "
                "documents in the meantime.",
                icon="ℹ️",
            )

    render_chat(
        pipeline,
        ingest,
        conversation_store,
        st.session_state.conversation_id,
        show_debug=sidebar.show_debug,
        strict_mode=sidebar.strict_mode,
        temperature=sidebar.temperature,
        auto_mode=sidebar.auto_mode,
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide",
    )
    page = st.navigation(
        [
            st.Page(chat_page, title="Chat", icon=":material/chat:", default=True),
            st.Page(render_evaluation_page, title="Use case results", icon=":material/fact_check:",
                    url_path="results"),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
