"""Sidebar: conversation history, document list, and settings UI.

Document upload itself lives in the chat input (the attach icon,
ChatGPT/Gemini-style) — this only shows what's already indexed and
lets you manage it.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.config.settings import settings
from app.ui.components import render_document_list

# Curated, known-good model IDs for the picker; "Custom..." lets you type
# any other ID your API key has access to (Google AI Studio has the
# current list, since Gemini model names change over time).
GEMINI_MODEL_OPTIONS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]
CUSTOM_MODEL_OPTION = "Custom..."


@dataclass
class SidebarState:
    """Everything the main app needs to know from this render of the sidebar."""

    new_chat: bool = False
    selected_conversation_id: str | None = None
    delete_conversation_id: str | None = None
    clear_index: bool = False
    show_debug: bool = False
    strict_mode: bool = True
    model_name: str = settings.gemini_model
    temperature: float = settings.gemini_temperature


def render_sidebar(
    conversations: list[dict], active_conversation_id: str | None
) -> SidebarState:
    """Render the sidebar and return the user's current choices.

    `conversations` is [{"id", "title", "updated_at"}, ...], most
    recently updated first — the saved conversation history shown
    below "New chat", ChatGPT-style.
    """
    with st.sidebar:
        st.title("🤖 RAG Chatbot")

        new_chat = st.button(
            "New chat", icon=":material/edit_square:", width="stretch"
        )

        st.subheader(":material/forum: Conversations")
        selected_conversation_id = None
        delete_conversation_id = None
        if not conversations:
            st.caption("No saved conversations yet.")
        else:
            for conversation in conversations:
                is_active = conversation["id"] == active_conversation_id
                title_col, delete_col = st.columns([6, 1], vertical_alignment="center")
                clicked = title_col.button(
                    conversation["title"],
                    key=f"conversation_{conversation['id']}",
                    width="stretch",
                    type="primary" if is_active else "secondary",
                )
                if clicked:
                    selected_conversation_id = conversation["id"]

                deleted = delete_col.button(
                    "",
                    icon=":material/delete:",
                    key=f"delete_conversation_{conversation['id']}",
                    help=f"Delete '{conversation['title']}'",
                )
                if deleted:
                    delete_conversation_id = conversation["id"]

        st.divider()
        st.subheader(":material/description: Documents")
        st.caption("Attach files using the 📎 icon in the chat box below.")

        if "document_names" not in st.session_state:
            st.session_state.document_names = []
        render_document_list(st.session_state.document_names)

        clear_index = st.button("Clear index", width="stretch")

        st.divider()

        with st.expander("⚙️ Advanced settings"):
            strict_mode = st.checkbox(
                "Strict document mode",
                value=st.session_state.get("strict_mode", True),
                help="When on, answers come only from your documents. If "
                "nothing relevant is found, the assistant says so instead "
                "of guessing.",
            )
            st.session_state.strict_mode = strict_mode

            show_debug = st.checkbox(
                "🔍 Show retrieval details",
                value=st.session_state.get("show_debug", False),
                help="Show the raw chunks retrieved for each answer, with "
                "their similarity scores — useful to verify retrieval is "
                "working.",
            )
            st.session_state.show_debug = show_debug

            st.divider()

            default_model = st.session_state.get("model_name", settings.gemini_model)
            known_models = (
                GEMINI_MODEL_OPTIONS
                if default_model in GEMINI_MODEL_OPTIONS
                else [default_model, *GEMINI_MODEL_OPTIONS]
            )
            model_choice = st.selectbox(
                "Model",
                options=[*known_models, CUSTOM_MODEL_OPTION],
                index=known_models.index(default_model),
                help="Which Gemini model answers your questions. "
                "'flash-lite'/'flash' are fast and cheap; 'pro' is slower "
                "but handles harder questions better.",
            )
            if model_choice == CUSTOM_MODEL_OPTION:
                model_name = (
                    st.text_input(
                        "Custom model ID",
                        placeholder="e.g. gemini-3.1-pro-preview",
                        help="Any model ID available for your API key — "
                        "check Google AI Studio for the current list.",
                    ).strip()
                    or settings.gemini_model
                )
            else:
                model_name = model_choice
            st.session_state.model_name = model_name

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.get("temperature", settings.gemini_temperature)),
                step=0.1,
                help="Lower (e.g. 0.2) gives literal, consistent answers "
                "grounded in your documents. Higher (1.0+) gives more "
                "varied, creative phrasing but can drift from the source "
                "text — keep it low for factual document Q&A.",
            )
            st.session_state.temperature = temperature
            st.caption(
                "0.2 = literal · 1.0 = balanced (Gemini default) · 2.0 = max creativity"
            )

    return SidebarState(
        new_chat=new_chat,
        selected_conversation_id=selected_conversation_id,
        delete_conversation_id=delete_conversation_id,
        clear_index=clear_index,
        show_debug=show_debug,
        strict_mode=strict_mode,
        model_name=model_name,
        temperature=temperature,
    )
