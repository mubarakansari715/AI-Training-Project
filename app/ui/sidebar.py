"""Sidebar: conversation history, document list, and settings UI.

Document upload itself lives in the chat input (the attach icon,
ChatGPT/Gemini-style) — this only shows what's already indexed and
lets you manage it.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.ui.components import render_document_list


@dataclass
class SidebarState:
    """Everything the main app needs to know from this render of the sidebar."""

    new_chat: bool = False
    selected_conversation_id: str | None = None
    delete_conversation_id: str | None = None
    clear_index: bool = False
    show_debug: bool = False
    strict_mode: bool = True


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

    return SidebarState(
        new_chat=new_chat,
        selected_conversation_id=selected_conversation_id,
        delete_conversation_id=delete_conversation_id,
        clear_index=clear_index,
        show_debug=show_debug,
        strict_mode=strict_mode,
    )
