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
# any other ID either provider's API key has access to. OpenRouter's are
# listed first since it's the primary provider (see app/rag/generator.py);
# Gemini's are the fallback. "openrouter/auto" is the default — it lets
# OpenRouter pick a suitable model for you.
OPENROUTER_MODEL_OPTIONS = [
    "openrouter/auto",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
]
# Check available model names for your key in Google AI Studio, since
# Gemini model names change over time.
GEMINI_MODEL_OPTIONS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]
CUSTOM_MODEL_OPTION = "Custom..."


def _is_gemini_model(model_id: str) -> bool:
    """Route a model ID to Gemini if it looks like one, OpenRouter otherwise.

    OpenRouter IDs are always "provider/model" (or "openrouter/auto");
    Gemini's own IDs are the only ones that start with "gemini-" bare, so
    this also correctly routes a typed-in "Custom..." entry.
    """
    return model_id in GEMINI_MODEL_OPTIONS or model_id.startswith("gemini-")


@dataclass
class SidebarState:
    """Everything the main app needs to know from this render of the sidebar."""

    new_chat: bool = False
    selected_conversation_id: str | None = None
    delete_conversation_id: str | None = None
    clear_index: bool = False
    show_debug: bool = False
    strict_mode: bool = settings.strict_document_mode


def render_model_picker() -> tuple[str | None, str | None]:
    """Render the model picker, returning
    `(gemini_model_override, openrouter_model_override)`.

    Exactly one of the two is set, matching whichever model the picker
    chose; the other is None so that provider keeps its own configured
    default (see AnswerGenerator.generate()).

    Meant to be rendered directly next to the chat input (see
    app/ui/chat.py) — that's where model switching lives in most chat
    apps, not tucked away in the sidebar.
    """
    all_models = [*OPENROUTER_MODEL_OPTIONS, *GEMINI_MODEL_OPTIONS]
    default_model = st.session_state.get("model_name", settings.openrouter_model)
    known_models = all_models if default_model in all_models else [default_model, *all_models]

    model_choice = st.selectbox(
        "Model",
        options=[*known_models, CUSTOM_MODEL_OPTION],
        index=known_models.index(default_model),
        label_visibility="collapsed",
        width=280,
        help="OpenRouter models are tried first, Gemini models as the "
        "fallback — picking one here overrides that provider's model, "
        "the other keeps its configured default. 'openrouter/auto' picks "
        "a model for you automatically.",
    )
    if model_choice == CUSTOM_MODEL_OPTION:
        model_name = (
            st.text_input(
                "Custom model ID",
                placeholder="e.g. anthropic/claude-3.5-sonnet or gemini-3.1-pro-preview",
                help="Any OpenRouter model ID (see openrouter.ai/models) or "
                "Gemini model ID your API key can access.",
            ).strip()
            or settings.openrouter_model
        )
    else:
        model_name = model_choice
    st.session_state.model_name = model_name

    if _is_gemini_model(model_name):
        return model_name, None
    return None, model_name


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
                value=st.session_state.get("strict_mode", settings.strict_document_mode),
                help="Off (default): answers come from your documents when "
                "possible, and from the model's general knowledge otherwise "
                "— each answer shows which one it came from. On: only ever "
                "answer from your documents, refusing when nothing relevant "
                "is found.",
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
