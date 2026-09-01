"""Small reusable Streamlit UI components."""

import streamlit as st

SUPPORTED_DOCUMENT_TYPES = ["pdf", "docx", "txt", "md"]

# ChatGPT-style chat layout: centered column, a rounded bubble
# right-aligned for the user's turn, plain left-aligned text (with
# avatar) for the assistant's turn. Targets Streamlit's chat-message
# data-testids, the documented public hooks for CSS customization.
_CHAT_STYLE = """
<style>
.stMainBlockContainer {
    max-width: 800px;
    margin: 0 auto;
}

[data-testid="stChatMessage"] {
    margin-bottom: 1rem;
}

/* User turn: hide the avatar, render as a right-aligned bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
    display: none;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background-color: rgb(233, 233, 236);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 20px;
    padding: 0.65rem 1.1rem;
    max-width: 75%;
}

@media (prefers-color-scheme: dark) {
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background-color: rgb(52, 53, 58);
        border-color: rgba(255, 255, 255, 0.12);
    }
}

/* Assistant turn: plain text, no bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background-color: transparent;
    padding: 0.3rem 0;
    max-width: 100%;
}
</style>
"""


def inject_chat_theme() -> None:
    """Apply ChatGPT-like styling: centered column + bubble-style user turns."""
    st.html(_CHAT_STYLE)


def render_document_list(filenames: list[str]) -> None:
    """Render the list of currently uploaded documents in the sidebar."""
    st.markdown(f"**Documents: {len(filenames)}**")

    if not filenames:
        st.caption("No documents uploaded yet.")
        return

    for name in filenames:
        st.markdown(f"✓ {name}")
