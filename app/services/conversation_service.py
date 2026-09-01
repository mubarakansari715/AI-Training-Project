"""Conversation persistence.

Saves each chat conversation as its own JSON file, so past
conversations survive an app restart and can be reopened later —
similar to ChatGPT's conversation history sidebar.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_DIR = "data/conversations"
MAX_TITLE_LENGTH = 50


class ConversationStore:
    """Reads/writes chat conversations, one JSON file per conversation."""

    def __init__(self, storage_dir: str | None = None):
        self.storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def save(self, conversation_id: str, messages: list[dict]) -> None:
        """Save/overwrite a conversation. A conversation with no
        messages yet isn't written, so empty "new chats" don't
        clutter the sidebar list."""
        if not messages:
            return

        payload = {
            "id": conversation_id,
            "title": _make_title(messages),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": [_serialize_message(m) for m in messages],
        }
        try:
            self._path(conversation_id).write_text(json.dumps(payload, indent=2))
        except OSError:
            logger.exception("Failed to save conversation %s", conversation_id)

    def load(self, conversation_id: str) -> list[dict]:
        """Return the message list for a conversation, or [] if missing/corrupt."""
        path = self._path(conversation_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load conversation %s", conversation_id)
            return []
        return [_deserialize_message(m) for m in payload.get("messages", [])]

    def list_conversations(self) -> list[dict]:
        """Return [{"id", "title", "updated_at"}, ...], most recently updated first."""
        summaries = []
        for path in self.storage_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable conversation file: %s", path)
                continue
            summaries.append(
                {
                    "id": payload.get("id", path.stem),
                    "title": payload.get("title") or "New conversation",
                    "updated_at": payload.get("updated_at", ""),
                }
            )
        summaries.sort(key=lambda s: s["updated_at"], reverse=True)
        return summaries

    def delete(self, conversation_id: str) -> None:
        self._path(conversation_id).unlink(missing_ok=True)

    def _path(self, conversation_id: str) -> Path:
        return self.storage_dir / f"{conversation_id}.json"


def _make_title(messages: list[dict]) -> str:
    """Auto-title from the first user message, like ChatGPT does."""
    first_user_message = next(
        (m["content"] for m in messages if m["role"] == "user"), ""
    )
    title = re.sub(r"\s+", " ", first_user_message).strip()
    if len(title) > MAX_TITLE_LENGTH:
        title = title[:MAX_TITLE_LENGTH].rstrip() + "…"
    return title or "New conversation"


def _serialize_message(message: dict) -> dict:
    return {
        "role": message["role"],
        "content": message["content"],
        "sources": [asdict(chunk) for chunk in message.get("sources", [])],
        "show_debug": message.get("show_debug", False),
        "attached_files": message.get("attached_files", []),
    }


def _deserialize_message(message: dict) -> dict:
    return {
        "role": message["role"],
        "content": message["content"],
        "sources": [RetrievedChunk(**c) for c in message.get("sources", [])],
        "show_debug": message.get("show_debug", False),
        "attached_files": message.get("attached_files", []),
    }
