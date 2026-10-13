"""Query rewriting before retrieval.

Embedding search is sensitive to typos ("wat is my vitamn d levl" scores
far lower than the correct spelling) and can't resolve follow-ups ("and
my cholesterol?"), while the LLM handles both easily. So before
searching, one short LLM call rewrites the user's message into a clean,
self-contained question (left unchanged when it already is one — rewriting
a clear question into keywords was measured to hurt the reranker) — the standard "condense question" step of
chat-style RAG apps. Only retrieval uses the rewrite; the answer step
still sees the user's original words and the conversation.

Any failure (no provider, error, empty or runaway output) falls back to
the original question, so rewriting can only ever help retrieval.
"""

from __future__ import annotations

import logging

from app.rag.generator import AnswerGenerator

logger = logging.getLogger(__name__)

# Recent turns given to the rewriter, and how much of each — enough to
# resolve "it"/"that", without sending long answers back in full.
_HISTORY_MESSAGES = 6
_HISTORY_CHARS = 300

REWRITE_PROMPT = """Prepare the user's latest message for searching their uploaded documents.
- If it is already a clear, correctly spelled, self-contained question, return it \
exactly unchanged.
- Otherwise fix spelling mistakes and typos, and if it is a follow-up, use the \
conversation to make it self-contained (replace words like "it", "that" or \
"and my X?" with what they refer to).
- Always return a natural-language question, never a keyword list, and keep the \
user's meaning, names, numbers and language. Do not answer it.
Return only the question, on a single line.

Conversation so far:
{history}

Latest message: {question}"""


def _format_history(history: list[dict]) -> str:
    recent = history[-_HISTORY_MESSAGES:]
    if not recent:
        return "(none)"
    return "\n".join(
        f"{turn['role']}: {turn['content'][:_HISTORY_CHARS]}" for turn in recent
    )


def _clean(output: str, question: str) -> str | None:
    """First non-empty line, minus quotes/labels; None if unusable."""
    line = next((ln.strip() for ln in output.splitlines() if ln.strip()), "")
    for prefix in ("query:", "search query:", "rewritten query:"):
        if line.lower().startswith(prefix):
            line = line[len(prefix):].strip()
    line = line.strip("\"'` ")
    # A rewrite much longer than the question means the model answered
    # or rambled instead of rewriting.
    if not line or len(line) > max(200, 4 * len(question)):
        return None
    return line


def rewrite_query(
    generator: AnswerGenerator,
    question: str,
    history: list[dict] | None = None,
    model_name: str | None = None,
    openrouter_model_name: str | None = None,
) -> str:
    """Return a spelling-corrected, self-contained version of `question`
    for retrieval, or `question` itself if rewriting isn't possible."""
    if not generator.is_configured():
        return question
    prompt = REWRITE_PROMPT.format(history=_format_history(history or []), question=question)
    output = generator.complete(
        prompt,
        model_name=model_name,
        openrouter_model_name=openrouter_model_name,
        temperature=0.0,
    )
    rewritten = _clean(output, question)
    if rewritten is None:
        return question
    if rewritten != question:
        logger.info("Rewrote query %r -> %r", question, rewritten)
    return rewritten
