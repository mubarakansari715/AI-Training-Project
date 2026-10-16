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

`needs_rewrite` skips that LLM call (about a second and 100-150 tokens)
when there is nothing to fix: every word is a common English word or
appears in the indexed documents, and it is not a follow-up. It errs
towards rewriting: any unknown word, even a correct one, still rewrites.
"""

from __future__ import annotations

import logging
import re

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


_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")

# Words a question uses that a document may not: question words, function
# words and common verbs. Anything else must appear in the documents.
COMMON_WORDS = frozenset("""
a about above across after again against all also am among an and any are as at be because been before
being below between both but by can could did do does doing done during each either else ever every
few for from had has have having how however if in into is it its just least less like many may me
might more most much must my no nor not now of off on once only or other our out over own per rather
same several should since so some such than that the their them then there these they this those
through to too under until up upon us very was we were what whatever when where whether which while
who whom whose why will with within without would yes yet you your
according amount average change compare compared comparison describe difference doc
document documents explain find give happen happened increase increased list main mean meant
mention mentioned number page pages percent percentage report reported say says said show shown
size state stated summarize summary tell total value year years
""".split())

# Words that point back at the conversation ("it", "that one", "and the 2027 figure?").
_FOLLOW_UP_WORDS = frozenset("""
it its that those this these they them their he she him his her one ones same also else
above previous earlier again latter former
""".split())
_FOLLOW_UP_START_RE = re.compile(r"^\s*(and|also|but|so|what about|how about)\b", re.IGNORECASE)


def words(text: str) -> list[str]:
    """Lower-case words without possessives or contractions ("iMac's" -> "imac")."""
    return [re.split(r"['’]", w)[0] for w in _WORD_RE.findall(text.lower())]


def _known(word: str, vocabulary: set[str]) -> bool:
    if word.isdigit() or any(ch.isdigit() for ch in word) or word in COMMON_WORDS or word in vocabulary:
        return True
    # Plural/singular of a document word ("chart" vs "charts").
    return (word.endswith("s") and word[:-1] in vocabulary) or word + "s" in vocabulary


def needs_rewrite(question: str, history: list[dict] | None, vocabulary: set[str]) -> bool:
    """False when rewriting cannot change the search: no conversation to
    resolve against (or no follow-up words) and every word is known."""
    tokens = words(question)
    if history and (_FOLLOW_UP_START_RE.match(question) or _FOLLOW_UP_WORDS.intersection(tokens)):
        return True
    return not all(_known(word, vocabulary) for word in tokens)


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
    groq_model_name: str | None = None,
) -> str:
    """Return a spelling-corrected, self-contained version of `question`
    for retrieval, or `question` itself if rewriting isn't possible."""
    if not generator.is_configured():
        return question
    prompt = REWRITE_PROMPT.format(history=_format_history(history or []), question=question)
    output = generator.complete(
        prompt,
        model_name=model_name,
        groq_model_name=groq_model_name,
        temperature=0.0,
    )
    rewritten = _clean(output, question)
    if rewritten is None:
        return question
    if rewritten != question:
        logger.info("Rewrote query %r -> %r", question, rewritten)
    return rewritten
