"""Query decomposition for multi-part questions.

One search for "What replaced FDsys, and when was it fully deployed?" or
"Compare passports and copies distributed" ranks chunks against the whole
question, so the best match for one part can push the other part's
evidence out of the top-k. Splitting the question into self-contained
sub-questions and searching each (app.rag.retriever.Retriever.retrieve_many)
gives every part its own share of the context. The answer step still sees
the original question.

`needs_decomposition` is a cheap check (no LLM call) for comparison and
multi-part wording; only then does one short LLM call split the question.
Questions that name document sections are already searched per section by
the retriever, so they skip this. Any failure (no provider, error,
unusable output) falls back to the single question, so decomposition can
only add searches, never lose the original one.

Auto (agent) mode does its own version of this: the model decides what
to search, and how many times.
"""

from __future__ import annotations

import json
import logging
import re

from app.rag.generator import AnswerGenerator

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 3

# "compare", "versus", "difference between", "both", or a second question
# joined on with ", and when/how/what…". Not "compared to/with", which
# measures one thing against a reference ("less energy compared to ENERGY STAR").
_MULTI_PART_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|differ|both)\b"
    r"|,?\s+and\s+(when|how|what|where|why|who|which)\b",
    re.IGNORECASE,
)

DECOMPOSE_PROMPT = """Split this question about the user's documents into separate search questions.
- If it asks about several things (a comparison, or two or more questions joined \
together), return one self-contained question per thing, at most {limit}. Repeat the \
names and details each one needs, since each is searched on its own.
- If it asks about only one thing, return just the question unchanged.
- Keep the user's names, numbers and language. Do not answer it.
Return only a JSON array of strings.

Question: {question}"""


def needs_decomposition(question: str) -> bool:
    return bool(_MULTI_PART_RE.search(question))


def _parse(output: str, question: str) -> list[str] | None:
    """The sub-questions in `output`, or None if it isn't a usable JSON array."""
    match = re.search(r"\[.*\]", output or "", re.DOTALL)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    queries = []
    for item in items:
        # A sub-question much longer than the question means the model answered or rambled.
        if not isinstance(item, str) or not item.strip() or len(item) > max(200, 2 * len(question)):
            return None
        if item.strip() not in queries:
            queries.append(item.strip())
    return queries[:MAX_SUB_QUESTIONS] or None


def decompose_query(
    generator: AnswerGenerator,
    question: str,
    model_name: str | None = None,
    groq_model_name: str | None = None,
) -> list[str]:
    """Self-contained sub-questions for `question`, or `[question]` when it
    has only one part or splitting isn't possible."""
    if not generator.is_configured():
        return [question]
    prompt = DECOMPOSE_PROMPT.format(limit=MAX_SUB_QUESTIONS, question=question)
    output = generator.complete(prompt, model_name=model_name, groq_model_name=groq_model_name, temperature=0.0)
    queries = _parse(output, question)
    if not queries or len(queries) == 1:
        return [question]
    logger.info("Decomposed query %r -> %r", question, queries)
    return queries
