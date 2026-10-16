"""LLM routing of a question to document sections.

The rule-based router (app.rag.structure.sections_mentioned) filters the
search to sections a question names ("South Asia", "Chapter 3"). It can't
see that "India" belongs to South Asia or that "sovereign spreads" is a
Chapter 3 topic. With QUERY_ROUTING=llm, a question that names no section
gets one short LLM call that picks the sections, from the document's own
section list, that would answer it.

The field guide's warning applies: a wrong filter hides the answer. So the
rules still go first, the model may answer "none" (no filter), only exact
labels from the list are accepted, and a filtered search that finds nothing
falls back to searching everything (app.rag.retriever).
"""

from __future__ import annotations

import json
import logging
import re

from app.rag.generator import AnswerGenerator

logger = logging.getLogger(__name__)

MAX_SECTIONS = 3

ROUTE_PROMPT = """A question about a long report, and the report's sections.
Pick the sections (at most {limit}) that would contain the answer, copied exactly from the list.
If the question is general or could be answered anywhere in the report, return [].
Return only a JSON array of section names.

Sections:
{sections}

Question: {question}"""


def _parse(output: str, labels: list[str]) -> list[str]:
    match = re.search(r"\[.*\]", output or "", re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    known = set(labels)
    picked = [item for item in items if isinstance(item, str) and item in known] if isinstance(items, list) else []
    return list(dict.fromkeys(picked))[:MAX_SECTIONS]


def llm_route(generator: AnswerGenerator, question: str, labels: list[str],
              groq_model_name: str | None = None) -> list[str]:
    """The section labels the model picks for `question`; [] = no filter
    (also on any failure, so routing can never block a search)."""
    if not labels or not generator.is_configured():
        return []
    prompt = ROUTE_PROMPT.format(limit=MAX_SECTIONS, sections="\n".join(f"- {label}" for label in labels),
                                 question=question)
    picked = _parse(generator.complete(prompt, groq_model_name=groq_model_name, temperature=0.0), labels)
    if picked:
        logger.info("Routed %r -> %r", question, picked)
    return picked
