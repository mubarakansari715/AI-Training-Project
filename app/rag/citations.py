"""Parse the page citations an answer makes ("p. 5", "pages 14 and 15"),
and turn excerpt-number markers ("【2】") into readable ones."""

from __future__ import annotations

import re

from app.rag.retriever import RetrievedChunk

_PAGE_REF_RE = re.compile(r"\b(?:pp?|pages?)\.?\s*(\d+(?:\s*(?:,|and|&|–|-|to)\s*\d+)*)", re.IGNORECASE)
# Some models (gpt-oss) cite the numbered excerpts in their own style,
# "【2】" or "【2†L10-L12】", instead of the requested plain-text citation.
_BRACKET_REF_RE = re.compile(r"\s*【(\d+)(?:†[^】]*)?】")


def cited_pages(answer: str) -> set[int]:
    """Page numbers an answer cites, e.g. 'p. 5', 'pages 14 and 15', 'pp. 4–5'."""
    pages: set[int] = set()
    for match in _PAGE_REF_RE.finditer(answer):
        pages.update(int(n) for n in re.findall(r"\d+", match.group(1)))
    return pages


def cited_chunks(answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """The chunks an answer actually cites by page. Chunks without page
    numbers (TXT/DOCX, transcripts) can't be matched that way, so they're
    kept whenever the answer cites no pages at all."""
    pages = cited_pages(answer)
    if not pages:
        return [c for c in chunks if c.page is None]
    return [c for c in chunks if c.page in pages]


def resolve_bracket_refs(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Replace "【n】" markers with the n-th excerpt's "(source, page p)",
    or drop a marker that names no excerpt."""
    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        if not 1 <= n <= len(chunks):
            return ""
        chunk = chunks[n - 1]
        return f" ({chunk.source}" + (f", page {chunk.page})" if chunk.page is not None else ")")

    return _BRACKET_REF_RE.sub(replace, answer)
