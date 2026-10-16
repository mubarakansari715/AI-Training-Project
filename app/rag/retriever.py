"""Semantic retrieval.

Embeds a user's question and searches the vector store for the most
relevant document chunks, so only that (small) context is ever sent
to the LLM instead of the entire document collection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config.settings import settings
from app.rag.embeddings import EmbeddingService
from app.rag.structure import is_front_matter, sections_mentioned
from app.rag.vector_store import VectorStore
from app.rag.visual import footnote_label

# A request to summarize a whole document/video ("summarize this video",
# "what is the report about?"). Those need broad coverage, not just the
# single closest chunk — and generic wording like "summarize" scores low
# against specific content, so the similarity threshold would drop most
# of the document. Both an intent word and a document word are required,
# so "summarize the French Revolution" is still a normal question.
_SUMMARY_INTENT_RE = re.compile(
    r"\b(summar\w*|overview|tl;?dr|gist|key\s+points|main\s+points|about)\b", re.IGNORECASE
)
_DOCUMENT_WORD_RE = re.compile(
    r"\b(video|audio|recording|transcript|document|doc|file|report|pdf|upload\w*)s?\b", re.IGNORECASE
)
_SUMMARY_TOP_K_MULTIPLIER = 3
_SUMMARY_TOP_K_MAX = 15
# Small-to-big: for a question about one named section, the best-matching
# chunks' whole pages go to the LLM. A section's headline facts (a region's
# forecast, a chapter's key numbers) sit on its opening page but often in a
# chunk that scored lower than its neighbours. Capped to bound the prompt.
_EXPAND_MAX_PAGES = 2
# Shortest repeated text treated as chunk overlap when joining a page.
_MIN_OVERLAP = 20
# A footnote marker in a retrieved passage ("more than 15 percent recycled
# content.[1]") pulls in that note's passage from the same page (see
# app.rag.visual), since the note itself rarely shares the question's words.
_FOOTNOTE_MARKER_RE = re.compile(r"\[(\d{1,3})\]")
_FOOTNOTE_TEXT_RE = re.compile(r"\[Footnote (\d{1,3}), referenced on page")
_FOOTNOTES_MAX = 3


def is_summary_request(question: str) -> bool:
    return bool(_SUMMARY_INTENT_RE.search(question) and _DOCUMENT_WORD_RE.search(question))


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its similarity score and source metadata."""

    text: str
    source: str
    page: int | None
    score: float
    chunk_id: str
    section: str | None = None
    # Every page the chunk covers (a sentence-packed chunk can span pages).
    pages: list[int] = field(default_factory=list)

    def covers(self, pages) -> bool:
        """True when the chunk comes from any of `pages`."""
        return self.page in pages or any(page in pages for page in self.pages)


class Retriever:
    """Retrieves the most relevant chunks for a question."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int | None = None,
        score_threshold: float | None = None,
        reranker=None,
        rerank_candidates: int | None = None,
        section_aware: bool = True,
    ):
        self._vector_store = vector_store
        self._embeddings = embedding_service
        self.top_k = top_k or settings.top_k
        self.score_threshold = (
            settings.similarity_threshold if score_threshold is None else score_threshold
        )
        # Optional second stage (app.rag.reranker.Reranker): fetch a wider
        # candidate set, then let the reranker pick the best top_k.
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates or settings.rerank_candidates
        self.section_aware = section_aware

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return relevant chunks for `question`, most similar first.

        Summary requests (see `is_summary_request`) fetch more chunks and
        skip the similarity threshold, so the whole document is covered.
        With a reranker, `rerank_candidates` chunks are fetched, filtered
        by the threshold, and reranked down to top_k.
        """
        if not question or not question.strip():
            return []

        summary = is_summary_request(question)
        top_k = min(self.top_k * _SUMMARY_TOP_K_MULTIPLIER, _SUMMARY_TOP_K_MAX) if summary else self.top_k

        query_embedding = self._embeddings.embed_query(question)
        named, where = self._section_filter(question)
        if len(named) > 1:
            # A question naming several sections ("compare South Asia and
            # Sub-Saharan Africa") needs evidence from each: give every named
            # section its own share instead of letting one fill the top-k. A
            # section's headline sentence measured 3rd-4th within its own
            # section, so two sections each get a full top_k; more sections
            # split ~2x top_k between them to keep the prompt bounded.
            share = top_k if len(named) <= 2 else max(2, -(-2 * top_k // len(named)))
            results, seen = [], set()
            for group in named:
                for chunk in self._search(question, query_embedding, share, {"section": {"$in": group}}, summary):
                    if chunk.chunk_id not in seen:
                        seen.add(chunk.chunk_id)
                        results.append(chunk)
            return self._attach_footnotes(results)
        results = self._search(question, query_embedding, top_k, where, summary)
        if named and not results:  # named section exists but nothing relevant there: search everything
            results = self._search(question, query_embedding, top_k, None, summary)
        elif len(named) == 1:
            results = self._expand_to_pages(results)
        return self._attach_footnotes(results)

    def _attach_footnotes(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Append the footnote passages that retrieved chunks refer to with
        an [n] marker (same document and page, at most `_FOOTNOTES_MAX`).
        A note referenced from several pages has the same text on each, so
        it is added once, and not at all if it was already retrieved."""
        notes = {(c.source, int(n)) for c in results for n in _FOOTNOTE_TEXT_RE.findall(c.text)}
        wanted = []
        for chunk in results:
            if chunk.page is None:
                continue
            for number in map(int, _FOOTNOTE_MARKER_RE.findall(chunk.text)):
                if (chunk.source, number) not in notes:
                    notes.add((chunk.source, number))
                    wanted.append((chunk.source, chunk.page, footnote_label(number, chunk.page), chunk.score))
        seen = {chunk.chunk_id for chunk in results}
        added = []
        for source, page, label, score in wanted:
            if len(added) == _FOOTNOTES_MAX:
                break
            for match in self._vector_store.page_chunks(source, page):
                if label in match["text"] and match["chunk_id"] not in seen:
                    seen.add(match["chunk_id"])
                    added.append(RetrievedChunk(
                        text=match["text"], source=source, page=page, score=score,
                        chunk_id=match["chunk_id"], section=match["section"],
                    ))
                    break
        return results + added

    def _expand_to_pages(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Replace the hits on the top `_EXPAND_MAX_PAGES` pages with those
        whole pages (reading order, same section); other hits stay as they are."""
        pages = []
        for chunk in results:
            key = (chunk.source, chunk.page)
            if chunk.page is not None and key not in pages and len(pages) < _EXPAND_MAX_PAGES:
                pages.append(key)
        expanded, seen = [], set()
        for chunk in results:
            key = (chunk.source, chunk.page)
            if key not in pages:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    expanded.append(chunk)
                continue
            matches = [m for m in self._vector_store.page_chunks(*key)
                       if m["chunk_id"] not in seen and m["section"] == chunk.section]
            if not matches:
                continue
            seen.update(m["chunk_id"] for m in matches)
            # One passage per page, not its overlapping chunks: fragments that
            # repeat each other's ends made the model skip a sentence's
            # continuation as a duplicate.
            expanded.append(RetrievedChunk(
                text=_join_page([m["text"] for m in matches], chunk.section), source=matches[0]["source"],
                page=matches[0]["page"], score=chunk.score, chunk_id=matches[0]["chunk_id"], section=chunk.section,
            ))
        return expanded

    def _search(
        self, question: str, query_embedding: list[float], top_k: int, where: dict | None, summary: bool
    ) -> list[RetrievedChunk]:
        """One filtered search: fetch candidates, apply the threshold, rerank to top_k."""
        fetch = max(top_k, self.rerank_candidates) if self.reranker else top_k
        matches = self._vector_store.search(query_embedding, top_k=fetch, where=where)

        results = []
        for match in matches:
            # Cosine distance in [0, 2] -> similarity score in [-1, 1].
            score = 1 - match["distance"]
            if score < self.score_threshold and not summary:
                continue
            results.append(_to_chunk(match, score))
        if self.reranker:
            return self.reranker.rerank(question, results, top_k)
        return results[:top_k]

    def _section_filter(self, question: str) -> tuple[list[list[str]], dict | None]:
        """(named groups, where): search only the sections the question names
        ("according to the executive summary…", "Chapter 3", "Box 1.1",
        "South Asia"); otherwise search everything except front matter
        (table of contents, abbreviations), whose titles match every topic."""
        labels = self._section_labels() if self.section_aware else []
        if not labels:
            return [], None
        named = sections_mentioned(question, labels)
        if named:
            return named, {"section": {"$in": sorted({label for group in named for label in group})}}
        front = [label for label in labels if is_front_matter(label)]
        return [], ({"section": {"$nin": front}} if front else None)

    def _section_labels(self) -> list[str]:
        """Distinct section labels in the index, cached until it changes."""
        count = self._vector_store.count()
        if getattr(self, "_labels_count", None) != count:
            self._labels = self._vector_store.sections()
            self._labels_count = count
        return self._labels


def _to_chunk(match: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(text=match["text"], source=match["source"], page=match["page"], score=score,
                          chunk_id=match["chunk_id"], section=match.get("section"), pages=match.get("pages") or [])


def _join_page(texts: list[str], section: str | None) -> str:
    """A page's chunks (reading order) as one text: the section label kept
    once, and each chunk's overlap with the one before it dropped."""
    label = f"[{section}]\n" if section and texts[0].startswith(f"[{section}]\n") else ""
    pieces = [t[len(label):] if label and t.startswith(label) else t for t in texts]
    joined = pieces[0]
    for piece in pieces[1:]:
        overlap = next((n for n in range(min(len(joined), len(piece)), _MIN_OVERLAP - 1, -1)
                        if joined.endswith(piece[:n])), 0)
        rest = piece[overlap:]
        joined += rest if overlap else "\n" + rest
    return label + joined
