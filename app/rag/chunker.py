"""Chunking service.

Splits cleaned document text into overlapping chunks for embedding.
Uses LangChain's RecursiveCharacterTextSplitter, which tries to split
on structural boundaries first (paragraph, then line, then sentence,
then word) and only falls back to a hard character cut as a last
resort — so chunks stay closer to natural text boundaries than a
fixed-length split would.

`strategy="sentence"` packs whole sentences instead, following the
Advanced RAG field guide: a chunk never starts or ends mid-sentence, its
overlap is the previous chunk's last whole sentence(s), and a paragraph
that runs onto the next page stays in one chunk. The recursive splitter can
still cut a sentence in two when a paragraph is longer than a chunk,
leaving a fact's number in one chunk and its subject in the next.

Sizes are measured by a length function: characters by default, or an
embedding model's token count (see app.services.rag_service).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.rag.document_loader import DocumentPage


@dataclass
class Chunk:
    """A chunk of text ready for embedding, with traceable metadata."""

    text: str
    source: str
    page: int | None
    chunk_id: str
    section: str | None = None
    # Every page the chunk covers (a sentence-packed chunk can run onto the
    # next page); `page` is the first, used for citations.
    pages: list[int] = field(default_factory=list)
    # Read from a page image (app.rag.visual), added after the document's
    # text: a re-upload compares and replaces the two sets separately.
    visual: bool = False


# A sentence ends at . ! or ? (plus closing quotes/brackets) followed by
# whitespace and a capital, digit or opening quote; a blank line also ends one.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"”’)\]]*\s+(?=[\"“(\[]?[A-Z0-9])|\n\s*\n")
# Words whose period is not a sentence end ("U.S. Government", "e.g. the").
_ABBREVIATIONS = frozenset(
    "u.s e.g i.e etc vs mr mrs ms dr st jr sr inc co corp ltd fig figs approx est jan feb mar apr jun jul "
    "aug sep sept oct nov dec dept govt".split()
)
_LAST_WORD_RE = re.compile(r"([A-Za-z][A-Za-z.]*)\.[\"”’)\]]*$")


def split_sentences(text: str) -> list[str]:
    """`text` as whole sentences (internal line breaks kept), never splitting
    after an abbreviation or an initial ("Franklin D. Roosevelt")."""
    sentences, start = [], 0
    for match in _SENTENCE_END_RE.finditer(text):
        word = _LAST_WORD_RE.search(text[start:match.start()])
        if word:
            word, number_follows = word.group(1).lower(), text[match.end():match.end() + 1].isdigit()
            if word in _ABBREVIATIONS or len(word) == 1 or (word in {"no", "nos"} and number_follows):
                continue
        sentence = text[start:match.start()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def pack_sentences(sentences: list[tuple[str, int | None]], target: int, overlap: int,
                   length: Callable[[str], int] = len) -> list[list[tuple[str, int | None]]]:
    """The field guide's packing rule, over (sentence, page) pairs: a chunk
    closes when the next sentence would take it past `target` and it already
    holds more than `overlap`; the next chunk starts with the closed one's
    last sentences that fit in `overlap`. Sizes are in `length` units."""
    chunks, current, size = [], [], 0
    for sentence, page in sentences:
        cost = length(sentence)
        if current and size + cost > target and size > overlap:
            chunks.append(current)
            tail, tail_size = [], 0
            for previous in reversed(current):
                if tail_size + length(previous[0]) > overlap:
                    break
                tail.insert(0, previous)
                tail_size += length(previous[0])
            current, size = tail, tail_size
        current.append((sentence, page))
        size += cost
    if current:
        chunks.append(current)
    return chunks


def _pieces(pages: list[DocumentPage], chunk_size: int, chunk_overlap: int, strategy: str,
            splitter: RecursiveCharacterTextSplitter,
            length: Callable[[str], int]) -> list[tuple[DocumentPage, list[int], str]]:
    """(first page, every page covered, text) per chunk.

    "recursive" splits each page on its own. "sentence" packs whole sentences
    across consecutive pages of the same document and section. Visual
    elements (a chart, table or footnote passage) stay one chunk each.
    """
    if strategy != "sentence":
        return [(page, [page.page] if page.page is not None else [], piece)
                for page in pages for piece in splitter.split_text(page.text)]
    pieces: list[tuple[DocumentPage, list[int], str]] = []
    run: list[DocumentPage] = []

    def flush() -> None:
        sentences = []
        for page in run:
            for sentence in split_sentences(page.text):
                parts = splitter.split_text(sentence) if length(sentence) > chunk_size else [sentence]
                sentences.extend((part, page.page) for part in parts)
        by_number = {page.page: page for page in run}
        for packed in pack_sentences(sentences, chunk_size, chunk_overlap, length):
            covered = list(dict.fromkeys(page for _, page in packed if page is not None))
            first = by_number[covered[0]] if covered else run[0]
            pieces.append((first, covered, " ".join(sentence for sentence, _ in packed)))
        run.clear()

    for page in pages:
        if page.element:
            flush()
            pieces.append((page, [page.page] if page.page is not None else [], page.text))
            continue
        if run and (page.source, page.section) != (run[-1].source, run[-1].section):
            flush()
        run.append(page)
    flush()
    return pieces


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    strategy: str = settings.chunk_strategy,
    length_function: Callable[[str], int] = len,
) -> list[Chunk]:
    """Split a list of DocumentPage into overlapping Chunks.

    "recursive" splits each page independently; "sentence" packs whole
    sentences, across pages of one section (see `_pieces`). Chunks never mix
    sections, and pages with a detected section get its label prepended to
    every chunk. Sizes are measured by `length_function` (characters by
    default, or a token count). `chunk_id` is deterministic: the
    same page text at the same position always produces the same ID,
    which later lets us skip re-indexing unchanged documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=length_function,
    )

    chunks: list[Chunk] = []
    index_by_page: dict[tuple[str, int | None], int] = {}
    for page, covered, piece in _pieces(pages, chunk_size, chunk_overlap, strategy, splitter, length_function):
        # A section header on every chunk ("contextual chunk header") makes
        # the embedding — and the LLM — aware of where the text sits, which
        # keeps look-alike sections (e.g. two regions' outlooks) apart.
        text = f"[{page.section}]\n{piece}" if page.section else piece
        index = index_by_page.get((page.source, page.page), 0)
        index_by_page[(page.source, page.page)] = index + 1
        chunks.append(
            Chunk(
                text=text,
                source=page.source,
                page=page.page,
                chunk_id=_make_chunk_id(page.source, page.page, index, text),
                section=page.section,
                pages=covered,
            )
        )

    return chunks


def _make_chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    """Build a deterministic ID from source, page, position, and content."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    page_part = page if page is not None else "na"
    return f"{source}:{page_part}:{index}:{digest}"
