"""Chunking service.

Splits cleaned document text into overlapping chunks for embedding.
Uses LangChain's RecursiveCharacterTextSplitter, which tries to split
on structural boundaries first (paragraph, then line, then sentence,
then word) and only falls back to a hard character cut as a last
resort — so chunks stay closer to natural text boundaries than a
fixed-length split would.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    """Split a list of DocumentPage into overlapping Chunks.

    Each page is split independently, so chunks never blend text from
    different source pages together. `chunk_id` is deterministic: the
    same page text at the same position always produces the same ID,
    which later lets us skip re-indexing unchanged documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source=page.source,
                    page=page.page,
                    chunk_id=_make_chunk_id(page.source, page.page, index, piece),
                )
            )

    return chunks


def _make_chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    """Build a deterministic ID from source, page, position, and content."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    page_part = page if page is not None else "na"
    return f"{source}:{page_part}:{index}:{digest}"
