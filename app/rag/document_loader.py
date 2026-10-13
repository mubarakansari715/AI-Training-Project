"""Document parsing.

Converts raw file bytes (PDF, DOCX, TXT, Markdown, and video/audio via
transcription) into a common list of DocumentPage objects, regardless
of source format.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import docx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

# Media types are transcribed by app.rag.media_transcriber (imported
# lazily, since it depends on this module for DocumentPage).
MEDIA_EXTENSIONS = {"mp4", "mov", "m4v", "webm", "mkv", "avi", "mp3", "wav", "m4a", "ogg", "flac"}
SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt", "md"} | MEDIA_EXTENSIONS


class DocumentParsingError(Exception):
    """Raised when a document cannot be parsed into text."""


@dataclass
class DocumentPage:
    """A unit of extracted text with its originating metadata."""

    text: str
    source: str
    page: int | None = None


def parse_document(filename: str, data: bytes) -> list[DocumentPage]:
    """Parse raw file bytes into a list of DocumentPage.

    Each PDF page becomes its own DocumentPage so page numbers are
    preserved. DOCX/TXT/Markdown have no reliable page concept, so a
    single DocumentPage is returned with `page=None`. Video/audio files
    are transcribed into an info page plus [mm:ss]-timestamped transcript
    pages (see app.rag.media_transcriber).

    Returns an empty list if the file is readable but contains no
    extractable text. Raises DocumentParsingError for empty files,
    unsupported types, or files that cannot be read at all.
    """
    if not data:
        raise DocumentParsingError(f"'{filename}' is empty.")

    extension = Path(filename).suffix.lower().lstrip(".")

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(
            f"Unsupported file type '.{extension}' for '{filename}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    try:
        if extension == "pdf":
            return _parse_pdf(data, filename)
        if extension == "docx":
            return _parse_docx(data, filename)
        if extension in MEDIA_EXTENSIONS:
            from app.rag.media_transcriber import transcribe_media

            return transcribe_media(data, filename)
        return _parse_text(data, filename)  # txt, md
    except DocumentParsingError:
        raise
    except Exception as exc:
        logger.exception("Failed to parse %s", filename)
        raise DocumentParsingError(
            f"Could not read '{filename}'. The file may be corrupted."
        ) from exc


def _parse_pdf(data: bytes, filename: str) -> list[DocumentPage]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise DocumentParsingError(
            f"Could not read '{filename}'. The PDF may be corrupted."
        ) from exc

    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(DocumentPage(text=text, source=filename, page=index))
    return pages


def _parse_docx(data: bytes, filename: str) -> list[DocumentPage]:
    document = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [DocumentPage(text=text, source=filename, page=None)]


def _parse_text(data: bytes, filename: str) -> list[DocumentPage]:
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    return [DocumentPage(text=text, source=filename, page=None)]
