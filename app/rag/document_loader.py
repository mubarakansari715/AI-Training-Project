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

from app.config.settings import settings
from app.rag.progress import Progress, report, scaled
from app.rag.structure import detect_sections, flatten_outline, font_outline
from app.rag.text_cleaner import expand_ligatures

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
    # Chapter/section path for structured documents, e.g.
    # "Ch. 2 Regional Outlooks › South Asia" (see app.rag.structure).
    section: str | None = None
    # A self-contained element (a chart, table or footnote passage from
    # app.rag.visual): chunked whole, never merged with neighbouring text.
    element: bool = False


def parse_document(
    filename: str, data: bytes, detect_structure: bool = True, read_visuals: bool = False,
    structure_source: str | None = None, layout_tables: str | None = None, progress: Progress | None = None,
) -> list[DocumentPage]:
    """Parse raw file bytes into a list of DocumentPage.

    Each PDF page becomes its own DocumentPage so page numbers are
    preserved. DOCX/TXT/Markdown have no reliable page concept, so a
    single DocumentPage is returned with `page=None`. Video/audio files
    are transcribed into an info page plus [mm:ss]-timestamped transcript
    pages (see app.rag.media_transcriber).

    `read_visuals` also reads a short PDF's charts, tables, call-outs and
    footnotes (see app.rag.visual). `structure_source` picks where section
    headings come from (default: settings.structure_source), and
    `layout_tables` whether tables are rebuilt from the page layout
    (default: settings.layout_tables; see app.rag.layout_tables).
    `progress` hears about each stage of reading a PDF (app.rag.progress).

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
            return _parse_pdf(data, filename, detect_structure, read_visuals,
                              structure_source or settings.structure_source, layout_tables or settings.layout_tables,
                              progress)
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


def _parse_pdf(
    data: bytes, filename: str, detect_structure: bool = True, read_visuals: bool = False,
    structure_source: str = "auto", layout_tables: str = "auto", progress: Progress | None = None,
) -> list[DocumentPage]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise DocumentParsingError(
            f"Could not read '{filename}'. The PDF may be corrupted."
        ) from exc

    # Ligatures expanded before section detection too, so headings and section
    # labels read "Fiscal", not "ﬁscal" (app.rag.text_cleaner).
    texts = {}
    for index, page in enumerate(reader.pages, start=1):
        report(progress, f"Reading text (page {index} of {len(reader.pages)})", 0.4 * (index - 1) / len(reader.pages))
        texts[index] = expand_ligatures(page.extract_text() or "").strip()
    report(progress, "Finding chapters and sections", 0.4)
    # Chapter/section per page, for structured reports (bookmarks, running
    # headers); empty for documents without detectable structure.
    sections = detect_sections(texts, _outline(reader, data, structure_source)) if detect_structure else {}
    visuals = []
    if read_visuals:
        from app.rag.visual import read_visuals as read_page_visuals
        from app.rag.visual import vision_models

        visuals = read_page_visuals(data, filename, progress=scaled(progress, 0.45, 0.85))
    vision = read_visuals and len(reader.pages) <= settings.visual_reading_max_pages and bool(vision_models())
    if layout_tables == "on" or (layout_tables == "auto" and not vision):
        from app.rag.layout_tables import read_layout_tables

        visuals += read_layout_tables(data, filename)
    if settings.index_measured_charts:
        report(progress, "Measuring charts", 0.85)
        visuals += _measured_charts(data, filename, sections)
    report(progress, "Text read", 1.0)
    return [
        DocumentPage(text=text, source=filename, page=index, section=sections.get(index))
        for index, text in texts.items()
        if text
    ] + visuals


def _measured_charts(data: bytes, filename: str, sections: dict) -> list[DocumentPage]:
    """One passage per bar chart measured from the PDF's drawing
    (app.rag.chart_measure), so its values can be found by search."""
    from app.rag.chart_measure import measure_pdf

    try:
        charts = measure_pdf(data)
    except Exception:
        logger.warning("Measuring the charts of %s failed", filename, exc_info=True)
        return []
    return [DocumentPage(text=f"[Chart on page {page} (values measured from the drawing)]\n{chart.describe()}",
                         source=filename, page=page, section=sections.get(page), element=True)
            for page, chart in charts]


def _outline(reader, data: bytes, source: str) -> list[tuple[str, int]]:
    """Section headings as (title, page): the PDF's bookmarks, or headings
    found by font size ("fonts", or "auto" when there are no bookmarks)."""
    bookmarks = [] if source == "fonts" else flatten_outline(reader)
    return bookmarks if bookmarks or source == "bookmarks" else font_outline(data)


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
