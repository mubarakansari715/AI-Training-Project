"""Document-level orchestration.

Parses a batch of uploaded files into DocumentPage objects and cleans
their text, collecting a friendly error per file instead of letting
one bad file crash the whole upload.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.rag.document_loader import DocumentPage, DocumentParsingError, parse_document
from app.rag.progress import Progress, scaled
from app.rag.text_cleaner import clean_text

logger = logging.getLogger(__name__)


@dataclass
class UploadedDocument:
    """An uploaded file's name and bytes, with Streamlit UploadedFile's
    interface; it outlives the Streamlit run that received the file, and
    can be sent to a worker process (app.services.ingest_jobs)."""

    name: str
    data: bytes = field(repr=False)

    def getvalue(self) -> bytes:
        return self.data


def parse_uploaded_files(
    files: Iterable, detect_structure: bool = True, read_visuals: bool = False, structure_source: str | None = None,
    layout_tables: str | None = None, progress: Progress | None = None, defer_visuals: bool = False,
) -> tuple[list[DocumentPage], dict[str, str]]:
    """Parse and clean Streamlit-uploaded files.

    Returns (pages, errors) where `errors` maps filename to a
    user-facing error message for any file that failed to parse or
    contained no extractable text (before or after cleaning).
    """
    pages: list[DocumentPage] = []
    errors: dict[str, str] = {}

    files = list(files)
    for number, file in enumerate(files):
        try:
            file_pages = parse_document(file.name, file.getvalue(), detect_structure, read_visuals, structure_source,
                                        layout_tables,
                                        scaled(progress, number / len(files), (number + 1) / len(files)),
                                        defer_visuals)
        except DocumentParsingError as exc:
            logger.warning("Parsing failed for %s: %s", file.name, exc)
            errors[file.name] = str(exc)
            continue

        cleaned_pages = []
        for page in file_pages:
            page.text = clean_text(page.text)
            if page.text:
                cleaned_pages.append(page)

        if not cleaned_pages:
            errors[file.name] = "No extractable text was found in this document."
            continue

        pages.extend(cleaned_pages)

    return pages, errors
