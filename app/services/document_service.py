"""Document-level orchestration.

Parses a batch of uploaded files into DocumentPage objects and cleans
their text, collecting a friendly error per file instead of letting
one bad file crash the whole upload.
"""

from __future__ import annotations

import logging
from typing import Iterable

from app.rag.document_loader import DocumentPage, DocumentParsingError, parse_document
from app.rag.text_cleaner import clean_text

logger = logging.getLogger(__name__)


def parse_uploaded_files(files: Iterable) -> tuple[list[DocumentPage], dict[str, str]]:
    """Parse and clean Streamlit-uploaded files.

    Returns (pages, errors) where `errors` maps filename to a
    user-facing error message for any file that failed to parse or
    contained no extractable text (before or after cleaning).
    """
    pages: list[DocumentPage] = []
    errors: dict[str, str] = {}

    for file in files:
        try:
            file_pages = parse_document(file.name, file.getvalue())
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
