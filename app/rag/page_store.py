"""Indexed PDFs kept on disk, so a page can be looked at again at answer time.

Text extraction only keeps a chart's labels; its values are bar heights and
line positions. To answer a question about a chart, app.rag.chart_reader
measures the page's bar charts from their drawing and renders the page for
a vision model, both of which need the original PDF. Files are stored under PAGE_STORE_DIR by document name (re-uploading a
document replaces its copy).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app.config.settings import settings

logger = logging.getLogger(__name__)

_RENDER_DPI = 130
# Charts pasted as pictures are often small on the page: each also gets a
# sharp close-up (with room for its title and axis labels), up to this many.
_CLOSEUP_DPI = 300
_MAX_CLOSEUPS = 4
_MIN_PICTURE_SHARE = 0.02  # of the page area: smaller images are logos and icons
# A page with this many vector drawings (bars, lines, shapes), any image,
# or a bar chart that can be measured is treated as holding a chart or diagram.
_VISUAL_DRAWINGS = 20


class PageStore:
    def __init__(self, directory: str | None = None):
        self.directory = Path(directory or settings.page_store_dir)

    def _path(self, source: str) -> Path:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        return self.directory / f"{digest}.pdf"

    def save(self, source: str, data: bytes) -> None:
        """Keep a PDF's bytes (other file types have no pages to look at)."""
        if not source.lower().endswith(".pdf"):
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path(source).write_bytes(data)
        except OSError:
            logger.warning("Could not store %s for chart reading", source, exc_info=True)

    def has(self, source: str) -> bool:
        return self._path(source).exists()

    def remove(self, source: str) -> None:
        self._path(source).unlink(missing_ok=True)

    def visual_pages(self, source: str, pages: list[int]) -> list[int]:
        """Those of `pages` (1-based) that hold a chart, diagram or image."""
        if not self.has(source):
            return []
        import pymupdf

        from app.rag.chart_measure import measure_page

        found = []
        with pymupdf.open(self._path(source)) as document:
            for number in pages:
                if 1 <= number <= document.page_count:
                    page = document[number - 1]
                    # A simple bar chart can have fewer drawings than that.
                    if page.get_images() or len(page.get_drawings()) >= _VISUAL_DRAWINGS or measure_page(page):
                        found.append(number)
        return found

    def pages_mentioning(self, source: str, text: str) -> list[int]:
        """Pages whose text contains `text` (case-insensitive), e.g. "Figure 3.1"."""
        if not self.has(source):
            return []
        import pymupdf

        needle = text.lower()
        with pymupdf.open(self._path(source)) as document:
            return [page.number + 1 for page in document if needle in page.get_text().lower()]

    def measure_charts(self, source: str, page: int) -> list:
        """The bar charts on a page measured from its drawing
        (app.rag.chart_measure), or [] when there are none or it fails."""
        if not self.has(source):
            return []
        import pymupdf

        from app.rag.chart_measure import measure_page

        try:
            with pymupdf.open(self._path(source)) as document:
                if not 1 <= page <= document.page_count:
                    return []
                return measure_page(document[page - 1])
        except Exception:
            logger.warning("Measuring the charts on %s page %d failed", source, page, exc_info=True)
            return []

    def closeups(self, source: str, page: int) -> list[bytes]:
        """Sharp PNG close-ups of the sizeable pictures on a page (charts
        pasted as images), largest first. A vision model misreads a small
        pasted chart on a whole-page render (GEP page 31: ≈63% for a bar at
        ≈96%) but reads its close-up."""
        if not self.has(source):
            return []
        import pymupdf

        with pymupdf.open(self._path(source)) as document:
            if not 1 <= page <= document.page_count:
                return []
            pdf_page = document[page - 1]
            area = abs(pdf_page.rect)
            boxes = [pymupdf.Rect(info["bbox"]) for info in pdf_page.get_image_info()]
            boxes = sorted((b for b in boxes if abs(b) >= _MIN_PICTURE_SHARE * area), key=lambda b: -abs(b))
            clips = [pymupdf.Rect(b.x0 - 5, b.y0 - 45, b.x1 + 5, b.y1 + 30) & pdf_page.rect
                     for b in boxes[:_MAX_CLOSEUPS]]
            return [pdf_page.get_pixmap(dpi=_CLOSEUP_DPI, clip=clip).tobytes("png") for clip in clips]

    def render(self, source: str, page: int) -> bytes | None:
        """One page as a PNG, or None when the document or page is unknown."""
        if not self.has(source):
            return None
        import pymupdf

        with pymupdf.open(self._path(source)) as document:
            if not 1 <= page <= document.page_count:
                return None
            return document[page - 1].get_pixmap(dpi=_RENDER_DPI).tobytes("png")
