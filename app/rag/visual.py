"""Layout-aware reading for short, visual PDFs (product reports, fact sheets).

Plain text extraction misses most of what such documents say: their facts
sit in charts (drawn as vector shapes, so there is no image to OCR), small
tables, icon call-outs, and numbered footnotes that limit a claim. Two
additions make those searchable:

- Page images: every page is rendered and a vision model transcribes its
  charts, tables and call-outs into text — tables as Markdown, charts with
  their printed values — one DocumentPage per element, so a table is never
  split from its title. Replies are cached on disk by image hash, so each
  page is paid for once.
- Footnotes: superscript note markers on a page are matched to the
  document's endnotes, and each referenced note becomes its own small
  passage on that page ("[Footnote 1, referenced on page 4] …"), so a
  question about a claim's condition finds the note itself — one block of
  every note on a page matched no single question well enough.

Each page read costs one vision call (about a minute on the free models),
so pages are read VISION_CONCURRENCY at a time, and only pages that draw
something are read: a page of plain text (no picture, almost no drawn
shapes) holds nothing the text extraction misses. That applies to documents
up to VISUAL_READING_MAX_PAGES (see app.config.settings). In a longer one,
only pages holding a chart that can't be measured from its drawing (a
picture; app.rag.chart_measure measures the rest for free) are read, up to
VISUAL_CHART_PAGES.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import time

import requests

from app.config.settings import settings
from app.rag.document_loader import DocumentPage
from app.rag.progress import Progress, report

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_TIMEOUT_SECONDS = 120
# Free vision models are rate-limited per minute: wait and retry a few times.
_RATE_LIMIT_ATTEMPTS = 3
_RATE_LIMIT_WAIT_SECONDS = 20.0
_RENDER_DPI = 110
# Free models that fail this many pages are rate-limited or down (each
# failure has already waited through every model's retries): stop rather
# than wait out the remaining pages.
_MAX_FAILED_PAGES = 3
_CHART_PAGE_DRAWINGS = 20
# A page with fewer drawn shapes than this (and no picture) is plain text:
# a header rule or two, no chart, table grid or icon.
_VISUAL_PAGE_DRAWINGS = 5
_MIN_IMAGE_SHARE = 0.02  # of the page area: smaller images are logos and icons
_CAPTION_RE = re.compile(r"\b(?:figure|chart|graph|exhibit)\s*[A-Z]?\d", re.IGNORECASE)
# Bump when VISION_PROMPT changes, so cached replies from the old prompt are not reused.
_PROMPT_VERSION = 1

VISION_PROMPT = """You are reading page {page} of the PDF "{filename}". Transcribe every chart,
table, infographic, icon call-out and highlighted figure on this page into text, so its
facts can be searched. Ignore ordinary body paragraphs (they are extracted separately),
page headers and footers, and purely decorative images.

For each element write one block:
TYPE: chart | table | callout | diagram
TITLE: <its title or a short description>
<content>

- table: a Markdown table with every row and column exactly as printed, then any note under it.
- chart: the chart type, then every printed value with its label and unit, and what the
  chart compares. Only report values printed on the page; never estimate bar sizes.
- callout: the exact text, with its number (e.g. "100% recycled aluminum in the stand").
- Keep footnote markers as [n] (e.g. "346 kg CO2e [10]").

Separate blocks with a line containing only ===. If the page has none of these elements,
reply with just NONE."""

_TYPE_LABELS = {"chart": "Chart", "table": "Table", "callout": "Call-out", "diagram": "Diagram"}
_ENDNOTES_HEADING_RE = re.compile(r"^\s*(endnotes|footnotes)\s*$", re.IGNORECASE | re.MULTILINE)
_NOTE_START_RE = re.compile(r"^\s*(\d{1,3})\t", re.MULTILINE)

# describe(png_bytes, page_number) -> the vision model's reply
Describer = Callable[[bytes, int], str]


# ----------------------------------------------------------------- footnotes


def parse_endnotes(text: str) -> dict[int, str]:
    """Numbered notes ("\\t 7\\tNote text…") after an Endnotes/Footnotes heading."""
    heading = _ENDNOTES_HEADING_RE.search(text)
    if not heading:
        return {}
    body = text[heading.end():]
    starts = list(_NOTE_START_RE.finditer(body))
    notes = {}
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        notes[int(match.group(1))] = " ".join(body[match.end():end].split())
    return notes


def superscript_markers(page) -> list[str]:
    """Superscript digits on a PyMuPDF page, in reading order (footnote markers)."""
    markers = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if span["flags"] & 1 and text.isdigit():  # flag bit 0 = superscript
                    markers.append(text)
    return markers


def footnote_label(number: int, page: int) -> str:
    """How a footnote passage starts; the retriever finds notes by it."""
    return f"[Footnote {number}, referenced on page {page}]"


def footnote_pages(markers_by_page: dict[int, list[str]], notes: dict[int, str], filename: str) -> list[DocumentPage]:
    """One DocumentPage per note a page refers to (each once per page)."""
    pages = []
    for page, markers in markers_by_page.items():
        for number in dict.fromkeys(int(m) for m in markers if int(m) in notes):
            pages.append(DocumentPage(
                text=f"{footnote_label(number, page)} {notes[number]}", source=filename, page=page, element=True,
            ))
    return pages


# -------------------------------------------------------------- page images


def parse_visual_blocks(reply: str) -> list[tuple[str, str, str]]:
    """Split a vision reply into (type, title, content) elements."""
    if not reply or reply.strip().upper() == "NONE":
        return []
    blocks = []
    for raw in re.split(r"^\s*===\s*$", reply, flags=re.MULTILINE):
        kind, title, content = "diagram", "", []
        for line in raw.strip().splitlines():
            if line.upper().startswith("TYPE:"):
                kind = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("TITLE:"):
                title = line.split(":", 1)[1].strip()
            elif line.strip().lower() in {"<content>", "</content>"}:  # some models echo the placeholder
                continue
            else:
                content.append(line)
        body = "\n".join(content).strip()
        if body:
            blocks.append((kind if kind in _TYPE_LABELS else "diagram", title, body))
    return blocks


def visual_pages(page_pngs: list[bytes], filename: str, describe: Describer,
                 numbers: list[int] | None = None, progress: Progress | None = None,
                 workers: int | None = None) -> list[DocumentPage]:
    """One DocumentPage per chart/table/call-out found on the rendered pages
    (page `numbers`, default 1, 2, ...), in page order. `workers` pages
    (default VISION_CONCURRENCY) are read at a time. A page no model could
    read is skipped (logged), keeping the others; after a few such pages the
    rest are skipped too."""
    numbers = list(numbers or range(1, len(page_pngs) + 1))
    todo = iter(zip(numbers, page_pngs))
    replies: dict[int, str] = {}
    failed = done = 0
    report(progress, f"Reading charts and tables (0 of {len(numbers)} pages)", 0.0)
    # Progress is reported from this thread only: Streamlit widgets can't be
    # updated from the worker threads.
    workers = max(1, workers or settings.vision_concurrency)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = {}

        def start_next() -> None:
            for number, png in todo:
                running[pool.submit(describe, png, number)] = number
                return

        for _ in range(workers):
            start_next()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                number = running.pop(future)
                done += 1
                try:
                    replies[number] = future.result()
                except Exception:
                    logger.exception("Could not read the image of page %d of %s", number, filename)
                    failed += 1
                if failed < _MAX_FAILED_PAGES:
                    start_next()
                report(progress, f"Reading charts and tables ({done} of {len(numbers)} pages)", done / len(numbers))
        if failed >= _MAX_FAILED_PAGES and done < len(numbers):
            logger.warning("Vision models failed on %d pages; not reading the rest of %s", failed, filename)

    pages = []
    for number in numbers:
        for kind, title, body in parse_visual_blocks(replies.get(number, "")):
            label = f"{_TYPE_LABELS[kind]} on page {number}" + (f": {title}" if title else "")
            pages.append(DocumentPage(text=f"[{label} (read from the page image)]\n{body}", source=filename, page=number,
                                      element=True))
    return pages


def openrouter_describer(filename: str, models: str | list[str] | None = None, cache_dir: str | None = None) -> Describer:
    """A Describer backed by OpenRouter vision models, with an on-disk cache.

    `models` (default VISION_MODEL, comma-separated) are tried in order for
    each page: free models are often rate-limited or briefly unavailable, so
    the next one takes over. A page any of them has read is never re-read.
    """
    models = models or settings.vision_model
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    cache = Path(cache_dir or settings.vision_cache_dir)

    def cache_file(png: bytes, model: str) -> Path:
        return cache / f"{hashlib.sha256(png + f'{model}|{_PROMPT_VERSION}'.encode()).hexdigest()[:32]}.txt"

    def describe(png: bytes, page: int) -> str:
        for model in models:
            if cache_file(png, model).exists():
                return cache_file(png, model).read_text(encoding="utf-8")
        image = "data:image/png;base64," + base64.b64encode(png).decode()
        errors = []
        for model in models:
            try:
                reply = _ask_vision_model(model, VISION_PROMPT.format(page=page, filename=filename), image)
            except Exception as exc:  # rate limit, provider error, malformed reply: try the next model
                logger.warning("Vision model %s failed on page %d: %s", model, page, exc)
                errors.append(f"{model}: {exc}")
                continue
            cache.mkdir(parents=True, exist_ok=True)
            cache_file(png, model).write_text(reply, encoding="utf-8")
            return reply
        raise RuntimeError(f"No vision model could read page {page}: {'; '.join(errors)}")

    return describe


def _ask_vision_model(model: str, prompt: str, image_url: str) -> str:
    """One vision request, retried while the model is rate-limited."""
    for attempt in range(1, _RATE_LIMIT_ATTEMPTS + 1):
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}],
            },
            timeout=VISION_TIMEOUT_SECONDS,
        )
        if response.status_code != 429 or attempt == _RATE_LIMIT_ATTEMPTS:
            break
        time.sleep(_RATE_LIMIT_WAIT_SECONDS)
    response.raise_for_status()
    body = response.json()
    if not body.get("choices"):
        raise RuntimeError(str(body.get("error") or body)[:200])
    return body["choices"][0]["message"]["content"] or ""


# -------------------------------------------------------------------- entry


def drawn_pages(doc) -> list[int]:
    """Pages (1-based) worth showing a vision model: a sizeable picture, or
    enough drawn shapes for a chart, table grid or icon call-out. The rest
    are plain text, which text extraction already reads."""
    import pymupdf

    found = []
    for page in doc:
        pictures = sum(abs(pymupdf.Rect(info["bbox"])) for info in page.get_image_info())
        if pictures >= _MIN_IMAGE_SHARE * abs(page.rect) or len(page.get_drawings()) >= _VISUAL_PAGE_DRAWINGS:
            found.append(page.number + 1)
    return found


def picture_chart_pages(doc) -> list[int]:
    """Pages (1-based) holding a chart or diagram that can't be measured
    from its drawing: a sizeable image (a pasted chart), or many drawn shapes
    under a figure caption where no bar chart could be measured (line charts,
    diagrams)."""
    import pymupdf

    from app.rag.chart_measure import measure_page

    found = []
    for page in doc:
        area = abs(page.rect)
        pictures = sum(abs(pymupdf.Rect(info["bbox"])) for info in page.get_image_info()) >= _MIN_IMAGE_SHARE * area
        drawn = len(page.get_drawings()) >= _CHART_PAGE_DRAWINGS and _CAPTION_RE.search(page.get_text())
        if pictures or (drawn and not measure_page(page)):
            found.append(page.number + 1)
    return found


def read_visuals(data: bytes, filename: str, describe: Describer | None = None,
                 progress: Progress | None = None) -> list[DocumentPage]:
    """Footnote and chart/table/call-out pages for a short PDF; for a long
    one, the chart pages that can't be measured from their drawing.

    Any failure (no vision key, provider error) degrades to the footnotes
    alone: it is logged and nothing visual is added.
    """
    import pymupdf  # PyMuPDF: page rendering and span-level font flags

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.page_count > settings.visual_reading_max_pages:
            numbers = picture_chart_pages(doc)[:settings.visual_chart_pages]
            pngs = [doc[n - 1].get_pixmap(dpi=_RENDER_DPI).tobytes("png") for n in numbers]
            if not numbers or (describe is None and not settings.openrouter_api_key):
                return []
            try:
                return visual_pages(pngs, filename, describe or openrouter_describer(filename), numbers, progress)
            except Exception:
                logger.exception("Reading the chart pages of %s failed; using text only", filename)
                return []
        notes_by_page = {page.number: parse_endnotes(page.get_text()) for page in doc}
        notes = {n: text for page_notes in notes_by_page.values() for n, text in page_notes.items()}
        markers_by_page = {
            page.number + 1: superscript_markers(page)
            for page in doc
            if not notes_by_page[page.number]  # the endnotes page itself
        }
        footnotes = footnote_pages(markers_by_page, notes, filename)
        numbers = drawn_pages(doc)
        pngs = [doc[n - 1].get_pixmap(dpi=_RENDER_DPI).tobytes("png") for n in numbers]

    if not numbers:
        return footnotes
    if describe is None:
        if not settings.openrouter_api_key:
            return footnotes
        describe = openrouter_describer(filename)
    try:
        return footnotes + visual_pages(pngs, filename, describe, numbers, progress)
    except Exception:
        logger.exception("Reading page images of %s failed; using text only", filename)
        return footnotes
