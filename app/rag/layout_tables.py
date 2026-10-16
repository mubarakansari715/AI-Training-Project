"""Tables rebuilt from page layout, without a vision model.

Plain text extraction flattens a table into a run of words, so a value
loses its row and column ("0.27W 0.27W 0.29W" with no "Off" or "115V"
next to it). Report tables are usually drawn with a dark header bar whose
cells mark the columns, so (the field guide's method):

1. find the header bars: dark filled rectangles in the page drawing;
2. take the columns from the bar cells along the bars' bottom edge;
3. read the rows: words below the header, grouped into lines by height
   and into columns by position, until a large vertical gap;
4. write the table as Markdown, one passage per table.

Free and local (PyMuPDF), so it also works for PDFs too long for the
vision model (app.rag.visual), which reads tables from page images.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.document_loader import DocumentPage

_DARK = 0.5  # fill luminance below this = a header bar
_BAR_HEIGHT = (8, 40)  # points
_BAR_MIN_WIDTH = 30
_STACK_GAP = 4  # header bars closer than this belong to one header
_LINE_TOLERANCE = 4  # words whose centres are this close vertically share a line
_ROW_GAP = 30  # a vertical gap between text lines larger than this ends the table
_RULE_GAP = 60  # ... and between drawn row fills/rules (rows of bullets sit between two rules)


@dataclass
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float


def _luminance(fill) -> float:
    return sum(fill[:3]) / 3 if fill else 1.0


def header_bars(drawings: list[dict]) -> list[_Rect]:
    """Dark filled rectangles shaped like header cells, top to bottom."""
    bars = []
    for drawing in drawings:
        rect = drawing["rect"]
        if (_luminance(drawing.get("fill")) < _DARK and _BAR_HEIGHT[0] <= rect.height <= _BAR_HEIGHT[1]
                and rect.width >= _BAR_MIN_WIDTH):
            bars.append(_Rect(rect.x0, rect.y0, rect.x1, rect.y1))
    return sorted(bars, key=lambda r: (r.y0, r.x0))


def _headers(bars: list[_Rect]) -> list[list[_Rect]]:
    """Bars grouped into one header per table (bars that touch vertically)."""
    groups: list[list[_Rect]] = []
    for bar in bars:
        if groups and bar.y0 <= max(b.y1 for b in groups[-1]) + _STACK_GAP:
            groups[-1].append(bar)
        else:
            groups.append([bar])
    return groups


def _lines(words: list[tuple]) -> list[list[tuple]]:
    """Words grouped into lines by vertical centre, top to bottom, each left to right."""
    lines: list[list[tuple]] = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        centre = (word[1] + word[3]) / 2
        if lines and abs(centre - (lines[-1][0][1] + lines[-1][0][3]) / 2) <= _LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def _cells(line: list[tuple], columns: list[tuple[float, float]]) -> list[str]:
    cells = [[] for _ in columns]
    for word in line:
        centre = (word[0] + word[2]) / 2
        index = next((i for i, (x0, x1) in enumerate(columns) if x0 <= centre < x1),
                     0 if centre < columns[0][0] else len(columns) - 1)
        cells[index].append(word[4])
    return [" ".join(cell) for cell in cells]


def _body_bottom(drawings: list[dict], left: float, right: float, header_bottom: float, limit: float) -> float | None:
    """Where the table body ends: the last row fill or rule in a chain of
    drawings below the header (each within `_RULE_GAP` of the one before);
    None when the body has no drawn rows."""
    rects = sorted((d["rect"] for d in drawings
                    if d["rect"].x0 < right and d["rect"].x1 > left and d["rect"].width < (right - left) * 2
                    and header_bottom - 1 <= d["rect"].y0 < limit), key=lambda r: r.y0)
    bottom = None
    for rect in rects:
        if rect.y0 > (bottom or header_bottom) + _RULE_GAP:
            break
        bottom = max(bottom or header_bottom, rect.y1)
    return bottom


def page_tables(words: list[tuple], drawings: list[dict]) -> list[str]:
    """Markdown for each header-bar table on a page (PyMuPDF words/drawings).

    A header must name every column, which rules out charts and diagrams
    that also use dark bars. Bars above the bottom header row (a title
    spanning several columns) become a caption line above the table.
    """
    tables = []
    headers = _headers(header_bars(drawings))
    for number, bars in enumerate(headers):
        bottom = max(b.y1 for b in bars)
        top = min(b.y0 for b in bars)
        bottom_row = [b for b in bars if abs(b.y1 - bottom) <= _STACK_GAP]
        row_top = min(b.y0 for b in bottom_row if b.y1 - b.y0 < bottom - top) if any(
            b.y1 - b.y0 < bottom - top for b in bottom_row) else top
        columns = sorted({(b.x0, b.x1) for b in bottom_row})
        left, right = min(b.x0 for b in bars), max(b.x1 for b in bars)
        columns = [(left if i == 0 else x0, right if i == len(columns) - 1 else x1)
                   for i, (x0, x1) in enumerate(columns)]
        if len(columns) < 2:
            continue
        limit = min(b.y0 for b in headers[number + 1]) if number + 1 < len(headers) else float("inf")
        inside = [w for w in words if left - 2 <= (w[0] + w[2]) / 2 <= right + 2]

        def centre(word):
            return (word[1] + word[3]) / 2

        caption = " ".join(w[4] for line in _lines([w for w in inside if top - 2 <= centre(w) < row_top - 1])
                           for w in line if (w[0] + w[2]) / 2 >= columns[1][0])
        header_words = [w for w in inside if top - 2 <= centre(w) <= bottom + 2 and not (
            centre(w) < row_top - 1 and (w[0] + w[2]) / 2 >= columns[1][0])]
        header = [" ".join(c for c in column if c)
                  for column in zip(*[_cells(line, columns) for line in _lines(header_words)])]
        if len(header) != len(columns) or not all(header):
            continue
        body_bottom = _body_bottom(drawings, left, right, bottom, limit)
        rows, last = [], bottom
        for line in _lines([w for w in inside if bottom + 2 < centre(w) < limit]):
            if (body_bottom is not None and centre(line[0]) > body_bottom + _LINE_TOLERANCE) or \
                    (body_bottom is None and line[0][1] - last > _ROW_GAP):
                break
            rows.append(_cells(line, columns))
            last = max(w[3] for w in line)
        if not rows:
            continue
        markdown = ([caption] if caption else []) + ["| " + " | ".join(header) + " |", "|" + "---|" * len(columns)]
        markdown += ["| " + " | ".join(row) + " |" for row in rows]
        tables.append("\n".join(markdown))
    return tables


def read_layout_tables(data: bytes, filename: str) -> list[DocumentPage]:
    """One element passage per header-bar table in the PDF."""
    import pymupdf

    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as document:
        for number, page in enumerate(document, start=1):
            for table in page_tables(page.get_text("words"), page.get_drawings()):
                pages.append(DocumentPage(text=f"[Table (from the page layout)]\n{table}", source=filename,
                                          page=number, element=True))
    return pages
