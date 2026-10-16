"""Measuring bar charts from a PDF's own drawing, instead of guessing from pixels.

Charts exported from Excel, PowerPoint, R or Python are stored in the PDF as
vector shapes: every bar is a rectangle with exact coordinates. Reading the
axis tick labels gives the scale (position -> value), the legend swatches
give each colour's series name, and the labels under the bars give the
categories, so each bar (or stacked segment) can be measured to within about
a percent of the axis range. No vision model manages that by eye: on GEP's
figure 3.1.D both ChatGPT and Gemini were off by 10-20 points.

Vertical bar, grouped-bar and stacked-bar charts on a linear axis only.
Anything that can't be measured confidently (charts pasted as pictures, line
and area charts, log axes, bars that don't sit on a baseline) is skipped,
and the chart reader falls back to the vision model's visual estimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER_RE = re.compile(r"^[−–-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$|^[−–-]?\d+(?:\.\d+)?%?$")
_PANEL_RE = re.compile(r"^[A-Z]\.\s")
_YEAR_RE = re.compile(r"(?:19|20)\d\d\w?")
_RHS_RE = re.compile(r"\bRHS\b|right(?:-hand)? (?:axis|scale)", re.IGNORECASE)
_TOUCH = 1.5  # points: a bar edge this close to a baseline or another bar sits on it
_LABEL_ZONE = 30  # points below the bars where category labels are looked for


@dataclass
class _Text:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    vertical: bool = False

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    color: tuple[float, float, float]

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class _Axis:
    ticks: list[tuple[float, float]]  # (y centre, value), top to bottom
    x0: float
    x1: float
    slope: float
    intercept: float

    @property
    def top(self) -> float:
        return self.ticks[0][0]

    @property
    def bottom(self) -> float:
        return self.ticks[-1][0]

    def value(self, y: float) -> float:
        return self.intercept + self.slope * y

    def y_of(self, value: float) -> float:
        return (value - self.intercept) / self.slope


@dataclass
class MeasuredBar:
    category: str
    series: str
    value: float


@dataclass
class MeasuredChart:
    title: str
    unit: str
    stacked: bool
    decimals: int = 1
    # One list per bar (or stack of segments, bottom to top), left to right.
    columns: list[list[MeasuredBar]] = field(default_factory=list)

    def describe(self) -> str:
        """The measurements as text for the answer model, one line per bar."""
        heading = self.title or "Bar chart"
        if self.unit:
            heading += f" (axis: {self.unit})"
        lines = [heading + (" — stacked bars, segments listed bottom to top:" if self.stacked else ":")]
        category, parts = None, []
        for bars in self.columns + [[]]:
            # Side-by-side bars of one category share a line.
            if parts and (not bars or bars[0].category != category):
                lines.append(f"- {category}: " + "; ".join(parts))
                parts = []
            if not bars:
                break
            category = bars[0].category
            stack = "; ".join(f"{b.series} ≈ {self._fmt(b.value)}" if b.series else f"≈ {self._fmt(b.value)}"
                              for b in bars)
            if len(bars) > 1:
                stack += f" (total ≈ {self._fmt(sum(b.value for b in bars))})"
            parts.append(stack)
        return "\n".join(lines)

    def _fmt(self, value: float) -> str:
        return f"{value:,.{self.decimals}f}"


def measure_page(page) -> list[MeasuredChart]:
    """Every vertical bar chart on a PyMuPDF page that can be measured."""
    texts = _texts(page)
    axes = _axes(texts)
    if not axes:
        return []
    rects = _rects(page)
    entries = _swatches(rects, texts)
    swatches = [r for r, _ in entries]
    by_axis: dict[int, list[_Rect]] = {}
    for r in rects:
        if r in swatches or _light(r.color):
            continue
        axis = _axis_for(r, axes, texts)
        if axis is not None:
            by_axis.setdefault(axis, []).append(r)
    charts = []
    for index, bars in by_axis.items():
        axis = axes[index]
        # A page can hold several charts reusing the same colours, so each
        # chart reads only the legend drawn around it.
        right = max(r.x1 for r in bars) + 60
        local = [(r, label) for r, label in entries
                 if axis.x0 - 10 <= r.x0 <= right and axis.top - 45 <= r.y0 <= axis.bottom + 50]
        legend: dict[tuple, str] = {}
        for r, label in local:
            legend.setdefault(r.color, label)
        # A legend entry drawn as a thin dash stands for markers or a line
        # ("January 2026 forecast"), not bars.
        markers = {r.color for r, _ in local if r.y1 - r.y0 <= 2 and r.x1 - r.x0 >= 2.5 * (r.y1 - r.y0)}
        bars = [r for r in bars if r.color not in markers]
        if not bars:
            continue
        chart = _measure(axis, bars, texts, legend, [r for r, _ in local])
        if chart is not None:
            charts.append(chart)
    return charts


def measure_pdf(data: bytes) -> list[tuple[int, MeasuredChart]]:
    """(page number, chart) for every measurable bar chart in a PDF."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as document:
        return [(page.number + 1, chart) for page in document for chart in measure_page(page)]


def _number(text: str) -> float | None:
    text = text.strip().replace("−", "-").replace("–", "-")
    if not _NUMBER_RE.match(text):
        return None
    return float(text.rstrip("%").replace(",", ""))


def _texts(page) -> list[_Text]:
    texts = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                texts.append(_Text(text, *line["bbox"], vertical=abs(line["dir"][0]) < 0.5))
    return texts


def _rects(page) -> list[_Rect]:
    """Filled, axis-aligned rectangles: each `re` item, and each closed path
    made only of horizontal and vertical lines (some exporters draw bars that
    way). A path with a slanted edge or a curve (area or pie shapes) is not one."""
    rects = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if not fill or len(fill) != 3:
            continue
        color = tuple(round(c, 3) for c in fill)
        path: list[tuple[float, float]] = []
        broken = False

        def close_path():
            nonlocal broken
            if len(path) >= 3 and not broken:
                xs, ys = [p[0] for p in path], [p[1] for p in path]
                rects.append(_Rect(min(xs), min(ys), max(xs), max(ys), color))
            path.clear()
            broken = False

        for item in drawing["items"]:
            kind = item[0]
            if kind == "l":
                start, end = item[1], item[2]
                if path and (abs(path[-1][0] - start.x) > 0.1 or abs(path[-1][1] - start.y) > 0.1):
                    close_path()
                if not path:
                    path.append((start.x, start.y))
                path.append((end.x, end.y))
                if abs(start.x - end.x) > 0.1 and abs(start.y - end.y) > 0.1:
                    broken = True
            elif kind == "c":
                if not path:
                    path.append((item[1].x, item[1].y))
                path.append((item[-1].x, item[-1].y))
                broken = True
            else:
                close_path()
                shape = item[1].rect if kind == "qu" and item[1].is_rectangular else item[1] if kind == "re" else None
                if shape is not None:
                    rects.append(_Rect(shape.x0, shape.y0, shape.x1, shape.y1, color))
        close_path()
    return [r for r in rects if r.x1 - r.x0 >= 0.8 and r.y1 - r.y0 >= 0.3]


def _axes(texts: list[_Text]) -> list[_Axis]:
    """Vertical value axes: three or more numeric labels stacked in a column
    (same right edge), evenly spaced and with an even step, e.g. 0 20 40 ... 100.
    A log axis (0 1 10 100) has no even step, so it isn't one."""
    columns: dict[int, list[tuple[_Text, float]]] = {}
    for t in texts:
        v = _number(t.text)
        if v is not None and not t.vertical and t.x1 - t.x0 < 40:
            columns.setdefault(round(t.x1 / 2), []).append((t, v))
    axes = []
    for column in columns.values():
        column.sort(key=lambda tv: tv[0].cy)
        chain: list[tuple[_Text, float]] = []
        for t, v in column + [(None, None)]:
            if t is not None and _extends(chain, t, v):
                chain.append((t, v))
                continue
            if len(chain) >= 3:
                axes.append(_fit(chain))
            chain = [(t, v)] if t is not None else []
    return [a for a in axes if a.slope < 0]


def _extends(chain: list[tuple[_Text, float]], t: _Text, v: float) -> bool:
    if not chain:
        return True
    last, last_v = chain[-1]
    gap = t.cy - last.cy
    if gap < 3 or gap > 150 or v == last_v:
        return False
    if len(chain) == 1:
        return True
    prev, prev_v = chain[-2]
    prev_gap = last.cy - prev.cy
    same_step = abs((v - last_v) - (last_v - prev_v)) <= 1e-6 * max(1.0, abs(v))
    return abs(gap - prev_gap) <= 0.25 * prev_gap and same_step


def _fit(chain: list[tuple[_Text, float]]) -> _Axis:
    ys = [t.cy for t, _ in chain]
    vs = [v for _, v in chain]
    n = len(ys)
    my, mv = sum(ys) / n, sum(vs) / n
    slope = sum((y - my) * (v - mv) for y, v in zip(ys, vs)) / sum((y - my) ** 2 for y in ys)
    return _Axis(ticks=list(zip(ys, vs)), x0=min(t.x0 for t, _ in chain), x1=max(t.x1 for t, _ in chain),
                 slope=slope, intercept=mv - slope * my)


def _swatches(rects: list[_Rect], texts: list[_Text]) -> list[tuple[_Rect, str]]:
    """Legend entries: small swatches with a label just to their right."""
    entries = []
    for r in rects:
        if r.x1 - r.x0 > 14 or r.y1 - r.y0 > 14:
            continue
        cy = (r.y0 + r.y1) / 2
        labels = [t for t in texts if 0 <= t.x0 - r.x1 <= 8 and t.y0 - 2 <= cy <= t.y1 + 2]
        if labels:
            label = min(labels, key=lambda t: t.x0 - r.x1)
            if _number(label.text) is None:
                entries.append((r, label.text))
    return entries


def _light(color: tuple[float, float, float]) -> bool:
    return min(color) > 0.85


def _axis_for(r: _Rect, axes: list[_Axis], texts: list[_Text]) -> int | None:
    """The nearest axis to the bar's left whose range covers the bar, unless
    another column of numbers (an axis that couldn't be read, e.g. a log
    scale) stands between them."""
    best, best_distance = None, None
    for index, axis in enumerate(axes):
        pad = 0.1 * (axis.bottom - axis.top)
        if r.y0 < axis.top - pad or r.y1 > axis.bottom + pad or r.x0 < axis.x1 - 1:
            continue
        distance = r.x0 - axis.x1
        if best_distance is None or distance < best_distance:
            best, best_distance = index, distance
    if best is not None:
        axis = axes[best]
        between = [t for t in texts if axis.x1 + 2 < t.x0 and t.x1 < r.x0 and not t.vertical
                   and _number(t.text) is not None and axis.top - 10 <= t.cy <= axis.bottom + 10]
        edges = [round(t.x1) for t in between]
        if any(edges.count(e) >= 2 for e in edges):
            return None
    return best


def _measure(axis: _Axis, rects: list[_Rect], texts: list[_Text], legend: dict, swatches: list[_Rect]):
    values = [v for _, v in axis.ticks]
    zero_based = min(values) <= 0 <= max(values)
    if zero_based:
        baseline = axis.y_of(0)
    else:
        bottoms = [round(r.y1) for r in rects]
        baseline = max(set(bottoms), key=bottoms.count)
        if abs(baseline - axis.bottom) > 4:  # bars must rise from the axis's lowest tick
            return None
    # Bars sitting on the baseline, then stacked segments sitting on them.
    kept = [r for r in rects if abs(r.y1 - baseline) <= _TOUCH or abs(r.y0 - baseline) <= _TOUCH]
    rest = [r for r in rects if r not in kept]
    grew = True
    while grew:
        grew = False
        for r in list(rest):
            if any(_stacked_on(r, k) for k in kept):
                kept.append(r)
                rest.remove(r)
                grew = True
    if not kept:
        return None
    stacked = any(_stacked_on(a, b) for a in kept for b in kept if a is not b)
    span = max(values) - min(values)
    chart = MeasuredChart(title=_title(axis, kept, texts, swatches), unit=_unit(axis, texts, legend),
                          stacked=stacked, decimals=0 if span >= 1000 else 1 if span >= 10 else 2)
    columns: dict[float, list[_Rect]] = {}
    for r in sorted(kept, key=lambda r: r.cx):
        key = next((k for k in columns if abs(k - r.cx) <= 1), r.cx)
        columns.setdefault(key, []).append(r)
    rows = _label_rows(texts, axis, max(max(r.y1 for r in kept), baseline), list(columns))
    for number, (cx, column) in enumerate(sorted(columns.items()), start=1):
        category = _category(cx, max(r.x1 - r.x0 for r in column), rows) or f"bar {number}"
        if _RHS_RE.search(category):
            continue  # plotted against the right-hand axis, which isn't read
        bars = []
        for r in sorted(column, key=lambda r: -r.y1):  # bottom to top
            if not zero_based and abs(r.y1 - baseline) <= _TOUCH:
                value = axis.value(r.y0)  # a bar rising from an axis that doesn't start at 0
            elif (r.y0 + r.y1) / 2 < baseline:
                value = axis.value(r.y0) - axis.value(r.y1)
            else:
                value = axis.value(r.y1) - axis.value(r.y0)  # below zero: negative
            series = legend.get(r.color, "")
            if not _RHS_RE.search(series):
                bars.append(MeasuredBar(category=category, series=series, value=value))
        if bars:
            chart.columns.append(bars)
    if not chart.columns:
        return None
    return chart


def _stacked_on(a: _Rect, b: _Rect) -> bool:
    """`a` sits directly on top of (or below) `b`, sharing its horizontal span."""
    same_column = abs(a.x0 - b.x0) <= 1 and abs(a.x1 - b.x1) <= 1
    return same_column and (abs(a.y1 - b.y0) <= _TOUCH or abs(a.y0 - b.y1) <= _TOUCH)


def _label_rows(texts: list[_Text], axis: _Axis, bottom: float, centres: list[float]) -> list[list[_Text]]:
    """Category labels under the bars: the first row ("2025 2026 2027"), and
    a second row of group names under it ("Advanced economies", "EMDEs")."""
    labels = _merge_lines([t for t in texts if bottom - 2 <= t.y0 <= bottom + 3 * _LABEL_ZONE
                           and t.x0 > axis.x1 - 2 and len(t.text) <= 40], centres)
    first = [t for t in labels if t.y0 <= bottom + _LABEL_ZONE]
    if not first:
        return []
    first_top = min(t.y0 for t in first)
    first = [t for t in first if t.y0 <= first_top + 4]
    first_bottom = max(t.y1 for t in first)
    second = [t for t in labels if first_bottom - 1 <= t.y0 <= first_bottom + 15 and t not in first
              and _number(t.text) is None]
    return [first, second] if second else [first]


def _merge_lines(labels: list[_Text], centres: list[float]) -> list[_Text]:
    """Join a label wrapped over two lines ("Non-investment" / "grade"), or
    over two rotated columns ("Energy" / "exporters" turned sideways). Rotated
    labels of neighbouring bars ("ROU" / "KAZ") sit just as close together,
    so rotated columns are joined only when they belong to the same bar."""
    def bar_of(t: _Text) -> int:
        return min(range(len(centres)), key=lambda i: abs(centres[i] - t.cx))

    merged: list[_Text] = []
    for t in sorted(labels, key=lambda t: (t.x0, t.y0) if t.vertical else (t.y0, t.x0)):
        if t.vertical:
            near = next((m for m in merged if m.vertical and 0 <= t.x0 - m.x1 <= 3 and t.y0 < m.y1
                         and t.y1 > m.y0 and bar_of(m) == bar_of(t)), None)
        else:
            near = next((m for m in merged if not m.vertical and 0 <= t.y0 - m.y1 <= 3
                         and m.x0 - 2 <= t.cx <= m.x1 + 2), None)
        if near is None:
            merged.append(_Text(t.text, t.x0, t.y0, t.x1, t.y1, t.vertical))
        else:
            near.text += t.text if near.text.endswith("-") else " " + t.text
            near.x0, near.y0 = min(near.x0, t.x0), min(near.y0, t.y0)
            near.x1, near.y1 = max(near.x1, t.x1), max(near.y1, t.y1)
    return merged


def _category(cx: float, width: float, rows: list[list[_Text]]) -> str:
    """The label under the bar (or its group of side-by-side bars), with the
    group name under it when there is one: "2026 (EMDEs)". A number on a
    numeric x-axis (20 40 60) names only the bar right above it."""
    names = []
    for row, reach in zip(rows, (40, 80)):
        near = [t for t in row if abs(t.cx - cx) <= (reach if _number(t.text) is None else width / 2 + 1)]
        if near:
            names.append(min(near, key=lambda t: abs(t.cx - cx)).text)
    if len(names) == 2:
        return f"{names[0]} ({names[1]})"
    return names[0] if names else ""


def _title(axis: _Axis, bars: list[_Rect], texts: list[_Text], swatches: list[_Rect]) -> str:
    """The panel title above the chart, e.g. "D. Risk of debt distress in ..."."""
    top = min([axis.top] + [s.y0 for s in swatches])
    left, right = axis.x0 - 10, max(r.x1 for r in bars)
    above = sorted((t for t in texts if top - 60 <= t.y1 <= top and t.x1 > left and t.x0 < right),
                   key=lambda t: t.y0)
    for i, t in enumerate(above):
        if _PANEL_RE.match(t.text):
            title = t.text
            for nxt in above[i + 1:]:
                if abs(nxt.x0 - t.x0) <= 2 and 0 <= nxt.y0 - t.y1 <= 4:
                    title += " " + nxt.text
                    t = nxt
            return title
    return ""


def _unit(axis: _Axis, texts: list[_Text], legend: dict) -> str:
    """The axis caption just above the top tick, e.g. "Percent"."""
    near = [t for t in texts if 0 <= axis.top - t.y1 <= 16 and abs(t.x0 - axis.x0) <= 10
            and _number(t.text) is None and t.text not in legend.values()]
    return min(near, key=lambda t: axis.top - t.y1).text if near else ""
