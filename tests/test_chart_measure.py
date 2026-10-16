import pymupdf
import pytest

from app.rag.chart_measure import measure_page

NAVY, RED, ORANGE = (0.0, 0.14, 0.27), (0.92, 0.11, 0.18), (0.97, 0.55, 0.16)
ZERO_Y, POINTS_PER_UNIT = 400, 2  # the axis: value 0 at y=400, 100 at y=200


def _y(value: float) -> float:
    return ZERO_Y - value * POINTS_PER_UNIT


def _text_centred(page, x: float, y: float, text: str, size: float = 8) -> None:
    """Text whose vertical centre is at y (insert_text takes the baseline)."""
    page.insert_text((x, y + size * 0.35), text, fontsize=size)


def _axis(page, ticks, x_right: float = 70, scale: float = 1) -> None:
    """Tick labels, right-aligned; `scale` stretches them (ticks 0-10 drawn where 0-100 would be)."""
    for value in ticks:
        label = str(value)
        _text_centred(page, x_right - pymupdf.get_text_length(label, fontsize=8), _y(value * scale), label)


def _legend(page, entries, x: float = 90, y: float = 170) -> None:
    for i, (color, label) in enumerate(entries):
        page.draw_rect(pymupdf.Rect(x + 90 * i, y - 3, x + 90 * i + 6, y + 3), color=None, fill=color)
        _text_centred(page, x + 90 * i + 9, y, label)


def _bar(page, x0: float, x1: float, bottom: float, top: float, color, as_lines: bool = False) -> None:
    if as_lines:  # some exporters draw a bar as a closed path of lines
        page.draw_polyline([(x0, _y(top)), (x1, _y(top)), (x1, _y(bottom)), (x0, _y(bottom))],
                           color=None, fill=color, closePath=True)
    else:
        page.draw_rect(pymupdf.Rect(x0, _y(top), x1, _y(bottom)), color=None, fill=color)


def _stacked_chart_page():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 140), "D. Risk of debt distress", fontsize=9)
    _text_centred(page, 55, 185, "Percent")
    _axis(page, [100, 80, 60, 40, 20, 0])
    _legend(page, [(NAVY, "High risk"), (RED, "Moderate risk"), (ORANGE, "Low risk")])
    for (x0, x1), segments, lines in [((100, 140), (30, 50, 20), False), ((200, 240), (45, 35, 20), True)]:
        bottom = 0
        for value, color in zip(segments, (NAVY, RED, ORANGE)):
            _bar(page, x0, x1, bottom, bottom + value, color, as_lines=lines)
            bottom += value
    _text_centred(page, 108, ZERO_Y + 10, "2015")
    _text_centred(page, 208, ZERO_Y + 10, "2026")
    return doc, page


def test_stacked_bars_are_measured_with_their_series_and_categories():
    doc, page = _stacked_chart_page()
    [chart] = measure_page(page)

    assert chart.title == "D. Risk of debt distress" and chart.unit == "Percent" and chart.stacked
    [first], [second] = [[bars] for bars in chart.columns]
    assert [b.category for b in first] == ["2015"] * 3 and [b.category for b in second] == ["2026"] * 3
    assert [b.series for b in first] == ["High risk", "Moderate risk", "Low risk"]
    assert [b.value for b in first] == pytest.approx([30, 50, 20], abs=1)
    # The 2026 bar is drawn as lines rather than rectangles: measured all the same.
    assert [b.value for b in second] == pytest.approx([45, 35, 20], abs=1)
    assert "- 2015: High risk ≈ 30" in chart.describe() and "(total ≈ 10" in chart.describe()


def test_a_page_reusing_colours_reads_each_charts_own_legend():
    doc, page = _stacked_chart_page()
    # A second, single-series chart far to the right, in a legend colour of the first.
    _axis(page, [10, 8, 6, 4, 2, 0], x_right=330, scale=10)
    for i, value in enumerate([4, 7]):
        page.draw_rect(pymupdf.Rect(350 + 40 * i, _y(value * 10), 375 + 40 * i, ZERO_Y), color=None, fill=NAVY)
        _text_centred(page, 352 + 40 * i, ZERO_Y + 10, f"Q{i + 1}")

    single = next(c for c in measure_page(page) if not c.stacked)
    assert [(bars[0].category, bars[0].series) for bars in single.columns] == [("Q1", ""), ("Q2", "")]
    assert [bars[0].value for bars in single.columns] == pytest.approx([4, 7], abs=0.2)


def test_negative_bars_hang_below_zero():
    doc = pymupdf.open()
    page = doc.new_page()
    _axis(page, [40, 20, 0, -20, -40])
    page.draw_rect(pymupdf.Rect(100, _y(0), 130, _y(-30)), color=None, fill=NAVY)
    page.draw_rect(pymupdf.Rect(160, _y(25), 190, _y(0)), color=None, fill=NAVY)
    _text_centred(page, 105, _y(-40) + 10, "Imports")
    _text_centred(page, 165, _y(-40) + 10, "Exports")

    [chart] = measure_page(page)
    assert [bars[0].value for bars in chart.columns] == pytest.approx([-30, 25], abs=1)


def test_bars_on_a_right_hand_axis_are_left_out():
    doc = pymupdf.open()
    page = doc.new_page()
    _axis(page, [100, 50, 0])
    _legend(page, [(NAVY, "Debt"), (RED, "Deficit (RHS)")])
    page.draw_rect(pymupdf.Rect(100, _y(60), 130, ZERO_Y), color=None, fill=NAVY)
    page.draw_rect(pymupdf.Rect(140, _y(20), 170, ZERO_Y), color=None, fill=RED)

    [chart] = measure_page(page)
    assert [b.series for bars in chart.columns for b in bars] == ["Debt"]


def test_what_cannot_be_measured_is_skipped():
    doc = pymupdf.open()
    page = doc.new_page()
    # A log axis (no even step) ...
    for value in [100, 10, 1, 0]:
        _text_centred(page, 60, {100: 200, 10: 260, 1: 320, 0: 380}[value], str(value))
    page.draw_rect(pymupdf.Rect(100, 250, 130, 380), color=None, fill=NAVY)
    # ... and a chart pasted as a picture: no axis text or shapes to read.
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    page.insert_image(pymupdf.Rect(300, 200, 500, 380), pixmap=pixmap)

    assert measure_page(page) == []
