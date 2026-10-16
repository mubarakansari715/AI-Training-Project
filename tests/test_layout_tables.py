from app.rag.layout_tables import read_layout_tables


def _pdf_with_table() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Power is measured at three voltages, as the table shows.", fontsize=10)
    dark = (0.2, 0.2, 0.2)
    for x0, x1 in [(50, 200), (200, 300), (300, 400)]:  # header bar, one cell per column
        page.draw_rect(pymupdf.Rect(x0, 100, x1, 118), color=dark, fill=dark)
    for x, text in [(55, "Mode"), (205, "100V"), (305, "230V")]:
        page.insert_text((x, 113), text, fontsize=9, color=(1, 1, 1))
    for i, (mode, low, high) in enumerate([("Off", "0.27W", "0.29W"), ("Sleep", "0.56W", "0.59W")]):
        y = 135 + i * 18
        page.draw_line((50, y + 5), (400, y + 5), color=(0.8, 0.8, 0.8))  # row rule
        for x, text in [(55, mode), (205, low), (305, high)]:
            page.insert_text((x, y), text, fontsize=9)
    page.insert_text((50, 300), "Unrelated closing paragraph far below the table.", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_header_bar_table_is_rebuilt_with_its_rows_and_columns():
    tables = read_layout_tables(_pdf_with_table(), "r.pdf")

    assert len(tables) == 1 and tables[0].element and tables[0].page == 1
    assert tables[0].text == ("[Table (from the page layout)]\n"
                              "| Mode | 100V | 230V |\n|---|---|---|\n"
                              "| Off | 0.27W | 0.29W |\n| Sleep | 0.56W | 0.59W |")


def test_pages_without_header_bars_have_no_tables():
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "Just prose, 2018 figures 802,834.", fontsize=10)
    assert read_layout_tables(doc.tobytes(), "r.pdf") == []
