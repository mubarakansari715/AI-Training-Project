from app.rag.visual import footnote_pages, parse_endnotes, parse_visual_blocks, visual_pages


def test_parse_endnotes_reads_numbered_notes_after_the_heading():
    text = (
        "iMac  |  Product Environmental Report\n11\nEndnotes\n"
        "\t 1\tProduct recycled content is the mass of certified recycled material, \n"
        "not including packaging.  \n"
        "\t 2\tWe estimate low-carbon electricity.\n"
    )
    assert parse_endnotes(text) == {
        1: "Product recycled content is the mass of certified recycled material, not including packaging.",
        2: "We estimate low-carbon electricity.",
    }


def test_text_without_an_endnotes_heading_has_no_notes():
    assert parse_endnotes("1\tA numbered list item\n2\tAnother one") == {}


def test_each_referenced_footnote_becomes_its_own_passage_on_the_page():
    notes = {7: "Baseline is business as usual.", 18: "Measured under ENERGY STAR 8.0."}
    pages = footnote_pages({2: ["18", "7", "18", "99"], 3: []}, notes, "r.pdf")
    assert [(p.page, p.text) for p in pages] == [
        (2, "[Footnote 18, referenced on page 2] Measured under ENERGY STAR 8.0."),
        (2, "[Footnote 7, referenced on page 2] Baseline is business as usual."),
    ]


def test_parse_visual_blocks_splits_typed_elements():
    reply = (
        "TYPE: table\nTITLE: Greenhouse gas emissions\n| Item | Value |\n|---|---|\n| Product use | 45% |\n"
        "===\n"
        "TYPE: callout\nTITLE: Stand\n100% recycled aluminum in the stand\n"
    )
    assert parse_visual_blocks(reply) == [
        ("table", "Greenhouse gas emissions", "| Item | Value |\n|---|---|\n| Product use | 45% |"),
        ("callout", "Stand", "100% recycled aluminum in the stand"),
    ]
    assert parse_visual_blocks("NONE") == []


def test_visual_pages_become_labelled_document_pages():
    def describe(png: bytes, page: int) -> str:
        return "TYPE: chart\nTITLE: Carbon footprint\n346 kg CO2e total" if page == 2 else "NONE"

    pages = visual_pages([b"png1", b"png2"], "imac.pdf", describe)

    assert len(pages) == 1
    assert pages[0].page == 2 and pages[0].source == "imac.pdf"
    assert pages[0].text == "[Chart on page 2: Carbon footprint (read from the page image)]\n346 kg CO2e total"


def test_describer_falls_back_to_the_next_model_and_caches(tmp_path, monkeypatch):
    from app.rag import visual

    calls = []

    def ask(model, prompt, image_url):
        calls.append(model)
        if model == "busy-model":
            raise RuntimeError("429 rate-limited upstream")
        return "TYPE: table\nTITLE: T\n| a | b |"

    monkeypatch.setattr(visual, "_ask_vision_model", ask)
    describe = visual.openrouter_describer("r.pdf", models="busy-model, backup-model", cache_dir=str(tmp_path))

    assert describe(b"png", 1) == "TYPE: table\nTITLE: T\n| a | b |"
    assert describe(b"png", 1) == "TYPE: table\nTITLE: T\n| a | b |"  # second read comes from the cache
    assert calls == ["busy-model", "backup-model"]


def test_echoed_content_placeholders_are_dropped():
    reply = "TYPE: table\nTITLE: T\n<content>\n| a | b |\n</content>"
    assert parse_visual_blocks(reply) == [("table", "T", "| a | b |")]


def test_a_page_no_model_can_read_is_skipped():
    def describe(png: bytes, page: int) -> str:
        if page == 1:
            raise RuntimeError("all vision models busy")
        return "TYPE: callout\nTITLE: Stand\n100% recycled aluminum in the stand"

    pages = visual_pages([b"p1", b"p2"], "imac.pdf", describe)
    assert [p.page for p in pages] == [2]


def test_reading_stops_after_several_pages_fail():
    asked = []

    def describe(png: bytes, page: int) -> str:
        asked.append(page)
        if page % 2:  # flaky free models: every other page fails
            raise RuntimeError("429 rate-limited upstream")
        return "NONE"

    assert visual_pages([b"p"] * 10, "gep.pdf", describe, workers=1) == []
    assert asked == [1, 2, 3, 4, 5]


def test_pages_are_read_in_parallel_and_kept_in_page_order():
    import threading
    import time

    active, peak, lock = [0], [0], threading.Lock()

    def describe(png: bytes, page: int) -> str:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05 * (5 - page))  # later pages finish first
        with lock:
            active[0] -= 1
        return f"TYPE: table\nTITLE: T{page}\n| a |"

    pages = visual_pages([b"p"] * 4, "r.pdf", describe, workers=4)

    assert [p.page for p in pages] == [1, 2, 3, 4]
    assert peak[0] > 1


def test_a_short_pdf_skips_its_plain_text_pages():
    import pymupdf

    from app.rag import visual

    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "Prose only.", fontsize=10)
    table = doc.new_page()  # page 2: a drawn table grid
    for y in range(100, 220, 20):
        table.draw_line((50, y), (400, y))
    asked = []

    def describe(png: bytes, page: int) -> str:
        asked.append(page)
        return "NONE"

    visual.read_visuals(doc.tobytes(), "short.pdf", describe)

    assert asked == [2]


def test_a_long_pdf_reads_only_its_unmeasurable_chart_pages(monkeypatch):
    from dataclasses import replace

    import pymupdf

    from app.rag import visual
    from tests.test_chart_measure import _stacked_chart_page

    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "Prose only.", fontsize=10)
    doc.insert_pdf(_stacked_chart_page()[0])  # page 2: measured from its drawing, no vision call
    picture = doc.new_page()  # page 3: a chart pasted as a picture
    picture.insert_text((50, 60), "Figure 1.2 Prices", fontsize=10)
    picture.insert_image(pymupdf.Rect(50, 100, 400, 350), pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False))
    monkeypatch.setattr(visual, "settings", replace(visual.settings, visual_reading_max_pages=2, visual_chart_pages=40))
    asked = []

    def describe(png: bytes, page: int) -> str:
        asked.append(page)
        return "TYPE: chart\nTITLE: Prices\nOil about 80"

    pages = visual.read_visuals(doc.tobytes(), "long.pdf", describe)

    assert asked == [3] and [p.page for p in pages] == [3]
