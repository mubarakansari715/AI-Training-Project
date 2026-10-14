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
