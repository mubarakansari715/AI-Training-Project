from app.rag.chart_reader import answer_from_charts, chart_pages, is_chart_question, measurements
from app.rag.page_store import PageStore
from app.rag.retriever import RetrievedChunk
from tests.test_chart_measure import _stacked_chart_page


class FakeGenerator:
    def __init__(self, vision_reply: str = "", text_reply: str = "text answer"):
        self.vision_reply, self.text_reply = vision_reply, text_reply
        self.vision_prompts, self.text_prompts = [], []

    def describe_images(self, prompt, images, model_name=None):
        self.vision_prompts.append(prompt)
        return self.vision_reply

    def complete(self, prompt, **kwargs):
        self.text_prompts.append(prompt)
        return self.text_reply


def _store_with_chart(tmp_path) -> PageStore:
    """report.pdf: page 1 prose, page 2 the stacked-bar chart, page 3 a different chart."""
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((50, 60), "Introduction, no charts here.", fontsize=10)
    chart_doc, _ = _stacked_chart_page()
    doc.insert_pdf(chart_doc)
    other = doc.new_page()
    for i in range(25):  # enough shapes to count as a chart page, with no matching labels
        other.draw_rect(pymupdf.Rect(50 + 10 * i, 300, 55 + 10 * i, 400), color=None, fill=(0, 0, 0))
    store = PageStore(str(tmp_path))
    store.save("report.pdf", doc.tobytes())
    return store


def _chunk(page: int, text: str = "excerpt") -> RetrievedChunk:
    return RetrievedChunk(text=text, source="report.pdf", page=page, score=0.9, chunk_id=f"c{page}")


def test_chart_questions_are_recognised():
    assert is_chart_question("what is the moderate risk difference from the chart")
    assert not is_chart_question("who wrote the report")


def test_chart_page_carrying_the_questions_labels_comes_first(tmp_path):
    store = _store_with_chart(tmp_path)
    chunks = [_chunk(3), _chunk(1), _chunk(2)]  # retrieval ranked the other chart page first

    pages = chart_pages(store, "moderate risk in 2015 vs 2026 from the chart", chunks)

    assert pages == [("report.pdf", 2), ("report.pdf", 3)]


def test_the_vision_model_gets_the_measurements_and_pdf_page_numbers(tmp_path):
    store = _store_with_chart(tmp_path)
    generator = FakeGenerator(vision_reply="≈ 50 (measured from the chart, page 2)")

    answer, saw_images = answer_from_charts(generator, store, "moderate risk?", [_chunk(2)], [("report.pdf", 2)])

    assert answer.startswith("≈ 50") and saw_images and not generator.text_prompts
    prompt = generator.vision_prompts[0]
    assert "2015: High risk ≈ 30" in prompt and "measured from the chart, page <n>" in prompt
    assert "Pages: report.pdf page 2" in prompt and "not the" in prompt


def test_without_the_vision_model_the_measurements_are_answered_from(tmp_path):
    store = _store_with_chart(tmp_path)
    generator = FakeGenerator(vision_reply="")  # no Gemini key, or it failed

    answer, saw_images = answer_from_charts(generator, store, "moderate risk?", [_chunk(2)], [("report.pdf", 2)])

    assert answer == "text answer" and not saw_images
    assert "Moderate risk ≈ 50" in generator.text_prompts[0]
    assert "images aren't available" in generator.text_prompts[0]


def test_nothing_measured_and_no_vision_model_gives_no_chart_answer(tmp_path):
    store = _store_with_chart(tmp_path)
    generator = FakeGenerator(vision_reply="")

    assert measurements(store, [("report.pdf", 3)]) == ""
    assert answer_from_charts(generator, store, "q", [_chunk(3)], [("report.pdf", 3)]) == ("", False)
    assert not generator.text_prompts



def test_pictures_on_a_chart_page_are_also_sent_as_close_ups(tmp_path):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(50, 100, 300, 300), pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False))
    page.insert_image(pymupdf.Rect(400, 50, 410, 60), pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 5, 5), False))
    store = PageStore(str(tmp_path))
    store.save("pics.pdf", doc.tobytes())
    sent = []

    class Vision(FakeGenerator):
        def describe_images(self, prompt, images, model_name=None):
            sent.extend(images)
            return "≈ 96 (estimated from the chart image, page 1)"

    answer, _ = answer_from_charts(Vision(), store, "share?", [], [("pics.pdf", 1)])

    assert answer.startswith("≈ 96")
    assert len(sent) == 2  # the page, and a close-up of the big picture only (not the icon)
