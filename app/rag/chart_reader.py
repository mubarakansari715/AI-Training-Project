"""Answering chart questions by looking at the chart, and measuring it when we can.

Many charts print no numbers: a stacked bar's shares or a line's levels exist
only as shapes, so text extraction (and the text-only answer model) sees the
axis ticks and labels and nothing else. For a question about a chart, the
best-matching retrieved pages that hold a chart are:

1. measured: bar charts drawn as vector shapes are read from the PDF's own
   drawing (app.rag.chart_measure), to within about 1% of the axis range;
2. rendered and shown, with the measurements, the question and the retrieved
   text, to a vision model (Gemini, free tier).

Every number in the answer says where it came from: printed values are quoted
exactly, measured ones read "≈ 51.7% (measured from the chart, page 110)",
and values the model could only judge by eye "≈ 30% (estimated from the chart
image, page 110)". When the vision model can't run, the measurements alone
are answered from by the text model; with neither, the normal text answer is
used.
"""

from __future__ import annotations

import re

from app.rag.page_store import PageStore
from app.rag.prompt import FORMAT_RULE
from app.rag.retriever import RetrievedChunk

MAX_PAGES = 2
# Retrieved pages looked at when choosing which chart pages to show.
_CANDIDATE_PAGES = 6

_CHART_WORDS_RE = re.compile(
    r"\b(charts?|graphs?|figures?|fig\.|plots?|diagrams?|bars?|axis|axes|candles?|candlesticks?|"
    r"infographics?|visuals?|from the (?:image|picture))\b",
    re.IGNORECASE,
)
_FIGURE_RE = re.compile(r"\b(?:figure|fig\.?|chart)\s*(\d+(?:\.\d+)*[a-z]?)\b", re.IGNORECASE)

_RULES = """Rules:
- Name each chart you use (its figure number, title or panel letter, and the page).
- If more than one chart matches the question, answer for each of them, one
  after the other, rather than picking one.
- A value printed on the chart or in the text: quote it exactly.
- MEASURED VALUES were measured from the chart's own drawing in the PDF (each
  bar's height against the axis scale), accurate to about 1% of the axis range.
  When a value you need is listed there, use it rather than judging the image,
  and write it as "≈ <value> (measured from the chart, page <n>)"; mark any
  difference or total computed from measured values the same way.
- {estimate_rule}
- If the chart can't answer the question (the series isn't shown or can't be
  read), say so plainly instead of guessing.
- Explain a change only with what the chart and excerpts show (e.g. which other
  series grew as this one shrank); don't add your own judgement such as calling
  it an improvement or a deterioration.
- Cite pages in plain text, e.g. "(report.pdf, page 110)", using the page
  numbers listed under "Pages" below: those are the PDF's page numbers, not the
  page number printed on the page itself.
- {format_rule}"""

_ESTIMATE_RULE = """A value that isn't printed or measured, read from a bar's height, a line's
  position, a segment's size, a candle's body or wick, or a position against
  the axis ticks, is an ESTIMATE: always write it as "≈ <value> (estimated
  from the chart image, page <n>)", and mark any difference or total computed
  from estimates the same way. Never present an estimate as an exact figure."""

_NO_IMAGE_RULE = """The chart images aren't available here, so use only printed and
  measured values; if those don't answer the question, say so."""

CHART_PROMPT = """You are answering a question about the user's document. {intro}

{rules}

Pages: {pages}

MEASURED VALUES:
{measured}

Text excerpts:
{excerpts}

Question: {question}"""

_IMAGE_INTRO = ("The images are pages of that document holding the chart(s) the question is about, "
                "each followed by close-ups of the charts pasted on it as pictures (read values from "
                "the close-ups); the text excerpts below were found by searching the document.")
_TEXT_INTRO = ("The pages below hold the chart(s) the question is about; the text excerpts were found "
               "by searching the document.")


def is_chart_question(question: str) -> bool:
    return bool(_CHART_WORDS_RE.search(question))


def chart_pages(store: PageStore, question: str, chunks: list[RetrievedChunk]) -> list[tuple[str, int]]:
    """Up to MAX_PAGES (document, page) with a chart to look at: a figure the
    question names ("Figure 3.1") first, then retrieved chart pages, those
    whose measured charts carry the question's labels ("moderate risk",
    "2015") ahead of the rest, else in retrieval order."""
    sources = list(dict.fromkeys(c.source for c in chunks))
    picked: list[tuple[str, int]] = []
    for name in _FIGURE_RE.findall(question):
        for source in sources:
            for page in store.visual_pages(source, store.pages_mentioning(source, f"figure {name}")):
                if (source, page) not in picked:
                    picked.append((source, page))
    ranked = list(dict.fromkeys((c.source, c.page) for c in chunks if c.page is not None))[:_CANDIDATE_PAGES]
    charted = [(source, page) for source, page in ranked
               if (source, page) not in picked and store.visual_pages(source, [page])]
    scores = {key: _label_matches(question, store.measure_charts(*key)) for key in charted}
    charted.sort(key=lambda key: -scores[key])  # stable: retrieval order breaks ties
    return (picked + charted)[:MAX_PAGES]


def _label_matches(question: str, charts) -> int:
    """How many of the charts' series and category names the question mentions."""
    asked = question.lower()
    labels = {b.series.lower() for c in charts for bars in c.columns for b in bars if len(b.series) >= 3}
    labels |= {b.category.lower() for c in charts for bars in c.columns for b in bars if len(b.category) >= 3}
    return sum(1 for label in labels if label in asked)


def measurements(store: PageStore, pages: list[tuple[str, int]]) -> str:
    """The measured bar charts on `pages`, as text for the prompt."""
    blocks = []
    for source, page in pages:
        for chart in store.measure_charts(source, page):
            blocks.append(f"({source}, page {page}) {chart.describe()}")
    return "\n\n".join(blocks)


def _excerpts(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.source}, page {c.page})\n{c.text}" for i, c in enumerate(chunks, start=1)) \
        or "(none)"


def build_chart_prompt(question: str, chunks: list[RetrievedChunk], pages: list[tuple[str, int]],
                       measured: str, with_images: bool) -> str:
    rules = _RULES.format(estimate_rule=_ESTIMATE_RULE if with_images else _NO_IMAGE_RULE,
                          format_rule=" ".join(FORMAT_RULE.split()))
    return CHART_PROMPT.format(
        intro=_IMAGE_INTRO if with_images else _TEXT_INTRO,
        rules=rules,
        pages=", ".join(f"{source} page {page}" for source, page in pages),
        measured=measured or "(none: no chart on these pages could be measured)",
        excerpts=_excerpts(chunks),
        question=question,
    )


def answer_from_charts(generator, store: PageStore, question: str, chunks: list[RetrievedChunk],
                       pages: list[tuple[str, int]], model_name: str | None = None,
                       groq_model_name: str | None = None) -> tuple[str, bool]:
    """The answer read from the chart pages, and whether the page images were
    seen: the vision model's answer, with the measurements; else the text
    model's from the measurements alone; else "" (nothing measured and no
    vision model, page image or provider)."""
    measured = measurements(store, pages)
    images = []
    for source, page in pages:
        png = store.render(source, page)
        if png:
            images += [png] + store.closeups(source, page)
    if images:
        answer = generator.describe_images(build_chart_prompt(question, chunks, pages, measured, True), images,
                                           model_name=model_name)
        if answer:
            return answer, True
    if not measured:
        return "", False
    answer = generator.complete(build_chart_prompt(question, chunks, pages, measured, False),
                                model_name=model_name, groq_model_name=groq_model_name, temperature=0.0)
    return answer, False
