"""Document structure: which chapter/section/box each page belongs to.

Long reports repeat the same vocabulary in many places ("growth is
projected to…" appears in every regional section), so plain similarity
search can return a page from the wrong section. Knowing each page's
section lets the pipeline label chunks with it, let questions that name a
section ("according to the executive summary…", "Chapter 3", "Box 1.1",
"South Asia") search only there, and keep front matter (table of
contents, abbreviations) out of answers.

Signals, combined because none is reliable alone:
- PDF bookmarks give section names, but can point a page or two off, so
  each title is re-located to the nearby page where it actually appears
  as a heading;
- running headers ("CHAPTER 3 … | JUNE 2026") give the chapter per page;
- uppercase "BOX n.n" labels mark boxed articles (body text refers to
  them in lowercase, "see box 1.1");
- near-empty pages (separators, figure-only pages) inherit the label of
  the page before.

Documents without bookmarks or chapter headers get no labels and are
handled exactly as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEPARATOR = " › "
_HEAD_CHARS = 400  # where a page's headings and running header live
_RELOCATE_WINDOW = 3  # pages either side of a bookmark to look for its title
FRONT_MATTER = {"contents", "summaryofcontents", "tableofcontents", "abbreviations", "acronyms"}

_CHAPTER_TITLE_RE = re.compile(r"^(?:ch\.?|chapter)\s*(\d+)\b\.?\s*:?\s*(.*)$", re.IGNORECASE)
_BOX_TITLE_RE = re.compile(r"^box\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_HEADER_CHAPTER_RE = re.compile(r"chapter(\d+)")
_BOX_LABEL_RE = re.compile(r"\bBOX\s*(\d+\.\d+)\b")  # case-sensitive on purpose
_NEAR_EMPTY_CHARS = 200


def compact(text: str) -> str:
    """Lowercase letters/digits only — robust to 'C H A PT ER' and punctuation."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title.replace("--", " ")).strip(" -–—:")
    return title


@dataclass
class _Entry:
    title: str
    kind: str  # "chapter" | "box" | "front" | "section"
    page: int
    number: str | None = None


def _classify(title: str, page: int) -> _Entry:
    if m := _CHAPTER_TITLE_RE.match(title):
        return _Entry(title, "chapter", page, m.group(1))
    if m := _BOX_TITLE_RE.match(title):
        return _Entry(title, "box", page, m.group(1))
    if compact(title) in FRONT_MATTER:
        return _Entry(title, "front", page)
    return _Entry(title, "section", page)


def _heading_key(entry: _Entry) -> str:
    """What the entry's heading looks like in page text (compacted)."""
    if entry.kind == "chapter":
        return f"chapter{entry.number}"
    if entry.kind == "box":
        return f"box{entry.number}"
    return compact(entry.title)


def _relocate(entry: _Entry, heads: dict[int, str]) -> int:
    """The page near the bookmark where the title actually appears as a heading."""
    key = _heading_key(entry)
    candidates = [p for p in range(entry.page - _RELOCATE_WINDOW, entry.page + _RELOCATE_WINDOW + 1)
                  if p in heads and key and key in heads[p]]
    return min(candidates, key=lambda p: (abs(p - entry.page), p)) if candidates else entry.page


def detect_sections(page_texts: dict[int, str], outline: list[tuple[str, int]]) -> dict[int, str]:
    """Map page number -> section label like 'Ch. 2 Regional Outlooks › South Asia'.

    `outline` is the flattened bookmark list as (title, 1-based page).
    Pages that can't be placed are left out of the result.
    """
    heads = {p: compact(t[:_HEAD_CHARS]) for p, t in page_texts.items()}
    entries = [_classify(clean_title(t), p) for t, p in outline if clean_title(t)]
    for e in entries:
        e.page = _relocate(e, heads)

    chapters = {e.number: e.title for e in entries if e.kind == "chapter"}
    boxes = {e.number: e.title for e in entries if e.kind == "box"}
    # Top-level markers: where a chapter, a standalone section or front matter starts.
    markers = sorted((e for e in entries if e.kind != "box"), key=lambda e: e.page)

    chapter_starts = {e.number: e.page for e in entries if e.kind == "chapter"}

    marker_pages = {m.page for m in markers}
    labels: dict[int, str] = {}
    previous: str | None = None
    for page in sorted(page_texts):
        text = page_texts[page]
        starts_something = page in marker_pages or _HEADER_CHAPTER_RE.search(heads[page][:120])
        if len(text.strip()) < _NEAR_EMPTY_CHARS and previous and not starts_something:
            labels[page] = previous  # blank separator / figure-only page
            continue
        marker = None
        for m in markers:
            if m.page > page:
                break
            marker = m
        header = _HEADER_CHAPTER_RE.search(heads[page][:120])
        chapter_no = header.group(1) if header else (marker.number if marker and marker.kind == "chapter" else None)

        if marker is None:
            parts = [_chapter_title(chapters, chapter_no)] if chapter_no else []
        elif marker.kind == "front":
            parts = [marker.title]
        elif marker.kind == "chapter":
            parts = [_chapter_title(chapters, chapter_no)]
        elif chapter_no is None:
            parts = [marker.title]  # standalone section, e.g. Executive Summary
        elif marker.page >= chapter_starts.get(chapter_no, 0):
            parts = [_chapter_title(chapters, chapter_no), marker.title]  # section inside the chapter
        else:
            parts = [_chapter_title(chapters, chapter_no)]  # header moved on: old section ended

        box = _BOX_LABEL_RE.search(text)
        if box and parts and not (marker and marker.kind == "front"):
            parts = [parts[0], boxes.get(box.group(1), f"Box {box.group(1)}")]
        if parts:
            labels[page] = previous = SEPARATOR.join(parts)
    return labels


def _chapter_title(chapters: dict[str, str], number: str) -> str:
    return chapters.get(number, f"Chapter {number}")


def flatten_outline(reader) -> list[tuple[str, int]]:
    """A pypdf reader's bookmarks as (title, 1-based page); [] if unavailable."""
    result: list[tuple[str, int]] = []

    def walk(items):
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            try:
                result.append((item.title, reader.get_destination_page_number(item) + 1))
            except Exception:  # noqa: BLE001, S112 — malformed bookmarks are skipped, not fatal
                continue

    try:
        walk(reader.outline)
    except Exception:  # noqa: BLE001
        return []
    return result


def is_front_matter(label: str | None) -> bool:
    return bool(label) and compact(label.split(SEPARATOR)[0]) in FRONT_MATTER


def _aliases(component: str) -> list[str]:
    """Ways a question may refer to one part of a section label."""
    if m := _CHAPTER_TITLE_RE.match(component):
        n = m.group(1)
        return [f"chapter {n}", f"ch. {n}", f"ch {n}"]
    if m := _BOX_TITLE_RE.match(component):
        return [f"box {m.group(1)}"]
    words = component.split()
    # Single generic words ("Introduction") would match too many questions.
    return [component.lower()] if len(words) >= 2 else []


def sections_mentioned(question: str, labels: list[str]) -> list[list[str]]:
    """The sections a question explicitly names, one group per thing named:
    "Chapter 3" -> [all Ch. 3 labels, incl. its boxes]; "South Asia and
    Sub-Saharan Africa" -> [[South Asia labels], [Sub-Saharan Africa labels]]."""
    q = " " + re.sub(r"\s+", " ", question.lower()) + " "
    groups: dict[str, list[str]] = {}
    for label in labels:
        for component in label.split(SEPARATOR):
            # Not followed by a letter/digit or ".digit" (so "box 1.1" doesn't
            # match "box 1.10"), but a sentence-ending period is fine.
            if any(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9]|\.\d)", q) for alias in _aliases(component)):
                groups.setdefault(component, []).append(label)
    return list(groups.values())
