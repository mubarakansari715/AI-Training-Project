"""Text cleaning.

Normalizes text extracted from documents so it's easier to chunk and
retrieve, without changing its meaning: fixes excessive whitespace,
collapses unnecessary blank lines, rejoins words broken across a line
by a hyphen (a common PDF-extraction artifact), expands typographic
ligatures, and strips stray control characters.

Ligatures matter more than they look: PDFs set by publishing software
(e.g. the World Bank report) store "fi", "fl", "ff", "ffi" as one character
("ﬁscal", "inﬂation"), so a search for "inflation" never matches the
document's word, and keyword search splits "inﬂation" into "in" + "ation".
"""

from __future__ import annotations

import re

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Latin ligature characters (U+FB00-U+FB06) -> their letters.
_LIGATURES = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
                            "\ufb05": "st", "\ufb06": "st"})
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")


def expand_ligatures(text: str) -> str:
    return text.translate(_LIGATURES)


def clean_text(text: str) -> str:
    """Clean extracted text without altering its meaning.

    - Normalizes line endings
    - Strips non-printable control characters
    - Expands ligature characters ("ﬁ" -> "fi")
    - Rejoins hyphen-broken words split across a line wrap
    - Collapses repeated spaces/tabs into one
    - Collapses 3+ consecutive blank lines into a single blank line
    - Trims leading/trailing whitespace
    """
    if not text:
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = expand_ligatures(text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)

    return text.strip()
