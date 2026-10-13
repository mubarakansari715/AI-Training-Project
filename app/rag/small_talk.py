"""Small talk and conversational-recall detection.

Strict document mode exists to stop the assistant from fabricating
*document* facts — it was never meant to make it refuse to say hello,
acknowledge an introduction, or recall something you told it two
messages ago. This lets those cases through to the LLM (which already
has instructions to use conversation history for continuity) even
when nothing in the documents matches them, while leaving genuine
document questions and unrelated general-knowledge questions subject
to the normal strict-mode refusal.
"""

from __future__ import annotations

import re

from app.config.settings import APP_NAME

_GREETING_RE = re.compile(
    r"^((hi|hello|hey|hiya|yo)(\s+there)?|"
    r"good\s*(morning|afternoon|evening|night)|"
    r"how('?s| is| are)\s+(it\s+going|things|you\b.*)|what'?s\s+up|"
    r"thanks?(\s+you)?|thank\s+you|ty|"
    r"bye|goodbye|see\s+ya|see\s+you|take\s+care|"
    r"ok(ay)?|cool|nice|great|awesome)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

# A short self-introduction, optionally preceded by a greeting, with
# nothing else in the message — e.g. "hi my name is ali", "i'm sam".
# Deliberately bounded to 1-3 short words so it doesn't swallow real
# sentences that happen to start with "I am ..." / "I'm ...".
_INTRO_RE = re.compile(
    r"^(hi|hello|hey)?[,!.\s]*"
    r"(my\s+name\s+is|i'?m|i\s+am)\s+"
    r"[a-z][a-z'\-]{0,20}(\s+[a-z][a-z'\-]{0,20}){0,2}"
    r"[\s!.,]*$",
    re.IGNORECASE,
)

# Questions about the conversation itself, not the documents.
_RECALL_RE = re.compile(
    r"^(what'?s|what\s+is)\s+my\s+name\s*\??\s*$|"
    r"^do\s+you\s+(remember|recall)\b|"
    r"^what\s+did\s+i\s+(say|tell\s+you|ask)\b|"
    r"^what\s+did\s+you\s+(say|tell\s+me)\b|"
    r"^(can\s+you\s+)?repeat\s+(that|what\s+i\s+said)\s*\??\s*$|"
    r"^what\s+were\s+we\s+talking\s+about\s*\??\s*$",
    re.IGNORECASE,
)

# Questions about the assistant itself ("who are you?", "what's your
# name?"), answered from its system instructions rather than documents.
_IDENTITY_RE = re.compile(
    r"^(who\s+are\s+you|what\s+are\s+you|"
    r"(what'?s|what\s+is)\s+your\s+name|"
    r"what\s+should\s+i\s+call\s+you|introduce\s+yourself)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

# Questions about who made/owns the assistant ("who built you?", "who is
# the founder of ALLY AI?"). The target must be the assistant itself, so
# "who is the owner of the property?" stays a document question.
_ROLE = r"(found\w*|creators?|owners?|makers?|developers?|authors?|builders?)"
_TARGET = (
    r"(you|this(\s+(app|bot|chatbot|assistant|ai|tool))?|(this\s+)?"
    + r"\s+".join(re.escape(w) for w in APP_NAME.lower().split())
    + r"|(this\s+)?" + re.escape(APP_NAME.lower().split()[0]) + ")"
)
_CREATOR_RE = re.compile(
    rf"^(who\s+(is|are|was)\s+(the\s+)?{_ROLE}(\s+(and|&|/)\s+{_ROLE})?\s+of\s+{_TARGET}|"
    rf"who\s+(is|are)\s+your\s+{_ROLE}(\s+(and|&|/)\s+{_ROLE})?|"
    rf"who\s+(built|created|made|developed|designed|founded|owns|programmed)\s+{_TARGET})"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)


def is_small_talk(text: str) -> bool:
    """Return True if `text` is a greeting, self-introduction, a
    question about the assistant, or a question about the conversation
    itself — not a real document question."""
    stripped = text.strip()
    return bool(
        _GREETING_RE.match(stripped)
        or _INTRO_RE.match(stripped)
        or _RECALL_RE.match(stripped)
        or _IDENTITY_RE.match(stripped)
        or _CREATOR_RE.match(stripped)
    )
