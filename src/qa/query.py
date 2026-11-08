"""Question analysis: split comparison questions into one sub-question per concept."""

from __future__ import annotations

import re

_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_TRAILING = re.compile(r"[\s?.!]+$")
_PATTERNS = [
    # "What is the difference between X and Y?", "differences between X and Y"
    re.compile(
        r"^(?:what\s+(?:is|are)\s+)?(?:the\s+)?(?:main\s+|key\s+)?(?:differences?|distinctions?|contrast)\s+"
        r"between\s+(?P<a>.+?)\s+and\s+(?P<b>.+)$",
        re.IGNORECASE,
    ),
    # "How is X different from Y?", "How do X differ from Y?", "How does X compare with Y?"
    re.compile(
        r"^how\s+(?:is|are|do|does)\s+(?P<a>.+?)\s+(?:different\s+from|differ\s+from|compared?\s+(?:to|with))\s+(?P<b>.+)$",
        re.IGNORECASE,
    ),
    # "Compare X and Y", "Contrast X with Y", "Distinguish between X and Y"
    re.compile(
        r"^(?:compare|contrast|distinguish)(?:\s+between)?\s+(?P<a>.+?)\s+(?:and|with|from)\s+(?P<b>.+)$",
        re.IGNORECASE,
    ),
    # "X vs Y", "X versus Y"
    re.compile(r"^(?P<a>.+?)\s+(?:vs\.?|versus)\s+(?P<b>.+)$", re.IGNORECASE),
]


def _clean(part: str) -> str:
    return _ARTICLE.sub("", _TRAILING.sub("", part.strip())).strip()


def split_comparison(question: str) -> list[str] | None:
    """Return the compared concepts, e.g. ["displacement reaction", "double displacement reaction"], or None.

    A shared head noun is copied to the shorter side: "aerobic and anaerobic respiration" ->
    ["aerobic respiration", "anaerobic respiration"].
    """
    text = _TRAILING.sub("", question.strip())
    for pattern in _PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        a, b = _clean(match.group("a")), _clean(match.group("b"))
        if not a or not b or a.lower() == b.lower():
            return None
        a_words, b_words = a.split(), b.split()
        if len(b_words) >= 2 and len(a_words) < len(b_words) and a_words[-1].lower() != b_words[-1].lower():
            a = f"{a} {b_words[-1]}"
        return [a, b]
    return None


def sub_question(concept: str) -> str:
    return f"What is {concept}?"
