"""Turn extracted page text into clean, section-aware retrieval chunks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from src.qa.types import Chunk

TARGET_WORDS = 110
MAX_WORDS = 180

_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)+\s+[A-Z][^.?!]{2,90}$")
_CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,:&'()\-–]{5,80}$")
_SPACED_LETTERS = re.compile(r"^(?:[A-Za-z]\s){3,}[A-Za-z?]?\s*\??$")
# Symbol-font bullets (Wingdings "n", "q", "l") come through as a lone letter before the text.
_BULLET_PREFIX = re.compile(r"^(?:[nqlv•▪●■◆➢✓]\s+)(?=[A-Z(])")
_REPEATED_PHRASE = re.compile(r"\b(.{3,60}?)(?:\s+\1\b){1,}", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'“])")
_FIGURE_REF = re.compile(r"\b(?:Fig(?:ure)?\.?|Table)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_EXERCISE_START = re.compile(r"^(?:\d+\.|\([a-z]\)|\([ivx]+\))\s")


def _boilerplate_lines(pages: list[str]) -> set[str]:
    """Lines repeated on many pages are running headers/footers (e.g. 'Reprint 2026-27')."""
    if len(pages) < 4:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        counts.update({line.strip().lower() for line in text.splitlines() if 0 < len(line.strip()) <= 60})
    threshold = max(3, int(len(pages) * 0.3))
    return {line for line, count in counts.items() if count >= threshold}


def clean_line(line: str) -> str:
    line = line.strip()
    line = _BULLET_PREFIX.sub("", line)
    line = _REPEATED_PHRASE.sub(r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


def is_heading(line: str) -> bool:
    if len(line) > 90 or line.endswith((".", "?", ",", ";")):
        return False
    return bool(_NUMBERED_HEADING.match(line) or _CAPS_HEADING.match(line))


_ENUMERATOR_BREAK = re.compile(r"\s+(?=(?:\d{1,2}\.|\([a-z]\)|\([ivx]+\))\s+[A-Z(])")
_BARE_ENUMERATOR = re.compile(r"^(?:\d{1,2}\.|\([a-z]\)|\([ivx]+\))$")


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends and before numbered items ('1. What ...'), keeping the number with its item."""
    pieces: list[str] = []
    for block in _ENUMERATOR_BREAK.sub("\n", text).split("\n"):
        pieces.extend(s.strip() for s in _SENTENCE_SPLIT.split(block) if s.strip())
    merged: list[str] = []
    for piece in pieces:
        if merged and _BARE_ENUMERATOR.match(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _is_exercise(sentences: list[str]) -> bool:
    if not sentences:
        return False
    questiony = sum(1 for s in sentences if s.endswith("?") or _EXERCISE_START.match(s))
    return questiony / len(sentences) >= 0.5


def _page_text(page: Mapping[str, Any]) -> str:
    return str(page.get("text") or page.get("raw_text") or "")


def chunk_pages(pages: Iterable[Mapping[str, Any]], document_id: str = "doc") -> list[Chunk]:
    """Split pages into ~110-word chunks that never cross a section heading or a page boundary."""
    page_list = [p for p in pages if isinstance(p, Mapping)]
    boilerplate = _boilerplate_lines([_page_text(p) for p in page_list])
    chunks: list[Chunk] = []
    section: str | None = None

    for page in page_list:
        page_number = int(page.get("page_number") or 0)
        buffer: list[str] = []
        last_heading: str | None = None

        def flush(page_number: int = page_number) -> None:
            nonlocal buffer
            text = " ".join(buffer).strip()
            buffer = []
            if len(text.split()) < 3:
                return
            sentences = split_sentences(text)
            refs = tuple(dict.fromkeys(_FIGURE_REF.findall(text)))
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:p{page_number}:c{len(chunks)}",
                    page_number=page_number,
                    text=text,
                    section=section,  # noqa: B023 - intentionally the section current at flush time
                    kind="exercise" if _is_exercise(sentences) else "body",
                    figure_refs=refs,
                )
            )

        for raw in _page_text(page).splitlines():
            line = clean_line(raw)
            if not line or line.lower() in boilerplate or _SPACED_LETTERS.match(line):
                continue
            if is_heading(line):
                if line == last_heading:
                    continue
                flush()
                last_heading = line
                section = line
                continue
            buffer.append(line)
            words = len(" ".join(buffer).split())
            if (words >= TARGET_WORDS and re.search(r"[.!?]$", line)) or words >= MAX_WORDS:
                flush()
        flush()
    return chunks
