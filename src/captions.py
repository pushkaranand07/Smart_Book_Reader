"""Figure caption detection shared by the digital and scanned pipelines.

A caption is text that *starts* with "Fig./Figure/Table N(.N)" and sits next to the figure (above or below, same
column). Anything else (body paragraphs that mention a figure, activity headings, nearby labels) is not a caption,
and the figure gets caption=None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAPTION_START = re.compile(
    r"^\s*(?P<kind>fig(?:ure)?s?\.?|table)\s*(?P<num>\d{1,2}(?:\s?\.\s?\d{1,2}){0,2})"
    r"(?:\s*\((?P<sub>[a-z])\))?(?![\d.]*\d)\s*[:.\-–—]?\s*(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
SYMBOL_FONT = re.compile(r"wingding|symbol|dingbat|webding|zapf|mt ?extra|marlett|bookshelf|monotype ?sorts", re.I)
# A lone letter at the start of a line is how symbol-font bullets come through when the font name is lost.
_LONE_BULLET = re.compile(r"(?m)^(?:[nqlvu•▪●■◆➢✓]\s+)+(?=[A-Za-z(])")
_REPEATED_PHRASE = re.compile(r"\b(.{3,60}?)(?:\s+\1\b)+", re.IGNORECASE)
MAX_CAPTION_CHARS = 220


@dataclass(frozen=True)
class CaptionMatch:
    kind: str  # "figure" | "table"
    number: str  # e.g. "1.2"
    sub: str | None
    text: str | None  # caption text without the "Figure 1.2" prefix; None when only a label is present

    @property
    def label(self) -> str:
        suffix = f"({self.sub})" if self.sub else ""
        return f"{'Table' if self.kind == 'table' else 'Figure'} {self.number}{suffix}"


def clean_caption_text(text: str) -> str | None:
    text = _LONE_BULLET.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—:.;,")
    text = _REPEATED_PHRASE.sub(r"\1", text)
    if not re.search(r"[A-Za-z]{2,}", text):
        return None
    if len(text) > MAX_CAPTION_CHARS:
        cut = text[:MAX_CAPTION_CHARS]
        text = cut[: cut.rfind(" ")] if " " in cut else cut
    return text or None


def parse_caption(text: str) -> CaptionMatch | None:
    """Return the caption parts if `text` starts with a figure/table label, else None."""
    match = CAPTION_START.match(_LONE_BULLET.sub("", text.strip()))
    if not match:
        return None
    kind = "table" if match.group("kind").lower().startswith("tab") else "figure"
    number = re.sub(r"\s", "", match.group("num"))
    rest = match.group("rest") or ""
    # Drop-shadowed labels are extracted several times: "Figure 2.9 Figure 2.9 Removing water ...".
    while (repeat := CAPTION_START.match(rest)) and re.sub(r"\s", "", repeat.group("num")) == number:
        rest = repeat.group("rest") or ""
    # "Figure 2 shows ..." is a sentence about the figure, not its caption.
    if re.match(r"(?:shows?|illustrates?|depicts?|gives?|is|are|was|and|in|of|to)\b", rest.strip(), re.IGNORECASE):
        return None
    return CaptionMatch(kind=kind, number=number, sub=match.group("sub"), text=clean_caption_text(rest))


_LABEL_ONLY = re.compile(r"^\s*(?:fig(?:ure)?s?\.?|table)\s*\d{1,2}(?:\.\d{1,2}){0,2}\s*[:.]?\s*$", re.IGNORECASE)


def text_without_symbol_fonts(block: dict[str, Any], caption_mode: bool = False) -> str:
    """Join a PyMuPDF 'dict' text block, dropping spans set in symbol/dingbat fonts (bullets).

    With caption_mode, stop at the first font/size change after the label: PDF blocks sometimes merge a caption
    with the body paragraph that follows it, and the two are set in different styles.
    """
    lines: list[str] = []
    reference: tuple[str, float] | None = None
    for line in block.get("lines", []):
        spans = [s for s in line.get("spans", []) if not SYMBOL_FONT.search(str(s.get("font", "")))]
        joined = "".join(s.get("text", "") for s in spans).strip()
        if not joined:
            continue
        if caption_mode and spans:
            main = max(spans, key=lambda s: len(str(s.get("text", ""))))
            style = (str(main.get("font", "")), round(float(main.get("size", 0.0)), 1))
            if not _LABEL_ONLY.match(joined):
                if reference is None:
                    reference = style
                elif style[0] != reference[0] or abs(style[1] - reference[1]) > 0.6:
                    break
        lines.append(joined)
    return "\n".join(lines)


def horizontal_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Overlap as a fraction of the narrower box width."""
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    narrow = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return max(0.0, overlap) / narrow


def nearest_caption(
    figure_bbox: tuple[float, float, float, float],
    candidates: list[tuple[tuple[float, float, float, float], CaptionMatch]],
    max_gap: float,
    min_overlap: float = 0.3,
) -> CaptionMatch | None:
    """Pick the caption directly above or below the figure in the same column, within `max_gap`."""
    best: tuple[float, CaptionMatch] | None = None
    _x0, y0, _x1, y1 = figure_bbox
    for bbox, caption in candidates:
        if horizontal_overlap(figure_bbox, bbox) < min_overlap:
            continue
        if bbox[1] >= y1 - 2:  # caption below
            gap = bbox[1] - y1
        elif bbox[3] <= y0 + 2:  # caption above
            gap = y0 - bbox[3]
        else:  # caption inside the figure box (compound crops include it)
            gap = 0.0
        if 0 <= gap <= max_gap and (best is None or gap < best[0]):
            best = (gap, caption)
    return best[1] if best else None


class FigureNamer:
    """Sequential, caption-free figure filenames: <images>/<document>/p{page}_fig{n}.png."""

    def __init__(self, images_root: Path, document: str, page_number: int) -> None:
        self.directory = images_root / re.sub(r"[^A-Za-z0-9_-]+", "_", document)[:80]
        self.page_number = page_number
        self.count = 0

    def next_path(self) -> tuple[Path, str]:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.count += 1
        name = f"p{self.page_number}_fig{self.count}.png"
        return self.directory / name, name
