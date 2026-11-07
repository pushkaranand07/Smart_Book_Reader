"""Caption detection: unit tests, a synthetic PDF, and an opt-in precision check on the labeled set."""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from src.captions import FigureNamer, nearest_caption, parse_caption, text_without_symbol_fonts
from src.digital_pipeline import parse_subfigures_from_caption

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "tests" / "fixtures" / "captions_labeled.json"
JUNK = re.compile(r"(^|\s)[nqlv](\s|$)|[•▪●■◆➢✓]|\.png|_p\d+_|fig\d+_\d")


@pytest.mark.parametrize(
    ("text", "label", "caption"),
    [
        ("Figure 1.1 Burning of a magnesium ribbon in air", "Figure 1.1", "Burning of a magnesium ribbon in air"),
        ("Fig. 12.1 A straight wire", "Figure 12.1", "A straight wire"),
        ("Table 2.1 Indicators", "Table 2.1", "Indicators"),
        ("Figure 1.8 (b) Iron nails", "Figure 1.8(b)", "Iron nails"),
        ("Figure 2.9 Figure 2.9 Removing water of crystallisation", "Figure 2.9", "Removing water of crystallisation"),
        ("Figure 4.12", "Figure 4.12", None),
    ],
)
def test_parse_caption_accepts_real_captions(text, label, caption):
    match = parse_caption(text)
    assert match is not None and match.label == label and match.text == caption


@pytest.mark.parametrize(
    "text",
    [
        "n Pass the gas produced in each case through lime water",
        "as shown in Fig. 12.1. The wire XY is kept",
        "Figure 1.2 shows hydrogen gas formation",
        "Figure 2014 was a year",
        "Activity 1.4 Take a beaker",
    ],
)
def test_parse_caption_rejects_body_text_and_bullets(text):
    assert parse_caption(text) is None


def test_subfigure_tags_ignore_state_symbols():
    assert parse_subfigures_from_caption("Variation of H+(aq) and OH–(aq) ions") == []
    assert [t for t, _ in parse_subfigures_from_caption("Leaf (a) before and (b) after starch test")] == ["a", "b"]


def test_symbol_font_spans_and_style_change_are_dropped():
    block = {
        "lines": [
            {
                "spans": [
                    {"text": "n", "font": "Wingdings-Regular", "size": 9},
                    {"text": "Figure 1.3", "font": "B", "size": 9},
                ]
            },
            {"spans": [{"text": "Formation of slaked lime", "font": "Bookman-LightItalic", "size": 9}]},
            {"spans": [{"text": "Calcium oxide reacts vigorously", "font": "Bookman-Light", "size": 10.5}]},
        ]
    }
    assert text_without_symbol_fonts(block, caption_mode=True) == "Figure 1.3\nFormation of slaked lime"
    assert "Calcium" in text_without_symbol_fonts(block)


def test_nearest_caption_requires_same_column_and_small_gap():
    cap = parse_caption("Figure 1.1 A test")
    assert cap is not None
    fig = (100.0, 100.0, 300.0, 250.0)
    assert nearest_caption(fig, [((110, 255, 290, 270), cap)], max_gap=40) == cap
    assert nearest_caption(fig, [((400, 255, 500, 270), cap)], max_gap=40) is None  # other column
    assert nearest_caption(fig, [((110, 400, 290, 415), cap)], max_gap=40) is None  # too far


def test_figure_namer_is_sequential_and_caption_free(tmp_path):
    namer = FigureNamer(tmp_path, "book name (1)", 7)
    names = [namer.next_path()[1] for _ in range(3)]
    assert names == ["p7_fig1.png", "p7_fig2.png", "p7_fig3.png"]
    assert namer.directory.name == "book_name_1_"


def _png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (200, 120), color)
    for x in range(0, 200, 10):  # add detail so the "meaningful image" filter keeps it
        for y in range(120):
            img.putpixel((x, y), (0, 0, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_synthetic_pdf_end_to_end(tmp_path, monkeypatch):
    import src.digital_pipeline as dp
    from src.pdf_processor import PDFProcessor

    monkeypatch.setattr(dp, "IMAGES_DIR", tmp_path / "images")
    pdf = tmp_path / "synthetic.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "n", fontname="zadb", fontsize=11)
    page.insert_text((90, 60), "Pass the gas through lime water and note what happens to it.", fontsize=11)
    page.insert_image(pymupdf.Rect(72, 90, 272, 210), stream=_png((200, 120, 40)))
    page.insert_text((72, 228), "Figure 1.1 Burning of a magnesium ribbon", fontsize=9)
    # Far enough below Figure 1.1 (> 220 pt) that it is a separate, uncaptioned picture.
    page.insert_text(
        (72, 540), "This paragraph sits near the second picture but is not its caption at all.", fontsize=11
    )
    page.insert_image(pymupdf.Rect(72, 560, 272, 680), stream=_png((40, 120, 200)))
    doc.save(pdf)
    doc.close()

    result = PDFProcessor().process_pdf(pdf)
    figures = result["pages"][0].figures
    captions = {f["figure_label"]: f["caption"] for f in figures}
    assert captions.get("Figure 1.1") == "Burning of a magnesium ribbon"
    uncaptioned = [f for f in figures if f["figure_label"].startswith("Page 1 image")]
    assert uncaptioned and all(f["caption"] is None for f in uncaptioned)
    for fig in figures:
        assert re.fullmatch(r"p1_fig\d+\.png", fig["image_filename"]), fig["image_filename"]
        assert not fig["caption"] or not JUNK.search(fig["caption"])


@pytest.mark.skipif(os.environ.get("RUN_CAPTION_EVAL") != "1", reason="set RUN_CAPTION_EVAL=1 (needs local PDFs)")
def test_caption_precision_on_labeled_books(tmp_path, monkeypatch):
    import src.digital_pipeline as dp
    import src.scanned_pipeline as sp
    from src.pdf_processor import PDFProcessor

    monkeypatch.setattr(dp, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(sp, "IMAGES_DIR", tmp_path / "images")
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["figures"]
    expected: dict[tuple[str, int, str], list[str]] = {}
    for f in labels:  # a label can repeat on a page (e.g. three panels all titled "Figure 4.6")
        expected.setdefault((f["book"], f["page"], f["label"]), []).append(f["caption_prefix"])
    processor = PDFProcessor()
    emitted, correct, scanned_engines = 0, 0, set()
    for book in sorted({f["book"] for f in labels}):
        src = ROOT / "data" / "uploads" / f"{book}.pdf"
        if not src.exists():
            pytest.skip(f"missing local PDF {src.name}")
        subset = next((f.get("pages_subset") for f in labels if f["book"] == book and f.get("pages_subset")), None)
        if subset:
            part = tmp_path / f"{book}.pdf"
            with pymupdf.open(src) as full, pymupdf.open() as out:
                out.insert_pdf(full, from_page=subset[0] - 1, to_page=subset[1] - 1)
                out.save(part)
            src = part
        for page in processor.process_pdf(src)["pages"]:
            if page.page_type == "Scanned":
                scanned_engines.add(page.confidence_note.split(" ")[0])
            for fig in page.figures:
                if not fig["caption"]:
                    continue
                emitted += 1
                assert not JUNK.search(fig["caption"]), fig["caption"]
                wanted = expected.get((book, page.page_number, fig["figure_label"]), [])
                correct += any(fig["caption"].startswith(w) for w in wanted)
    precision = correct / max(1, emitted)
    print(f"caption precision {correct}/{emitted} = {precision:.3f}; scanned OCR engines: {scanned_engines}")
    assert len(labels) >= 30 and precision >= 0.95
    assert scanned_engines and scanned_engines <= {"RapidOCR", "PaddleOCR"}
