from PIL import Image

import src.scanned_pipeline as scanned_module
from src.scanned_pipeline import ScannedPipeline


def _primary_result(text="Mitochondria produce ATP.", confidence=0.95):
    return {
        "text": text,
        "words": text.split(),
        "boxes": [[10, 20, 30, 12] for _ in text.split()],
        "confidences": [confidence for _ in text.split()],
    }


def test_run_ocr_accepts_good_primary_result(monkeypatch):
    monkeypatch.setattr(
        scanned_module,
        "ocr_image_with_metadata",
        lambda *args, **kwargs: _primary_result(),
    )

    pipeline = ScannedPipeline(tesseract_ok=True)
    text, _, ocr_data, note = pipeline.run_ocr(Image.new("RGB", (200, 100), "white"))

    assert text == "Mitochondria produce ATP."
    assert ocr_data["text"] == ["Mitochondria", "produce", "ATP."]
    assert "PaddleOCR accepted" in note


def test_run_ocr_uses_complete_tesseract_result_when_primary_is_weak(monkeypatch):
    monkeypatch.setattr(
        scanned_module,
        "ocr_image_with_metadata",
        lambda *args, **kwargs: _primary_result("Mitochondria produce AIP.", 0.35),
    )

    tesseract_data = {
        "text": ["Mitochondria", "produce", "ATP."],
        "block_num": [1, 1, 1],
        "par_num": [1, 1, 1],
        "line_num": [1, 1, 1],
        "left": [10, 80, 150],
        "top": [20, 20, 20],
        "width": [60, 60, 30],
        "height": [12, 12, 12],
    }
    monkeypatch.setattr(
        scanned_module.pytesseract,
        "image_to_data",
        lambda *args, **kwargs: tesseract_data,
    )

    pipeline = ScannedPipeline(tesseract_ok=True)
    text, _, _, note = pipeline.run_ocr(Image.new("RGB", (200, 100), "white"))

    assert text == "Mitochondria produce ATP."
    assert "Tesseract selected" in note


def test_ocr_quality_rejects_empty_or_unreliable_results():
    assert ScannedPipeline._ocr_quality({}) == 0.0
    assert ScannedPipeline._ocr_quality(_primary_result("AIP", 0.35)) < 0.70