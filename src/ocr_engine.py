"""OCR engines with structured output.

Primary engine is selected with OCR_ENGINE:
- "rapidocr" (default): RapidOCR on onnxruntime (Apache-2.0), PP-OCR models, no Paddle runtime.
- "paddleocr": PaddleOCR 3.x, if installed.
Tesseract is the last-resort fallback (see ScannedPipeline.run_ocr), and pages that use it are flagged "degraded".
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# PaddleX enables oneDNN by default, but its Windows executor fails on some PaddleOCR 3.x models.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "false")


class OCRUnavailable(RuntimeError):
    """The requested OCR engine is not installed or failed to initialise."""


_lock = threading.Lock()
_rapid: Any = None
_paddle_readers: dict[str, Any] = {}


def primary_engine_name() -> str:
    return os.environ.get("OCR_ENGINE", "rapidocr").strip().lower()


def _image_array(image: Any) -> np.ndarray:
    from PIL import Image as PILImage

    return np.array(image.convert("RGB") if isinstance(image, PILImage.Image) else image)


def _to_xywh(points: Any) -> list[int]:
    arr = np.asarray(points, dtype=float)
    if arr.shape == (4,):
        x0, y0, x1, y1 = arr
    else:
        x0, y0 = arr.min(axis=0)
        x1, y1 = arr.max(axis=0)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _structured(engine: str, detections: Iterator[tuple[Any, str, float]], min_confidence: float) -> dict[str, Any]:
    words: list[str] = []
    boxes: list[list[int]] = []
    confidences: list[float] = []
    for box, text, score in detections:
        text = str(text).strip()
        if not text or float(score) < min_confidence:
            continue
        words.append(text)
        boxes.append(_to_xywh(box))
        confidences.append(float(score))
    return {"text": "\n".join(words), "words": words, "boxes": boxes, "confidences": confidences, "engine": engine}


# ----------------------------------------------------------------------------- RapidOCR


def _get_rapidocr() -> Any:
    global _rapid
    with _lock:
        if _rapid is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise OCRUnavailable("rapidocr is not installed") from exc
            try:
                _rapid = RapidOCR()
            except (OSError, RuntimeError, ValueError) as exc:
                logger.error("RapidOCR failed to initialise: %s", exc)
                raise OCRUnavailable(f"RapidOCR failed to initialise: {exc}") from exc
        return _rapid


def ocr_image_rapidocr_structured(image: Any, min_confidence: float = 0.3) -> dict[str, Any]:
    output = _get_rapidocr()(_image_array(image))
    boxes = getattr(output, "boxes", None)
    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    if boxes is None or texts is None or scores is None:
        return _structured("RapidOCR", iter(()), min_confidence)
    return _structured("RapidOCR", zip(boxes, texts, scores, strict=False), min_confidence)


# ----------------------------------------------------------------------------- PaddleOCR (optional)


def _get_paddleocr(lang_list: list[str] | None = None) -> Any:
    language = (lang_list or ["en"])[0]
    with _lock:
        if language not in _paddle_readers:
            try:
                # PaddlePaddle and PyTorch ship overlapping Windows DLLs; importing torch first avoids a clash.
                import torch  # noqa: F401
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise OCRUnavailable("paddleocr is not installed") from exc
            try:
                _paddle_readers[language] = PaddleOCR(
                    lang=language,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise OCRUnavailable(f"PaddleOCR failed to initialise: {exc}") from exc
    return _paddle_readers[language]


def _iter_paddle_results(results: Any) -> Iterator[tuple[Any, str, float]]:
    """Normalise PaddleOCR 3.x result objects into (box, text, score)."""
    import json

    for result in results or []:
        data = getattr(result, "json", result)
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = json.loads(data)
        data = data.get("res", data) if isinstance(data, dict) else {}
        boxes = data.get("rec_boxes", []) or data.get("dt_polys", [])
        yield from zip(boxes, data.get("rec_texts", []), data.get("rec_scores", []), strict=False)


def ocr_image_paddleocr_structured(
    image: Any, languages: list[str] | None = None, min_confidence: float = 0.3
) -> dict[str, Any]:
    reader = _get_paddleocr(languages or ["en"])
    return _structured("PaddleOCR", _iter_paddle_results(reader.predict(_image_array(image))), min_confidence)


# ----------------------------------------------------------------------------- public API


def ocr_image_with_metadata(
    image: Any, languages: list[str] | None = None, min_confidence: float = 0.3
) -> dict[str, Any]:
    """Run the configured primary OCR engine and return text, line boxes and confidences."""
    if primary_engine_name() == "paddleocr":
        return ocr_image_paddleocr_structured(image, languages=languages, min_confidence=min_confidence)
    return ocr_image_rapidocr_structured(image, min_confidence=min_confidence)
