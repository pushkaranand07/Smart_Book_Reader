"""Primary PaddleOCR engine with structured output and Tesseract fallback."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List

import numpy as np

_paddle_lock = threading.Lock()
_paddle_readers: Dict[str, Any] = {}
logger = logging.getLogger(__name__)

# PaddleX enables oneDNN by default, but its Windows executor currently fails
# on some PaddleOCR 3.x models. This must be set before PaddlePaddle imports.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "false")


def _get_paddleocr(lang_list: List[str] = None):
    """Lazy-load one PaddleOCR reader per requested language."""
    language = (lang_list or ["en"])[0]
    with _paddle_lock:
        if language not in _paddle_readers:
            try:
                # PaddlePaddle and PyTorch expose overlapping Windows DLLs;
                # loading Torch first prevents PaddlePaddle from breaking it.
                import torch  # noqa: F401
                from paddleocr import PaddleOCR

                _paddle_readers[language] = PaddleOCR(
                    lang=language,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
            except Exception as exc:
                logger.exception("PaddleOCR failed to initialize")
                raise RuntimeError(f"PaddleOCR failed to initialize: {exc}") from exc
    return _paddle_readers[language]


def _iter_paddle_results(results):
    """Normalize PaddleOCR 3.x result objects into boxes and text scores."""
    for result in results or []:
        data = getattr(result, "json", result)
        if callable(data):
            data = data()
        if isinstance(data, str):
            import json

            data = json.loads(data)
        data = data.get("res", data) if isinstance(data, dict) else {}
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = data.get("rec_boxes", [])
        if not boxes:
            boxes = data.get("dt_polys", [])
        for box, text, score in zip(boxes, texts, scores):
            yield box, (text, score)


def _image_array(image):
    from PIL import Image as PILImage

    return np.array(image.convert("RGB") if isinstance(image, PILImage.Image) else image)


def ocr_image_paddleocr_structured(
    image,
    languages: List[str] = None,
    min_confidence: float = 0.3,
) -> Dict[str, Any]:
    """Return PaddleOCR text, boxes, and confidence metadata."""
    reader = _get_paddleocr(languages or ["en"])
    results = reader.predict(_image_array(image))

    words: List[str] = []
    boxes: List[List[int]] = []
    confidences: List[float] = []
    for bbox, detected in _iter_paddle_results(results):
        text, confidence = detected
        text = str(text).strip()
        confidence = float(confidence)
        if not text or confidence < min_confidence:
            continue
        points = np.asarray(bbox, dtype=float)
        if points.shape == (4,):
            x0, y0, x1, y1 = points
        else:
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
        words.append(text)
        boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
        confidences.append(confidence)

    return {
        "text": "\n".join(words),
        "words": words,
        "boxes": boxes,
        "confidences": confidences,
        "engine": "PaddleOCR",
    }


def ocr_image_paddleocr(image, languages: List[str] = None, min_confidence: float = 0.3) -> str:
    """Extract text with PaddleOCR."""
    return ocr_image_paddleocr_structured(
        image,
        languages=languages,
        min_confidence=min_confidence,
    )["text"]


def ocr_image_tesseract(image, lang: str = "eng") -> str:
    """Fallback OCR using pytesseract."""
    try:
        import pytesseract

        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        words = [
            word
            for word, confidence in zip(data["text"], data["conf"])
            if word.strip() and int(float(confidence)) > 40
        ]
        return " ".join(words)
    except Exception:
        logger.exception("Tesseract OCR failed")
        return ""


def ocr_image(
    image,
    prefer_paddleocr: bool = True,
    languages: List[str] = None,
    min_confidence: float = 0.3,
) -> str:
    """Run PaddleOCR or fall back to Tesseract."""
    if prefer_paddleocr:
        try:
            text = ocr_image_paddleocr(image, languages=languages, min_confidence=min_confidence)
            if text.strip():
                return text
        except Exception:
            logger.exception("PaddleOCR text extraction failed; using Tesseract fallback")
    return ocr_image_tesseract(image)


def ocr_image_with_metadata(
    image,
    languages: List[str] = None,
    min_confidence: float = 0.3,
) -> Dict[str, Any]:
    """Run only PaddleOCR and return structured output for quality gating."""
    return ocr_image_paddleocr_structured(
        image,
        languages=languages,
        min_confidence=min_confidence,
    )
