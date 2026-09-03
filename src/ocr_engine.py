"""
ocr_engine.py — Advanced OCR using EasyOCR with Tesseract fallback.

Supports 80+ languages, better accuracy on blurry/scanned pages, no system install needed.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np

_easyocr_lock = threading.Lock()
_easyocr_reader = None


def _get_easyocr(lang_list: List[str] = None):
    """Lazy-load EasyOCR reader (thread-safe, loaded once)."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    with _easyocr_lock:
        if _easyocr_reader is None:
            try:
                import easyocr
                langs = lang_list or ["en"]
                _easyocr_reader = easyocr.Reader(
                    langs,
                    gpu=False,          # CPU mode — works everywhere
                    verbose=False,
                    model_storage_directory=None,  # use default ~/.EasyOCR
                )
            except Exception as e:
                raise RuntimeError(f"EasyOCR failed to initialize: {e}")
    return _easyocr_reader


def ocr_image_easyocr(
    image,  # PIL Image or numpy array
    languages: List[str] = None,
    min_confidence: float = 0.3,
) -> str:
    """Extract text from image using EasyOCR.

    Args:
        image: PIL Image or numpy uint8 array.
        languages: Language codes (e.g. ['en', 'hi']). Defaults to ['en'].
        min_confidence: Minimum confidence threshold (0-1). Defaults to 0.3.

    Returns:
        Extracted text string, joined by newlines.
    """
    import numpy as np
    from PIL import Image as PILImage

    langs = languages or ["en"]
    reader = _get_easyocr(langs)

    # Convert PIL → numpy if needed
    if isinstance(image, PILImage.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = np.array(image)

    results = reader.readtext(img_array, detail=1, paragraph=False)

    lines = []
    for (bbox, text, confidence) in results:
        if confidence >= min_confidence and text.strip():
            lines.append(text.strip())

    return "\n".join(lines)


def ocr_image_tesseract(image, lang: str = "eng") -> str:
    """Fallback OCR using pytesseract (original pipeline)."""
    try:
        import pytesseract
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        words = [
            w for w, conf in zip(data["text"], data["conf"])
            if w.strip() and int(conf) > 40
        ]
        return " ".join(words)
    except Exception:
        return ""


def ocr_image(
    image,
    prefer_easyocr: bool = True,
    languages: List[str] = None,
    min_confidence: float = 0.3,
) -> str:
    """Run OCR with EasyOCR (preferred) or fall back to Tesseract.

    Args:
        image: PIL Image or numpy array.
        prefer_easyocr: If True, tries EasyOCR first then falls back to Tesseract.
        languages: Language codes for EasyOCR (e.g. ['en', 'hi']).
        min_confidence: Minimum confidence for EasyOCR results.

    Returns:
        Extracted text string.
    """
    if prefer_easyocr:
        try:
            text = ocr_image_easyocr(image, languages=languages, min_confidence=min_confidence)
            if text.strip():
                return text
        except Exception:
            pass  # Fall through to Tesseract
    return ocr_image_tesseract(image)
