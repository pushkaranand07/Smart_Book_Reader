"""
florence_detector.py — Application integration for Florence-2 Picture detection.

Uses the canonical training pipeline in src/layout/ for inference.
"""

from __future__ import annotations

import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.layout.florence_model import load_trained_model
from src.layout.florence_processor import generate_detections
from src.layout.paths import FLORENCE_LAYOUT_DIR

logger = logging.getLogger(__name__)

_HUB_MODEL_ID = "microsoft/Florence-2-base"

_florence_lock = threading.Lock()
_florence_model = None
_florence_processor = None
_florence_device = None
_florence_available = False
_florence_status = "Florence-2 not yet initialized"


def _resolve_adapter_dir() -> Path:
    """Only use the canonical Colab/train output path (no legacy adapters)."""
    return FLORENCE_LAYOUT_DIR / "best"


def _load_florence():
    global _florence_model, _florence_processor, _florence_device, _florence_available, _florence_status
    if _florence_model is not None:
        return _florence_available, _florence_status

    with _florence_lock:
        if _florence_model is not None:
            return _florence_available, _florence_status
        try:
            adapter = _resolve_adapter_dir()
            _florence_model, _florence_processor, _florence_device = load_trained_model(adapter)
            if (adapter / "adapter_model.safetensors").exists():
                _florence_status = f"Florence-2 fine-tuned Picture detector ready on {_florence_device.upper()}"
            else:
                _florence_status = f"Florence-2-base (no adapter) on {_florence_device.upper()}"
            _florence_available = True
        except Exception as e:
            _florence_available = False
            _florence_status = f"Florence-2 initialization error: {e}"

    return _florence_available, _florence_status


class FlorenceVisualDetector:
    """Detects diagram/figure regions using fine-tuned Florence-2 <OD>."""

    def __init__(self):
        self.is_available, self.status_msg = _load_florence()

    def detect_visual_boxes(
        self,
        pil_image: Image.Image,
        min_box_area: int = 4000,
        max_page_coverage: float = 0.55,
    ) -> List[Dict[str, Any]]:
        if not self.is_available:
            return []

        img_w, img_h = pil_image.size
        page_area = img_w * img_h
        results: List[Dict[str, Any]] = []

        try:
            with _florence_lock:
                det = generate_detections(_florence_model, _florence_processor, pil_image, _florence_device)
            for bbox, label in zip(det["bboxes"], det.get("labels") or ["Picture"] * len(det["bboxes"])):
                clean = (label or "Picture").strip()
                if clean.lower() not in ("picture", "diagram", "figure", "image", "chart", "graph"):
                    continue
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                w, h = x1 - x0, y1 - y0
                area = w * h
                if area < min_box_area or area > page_area * max_page_coverage:
                    continue
                results.append({
                    "label": clean,
                    "confidence": None,
                    "bbox": (x0, y0, x1, y1),
                    "width": w,
                    "height": h,
                })
        except Exception:
            logger.exception("Florence visual detection failed")
            pass

        results.sort(key=lambda r: r["width"] * r["height"], reverse=True)
        return results

    def caption_image(self, pil_image: Image.Image) -> str:
        if not self.is_available:
            return ""
        try:
            with _florence_lock:
                det = generate_detections(
                    _florence_model, _florence_processor, pil_image, _florence_device, max_new_tokens=256
                )
            return det.get("raw_text", "").strip()
        except Exception:
            logger.exception("Florence image captioning failed")
            return ""

    def detect_and_caption(self, pil_image: Image.Image) -> Dict[str, Any]:
        return {
            "caption": self.caption_image(pil_image),
            "regions": self.detect_visual_boxes(pil_image),
        }


_detector_instance: Optional[FlorenceVisualDetector] = None


def get_florence_detector() -> Tuple[bool, Optional[FlorenceVisualDetector], str]:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FlorenceVisualDetector()
    return (
        _detector_instance.is_available,
        _detector_instance,
        _detector_instance.status_msg,
    )
