"""YOLOv8 Document Layout & Visual Region Detector.

Provides deep learning-based detection of illustrations, figures, diagrams, and tables
from rendered textbook pages using YOLOv8 Ultralytics models.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

# Global cached model instance
_YOLO_MODEL = None
_YOLO_INITIALIZED = False
_YOLO_AVAILABLE = False
_YOLO_ERROR_MSG = ""


def get_yolo_detector(model_name: str = "yolov8n.pt") -> Tuple[bool, Optional[Any], str]:
    """Load or retrieve the cached YOLOv8 model.

    Returns:
        (is_available, model_instance, status_message)
    """
    global _YOLO_MODEL, _YOLO_INITIALIZED, _YOLO_AVAILABLE, _YOLO_ERROR_MSG

    if _YOLO_INITIALIZED:
        return _YOLO_AVAILABLE, _YOLO_MODEL, _YOLO_ERROR_MSG

    try:
        from ultralytics import YOLO

        model = YOLO(model_name)
        _YOLO_MODEL = model
        _YOLO_AVAILABLE = True
        _YOLO_INITIALIZED = True
        _YOLO_ERROR_MSG = f"YOLOv8 visual detector ready ({model_name})"
        return True, _YOLO_MODEL, _YOLO_ERROR_MSG
    except Exception as e:
        _YOLO_AVAILABLE = False
        _YOLO_INITIALIZED = True
        _YOLO_ERROR_MSG = f"YOLOv8 initialization error: {e}"
        return False, None, _YOLO_ERROR_MSG


class YOLOVisualDetector:
    """Detects visual diagrams, apparatus setups, figures, and tables on document pages using YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.is_available, self.model, self.status_msg = get_yolo_detector(model_name)

    def detect_visual_boxes(
        self,
        pil_image: Image.Image,
        min_box_area: int = 5000,
        max_page_coverage: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """Detect visual diagram/picture regions on a page image.

        Args:
            pil_image: PIL RGB image of the rendered page.
            min_box_area: Minimum bounding box pixel area.
            max_page_coverage: Maximum fraction of the entire page to avoid whole-page bounding boxes.

        Returns:
            List of detected visual boxes with coordinates and confidence.
        """
        if not self.is_available or self.model is None:
            return []

        try:
            results = self.model.predict(pil_image, conf=self.conf_threshold, verbose=False)
            detected_regions: List[Dict[str, Any]] = []
            img_w, img_h = pil_image.size
            page_area = img_w * img_h

            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue

                names = getattr(r, "names", {})
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    raw_label = str(names.get(cls_id, cls_id)).lower()
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()
                    x0, y0, x1, y1 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

                    w = x1 - x0
                    h = y1 - y0
                    box_area = w * h

                    # Reject tiny specks or whole-page bounding boxes
                    if box_area < min_box_area or box_area > (page_area * max_page_coverage):
                        continue

                    detected_regions.append({
                        "label": raw_label,
                        "confidence": round(conf, 3),
                        "bbox": (x0, y0, x1, y1),
                        "width": w,
                        "height": h,
                    })

            detected_regions.sort(key=lambda item: item["confidence"], reverse=True)
            return detected_regions
        except Exception:
            return []
