"""
Canonical bounding-box utilities for Florence-2 <OD> task.

Verified against microsoft/Florence-2-base processing_florence2.py:
  - Token order: Label<loc_x1><loc_y1><loc_x2><loc_y2>
  - Quantization: floor(x / (width/1000)), clamped to [0, 999]
  - Coordinate system: original image pixels (x1,y1,x2,y2) before processor resize
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

LOC_BINS = 1000
TASK_PROMPT = "<OD>"
TARGET_CLASS = "Picture"


class BBoxError(ValueError):
    pass


def validate_bbox_xyxy(bbox: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise BBoxError(f"bbox must have 4 values, got {len(bbox)}")

    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    if x2 < x1 or y2 < y1:
        raise BBoxError(f"invalid bbox order: {bbox}")
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise BBoxError(f"zero-area bbox: {bbox}")

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    if x2 <= x1 or y2 <= y1:
        raise BBoxError(f"degenerate bbox after clamp: {bbox}")

    return x1, y1, x2, y2


def bbox_xyxy_to_florence(bbox: Sequence[float], width: int, height: int) -> str:
    """Convert pixel xyxy bbox to Florence location token string (x1,y1,x2,y2 order)."""
    x1, y1, x2, y2 = validate_bbox_xyxy(bbox, width, height)

    # Florence BoxQuantizer uses floor, not round
    nx1 = max(0, min(LOC_BINS - 1, math.floor(x1 / (width / LOC_BINS))))
    ny1 = max(0, min(LOC_BINS - 1, math.floor(y1 / (height / LOC_BINS))))
    nx2 = max(0, min(LOC_BINS - 1, math.floor(x2 / (width / LOC_BINS))))
    ny2 = max(0, min(LOC_BINS - 1, math.floor(y2 / (height / LOC_BINS))))

    return f"<loc_{nx1}><loc_{ny1}><loc_{nx2}><loc_{ny2}>"


def sort_picture_annotations(annotations: list) -> list:
    """Sort Picture boxes top-to-bottom, then left-to-right for deterministic targets."""
    pics = [a for a in annotations if a.get("category") == TARGET_CLASS]
    pics.sort(key=lambda a: (a["bbox"][1], a["bbox"][0]))
    return pics


def build_od_target(annotations: list, width: int, height: int) -> str:
    """Build a Florence <OD> target; an empty string represents a negative page."""
    parts: list[str] = []
    for ann in sort_picture_annotations(annotations):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        loc = bbox_xyxy_to_florence(bbox, width, height)
        parts.append(f"{TARGET_CLASS}{loc}")
    return "".join(parts)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def box_area_ratio(bbox: Sequence[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = validate_bbox_xyxy(bbox, width, height)
    return ((x2 - x1) * (y2 - y1)) / max(1, width * height)
