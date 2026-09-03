"""CLI for figure detection on a single image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.florence_model import load_trained_model
from src.layout.florence_processor import generate_detections
from src.layout.paths import FLORENCE_LAYOUT_DIR


def detect_figures(image: Image.Image, adapter_dir: str | None = None) -> list[dict]:
    model, processor, device = load_trained_model(adapter_dir or FLORENCE_LAYOUT_DIR / "best")
    det = generate_detections(model, processor, image, device)
    results = []
    for bbox, label in zip(det["bboxes"], det["labels"] or ["Picture"] * len(det["bboxes"])):
        if (label or "").strip().lower() not in ("picture", "diagram", "figure", "image", ""):
            continue
        results.append({
            "bbox": [int(b) for b in bbox],
            "label": "Picture",
            "score": None,  # Florence OD does not provide calibrated confidence
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect figures in an image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--adapter", default=str(FLORENCE_LAYOUT_DIR / "best"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    results = detect_figures(img, args.adapter)
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
