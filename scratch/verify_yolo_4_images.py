"""Verification script: Test YOLOv8 detection and cropping on 4 visual pages."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz
from PIL import Image

from src.yolo_detector import YOLOVisualDetector
from src.storage import IMAGES_DIR, ensure_directories

def run_verification():
    ensure_directories()
    pdf_path = Path("data/uploads/jesc102.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return

    doc = fitz.open(str(pdf_path))
    detector = YOLOVisualDetector()
    print(f"=== YOLOv8 Visual Engine Status: {detector.is_available} ({detector.status_msg}) ===")
    print(f"Testing PDF: {pdf_path.name} (Total Pages: {len(doc)})\n")

    # Select 4 pages with diagrams / apparatus in Chapter 2
    test_pages = [3, 4, 6, 8]  # 1-indexed pages
    success_count = 0

    for p_num in test_pages:
        if p_num > len(doc):
            continue
        page = doc[p_num - 1]
        pix = page.get_pixmap(dpi=200)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        boxes = detector.detect_visual_boxes(pil_img, min_box_area=6000)
        print(f"--- Page {p_num} ---")
        print(f"Detected Visual Regions: {len(boxes)}")

        for idx, b in enumerate(boxes):
            x0, y0, x1, y1 = b["bbox"]
            conf = b["confidence"]
            lbl = b["label"]
            w = b["width"]
            h = b["height"]

            crop = pil_img.crop((x0, y0, x1, y1))
            out_fn = f"verify_yolo_p{p_num}_fig_{idx+1}.png"
            out_path = IMAGES_DIR / out_fn
            crop.save(str(out_path))

            print(f"  [Figure {idx+1}] Label: '{lbl}' | Conf: {conf:.2f} | BBox: ({x0}, {y0}, {x1}, {y1}) | Size: {w}x{h}px -> Saved to: {out_fn}")
            success_count += 1
        print()

    print(f"=== VERIFICATION SUMMARY: Successfully extracted {success_count} visual regions across {len(test_pages)} pages! ===")

if __name__ == "__main__":
    run_verification()
