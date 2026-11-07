"""Scanned / Image PDF Processing Pipeline.

Specialized exclusively for scanned image pages, photographed documents, and OCR.
Handles:
- High-DPI page rendering to PIL Image
- RapidOCR (or PaddleOCR) text extraction with Tesseract fallback
- Optical caption boundary localization
- OpenCV morphological contour & visual diagram extraction
- Scanned subfigure isolation
"""

import logging
import re
import time
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz
import pytesseract
from PIL import Image

from src.captions import CaptionMatch, FigureNamer, clean_caption_text, nearest_caption, parse_caption
from src.ocr_engine import OCRUnavailable, ocr_image_with_metadata
from src.storage import IMAGES_DIR

_TESSERACT_ERRORS = (pytesseract.TesseractError, pytesseract.TesseractNotFoundError, OSError, RuntimeError)

logger = logging.getLogger(__name__)


class ScannedPipeline:
    """Extractor for scanned pages using OCR, visual detection, and OpenCV."""

    def __init__(
        self,
        ocr_dpi: int = 300,
        render_dpi: int = 200,
        tesseract_ok: bool = True,
        tesseract_msg: str = "",
        visual_detector: Any | None = None,
        use_primary_ocr: bool = True,
        ocr_languages: list | None = None,
    ):
        self.ocr_dpi = ocr_dpi
        self.render_dpi = render_dpi
        self.tesseract_ok = tesseract_ok
        self.tesseract_msg = tesseract_msg
        self.visual_detector = visual_detector
        self.use_primary_ocr = use_primary_ocr
        self.ocr_languages = ocr_languages or ["en"]

    def render_page_image(self, page: fitz.Page, dpi: int = 150) -> Image.Image:
        """Render a PyMuPDF page to a PIL Image at specified DPI."""
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def run_ocr(self, pil_img: Image.Image) -> tuple[str, float, dict[str, Any] | None, str]:
        """Run primary OCR, then use Tesseract only when the result is weak."""
        ocr_start = time.perf_counter()

        if self.use_primary_ocr:
            try:
                primary = ocr_image_with_metadata(
                    pil_img,
                    languages=self.ocr_languages,
                    min_confidence=0.3,
                )
                primary_quality = self._ocr_quality(primary)
                if primary_quality >= 0.70 and primary["text"].strip():
                    ocr_duration = round(time.perf_counter() - ocr_start, 2)
                    ocr_data = self._primary_ocr_to_tesseract_data(primary)
                    engine = primary.get("engine", "PaddleOCR")
                    note = f"{engine} accepted (quality {primary_quality:.2f}, {ocr_duration}s)"
                    return primary["text"], ocr_duration, ocr_data, note
            except (OCRUnavailable, RuntimeError, ValueError, OSError) as exc:
                logger.warning("Primary OCR failed; trying Tesseract: %s", exc)

        if not self.tesseract_ok:
            return "", 0.0, None, f"Tesseract unavailable: {self.tesseract_msg}"

        try:
            ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            ocr_duration = round(time.perf_counter() - ocr_start, 2)

            lines_dict: dict[tuple[int, int, int], list[str]] = {}
            for i in range(len(ocr_data["text"])):
                t = ocr_data["text"][i].strip()
                if not t:
                    continue
                k = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])
                lines_dict.setdefault(k, []).append(t)

            line_texts = [" ".join(words) for words in lines_dict.values()]
            ocr_text = "\n".join(line_texts).strip()
            cleaned_ocr = " ".join(ocr_text.split())

            if len(cleaned_ocr) > 0:
                note = (
                    f"Tesseract selected after primary OCR quality check ({len(cleaned_ocr)} chars in {ocr_duration}s)"
                )
            else:
                note = "Scanned/Image page with no readable text detected by OCR"

            return ocr_text, ocr_duration, ocr_data, note
        except _TESSERACT_ERRORS as e:
            ocr_duration = round(time.perf_counter() - ocr_start, 2)
            return "", ocr_duration, None, f"OCR Error: {e}"

    @staticmethod
    def _ocr_quality(result: dict[str, Any]) -> float:
        """Estimate OCR reliability for fallback routing, not semantic correctness."""
        text = str(result.get("text", "")).strip()
        confidences = [float(value) for value in result.get("confidences", [])]
        if not text or not confidences:
            return 0.0

        confidence_score = sum(confidences) / len(confidences)
        low_confidence_ratio = sum(value < 0.55 for value in confidences) / len(confidences)
        alpha_chars = sum(char.isalnum() for char in text)
        sanity_score = min(1.0, alpha_chars / max(1, len(text) * 0.55))
        return max(0.0, min(1.0, 0.65 * confidence_score + 0.20 * (1 - low_confidence_ratio) + 0.15 * sanity_score))

    @staticmethod
    def _primary_ocr_to_tesseract_data(result: dict[str, Any]) -> dict[str, Any]:
        """Adapt primary OCR detections to the existing caption-box parser contract."""
        boxes = result.get("boxes", [])
        words = result.get("words", [])
        confidences = result.get("confidences", [])
        return {
            "text": words,
            "left": [box[0] for box in boxes],
            "top": [box[1] for box in boxes],
            "width": [box[2] for box in boxes],
            "height": [box[3] for box in boxes],
            "conf": [str(value * 100) for value in confidences],
            "block_num": list(range(len(words))),
            "par_num": [1] * len(words),
            "line_num": list(range(len(words))),
        }

    def _find_drawing_bbox_near_caption(
        self,
        img_np: np.ndarray,
        cap_box: list[int],
        line_boxes: list[list[int]],
        search_dir: str = "above",
        max_search_dist: int = 600,
    ) -> tuple[int, int, int, int] | None:
        """Find the bounding box of a drawing / illustration near an OCR caption."""
        H, W, _ = img_np.shape
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        cx0, cy0, cx1, cy1 = cap_box

        if search_dir == "above":
            sy0 = max(int(H * 0.02), cy0 - max_search_dist)
            sy1 = max(sy0 + 20, cy0 - 5)
            sx0 = max(int(W * 0.02), cx0 - 250)
            sx1 = min(int(W * 0.98), cx1 + 250)
        else:
            sy0 = min(H - 20, cy1 + 5)
            sy1 = min(int(H * 0.98), cy1 + max_search_dist)
            sx0 = max(int(W * 0.02), cx0 - 250)
            sx1 = min(int(W * 0.98), cx1 + 250)

        region = gray[sy0:sy1, sx0:sx1]
        if region.size == 0 or region.shape[0] < 20 or region.shape[1] < 20:
            return None

        mask = np.ones(region.shape, dtype=np.uint8) * 255
        for tb in line_boxes:
            tx0, ty0, tx1, ty1 = tb
            rx0 = max(0, tx0 - sx0)
            ry0 = max(0, ty0 - sy0)
            rx1 = min(region.shape[1], tx1 - sx0)
            ry1 = min(region.shape[0], ty1 - sy0)
            if rx1 > rx0 and ry1 > ry0 and (ty1 - ty0) < 45:
                mask[ry0:ry1, rx0:rx1] = 0

        _, binary = cv2.threshold(region, 215, 255, cv2.THRESH_BINARY_INV)
        binary_masked = cv2.bitwise_and(binary, mask)

        ink_pts = cv2.findNonZero(binary_masked)
        if ink_pts is None or len(ink_pts) < 120:
            ink_pts = cv2.findNonZero(binary)
            if ink_pts is None or len(ink_pts) < 200:
                return None

        bx, by, bw, bh = cv2.boundingRect(ink_pts)
        if bw < 70 or bh < 50:
            return None

        pad = 14
        fx0 = max(0, sx0 + bx - pad)
        fy0 = max(0, sy0 + by - pad)
        fx1 = min(W, sx0 + bx + bw + pad)
        fy1 = min(H, sy0 + by + bh + pad)

        return (fx0, fy0, fx1, fy1)

    def extract_visuals(
        self,
        page: fitz.Page,
        page_number: int,
        pdf_stem: str,
        pil_page_img: Image.Image,
        ocr_data: dict[str, Any] | None,
        is_meaningful_fn,
        parse_subfigs_fn,
        extract_terms_fn,
        figure_factory_fn,
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        """Extract diagrams, figures and captions from a scanned page using YOLOv8, OCR data & OpenCV."""
        img_np = np.array(pil_page_img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        H, W, _ = img_np.shape

        lines: dict[tuple[int, int, int], dict[str, Any]] = {}
        line_list: list[dict[str, Any]] = []
        line_boxes: list[list[int]] = []

        if ocr_data and "text" in ocr_data:
            # Parse OCR lines & bounding boxes
            for i in range(len(ocr_data["text"])):
                t = ocr_data["text"][i].strip()
                if not t:
                    continue
                k = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])
                if k not in lines:
                    lines[k] = {
                        "text": [t],
                        "box": [
                            ocr_data["left"][i],
                            ocr_data["top"][i],
                            ocr_data["left"][i] + ocr_data["width"][i],
                            ocr_data["top"][i] + ocr_data["height"][i],
                        ],
                    }
                else:
                    lines[k]["text"].append(t)
                    lines[k]["box"][0] = min(lines[k]["box"][0], ocr_data["left"][i])
                    lines[k]["box"][1] = min(lines[k]["box"][1], ocr_data["top"][i])
                    lines[k]["box"][2] = max(lines[k]["box"][2], ocr_data["left"][i] + ocr_data["width"][i])
                    lines[k]["box"][3] = max(lines[k]["box"][3], ocr_data["top"][i] + ocr_data["height"][i])

            line_list = [{"text": " ".join(v["text"]), "box": v["box"]} for v in lines.values()]
            line_list.sort(key=lambda item: item["box"][1])
            line_boxes = [ln["box"] for ln in line_list]

        saved_image_paths: list[str] = []
        extracted_figures: list[Any] = []
        figure_refs_found: set[str] = set()
        claimed_boxes: list[tuple[int, int, int, int]] = []
        namer = FigureNamer(IMAGES_DIR, pdf_stem, page_number)
        caption_lines: list[tuple[tuple[float, float, float, float], CaptionMatch]] = []
        for line in line_list:
            parsed = parse_caption(line["text"])
            if parsed is not None:
                caption_lines.append((tuple(line["box"]), parsed))

        def caption_near(box: tuple[float, float, float, float]) -> CaptionMatch | None:
            # Scanned pages are rendered at ~150 dpi, so 90 px is roughly 0.6 inch.
            return nearest_caption(box, caption_lines, max_gap=90.0)

        # 1. Caption-linked Scanned Figure Extraction
        for idx, ln in enumerate(line_list):
            txt = ln["text"]
            m = re.search(r"(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?)\s*([0-9]+(?:\.[0-9]+)?[\w-]*)", txt)
            if not m:
                continue

            fig_id = m.group(1)
            figure_refs_found.add(fig_id)

            # A body sentence that merely mentions "Fig. 12.1" is a reference, not a caption.
            caption_match = parse_caption(txt)
            is_caption_start = caption_match is not None
            if not is_caption_start:
                continue
            # "...as shown in / Fig. 12.2. Why do..." continues the sentence on the line above: not a caption.
            above = [
                p
                for p in line_list[:idx]
                if abs(p["box"][0] - ln["box"][0]) <= 40 and -12 <= ln["box"][1] - p["box"][3] <= 25
            ]
            if above and not re.search(r"[.!?:)]\s*$", above[-1]["text"]):
                continue
            fig_id = f"{caption_match.number}.{caption_match.sub}" if caption_match.sub else caption_match.number

            cap_box = list(ln["box"])
            full_cap_text = txt

            if is_caption_start:
                taken = 0
                for next_idx in range(idx + 1, min(idx + 10, len(line_list))):
                    next_l = line_list[next_idx]
                    if taken >= 3:
                        break
                    # OCR boxes overlap by a few pixels; the caption column must line up (not the next column).
                    aligned = -12 <= next_l["box"][1] - cap_box[3] <= 35 and abs(next_l["box"][0] - cap_box[0]) <= 40
                    if (
                        aligned
                        and len(next_l["text"]) < 90
                        and not re.search(r"^(?:Fig|Activity|Table|[0-9]+\.[0-9]+)", next_l["text"])
                    ):
                        full_cap_text += " " + next_l["text"]
                        taken += 1
                        cap_box[0] = min(cap_box[0], next_l["box"][0])
                        cap_box[2] = max(cap_box[2], next_l["box"][2])
                        cap_box[3] = max(cap_box[3], next_l["box"][3])

            bbox = self._find_drawing_bbox_near_caption(img_np, cap_box, line_boxes, search_dir="above")
            if not bbox:
                bbox = self._find_drawing_bbox_near_caption(img_np, cap_box, line_boxes, search_dir="below")

            if bbox:
                fx0, fy0, fx1, fy1 = bbox
                if any(abs(fx0 - cx0) < 30 and abs(fy0 - cy0) < 30 for cx0, cy0, cx1, cy1 in claimed_boxes):
                    continue

                claimed_boxes.append((fx0, fy0, fx1, fy1))
                crop = img_np[fy0:fy1, fx0:fx1]
                pil_crop = Image.fromarray(crop)

                if is_meaningful_fn(pil_crop):
                    out_path, out_filename = namer.next_path()
                    pil_crop.save(str(out_path))
                    saved_image_paths.append(str(out_path))

                    full_match = parse_caption(full_cap_text)
                    clean_caption = full_match.text if full_match else clean_caption_text(full_cap_text)

                    fig_meta = figure_factory_fn(
                        figure_id=fig_id,
                        figure_label=caption_match.label,
                        subfigure_id=None,
                        caption=clean_caption,
                        page_number=page_number,
                        image_path=str(out_path),
                        image_filename=out_filename,
                        bounding_box=(fx0, fy0, fx1, fy1),
                        source_type="scanned_crop",
                        width=pil_crop.width,
                        height=pil_crop.height,
                        confidence=0.95,
                        associated_keywords=extract_terms_fn(f"{full_cap_text} {clean_caption or ''}"),
                        context=f"Page {page_number} • {caption_match.label}: {clean_caption or ''}",
                    )
                    extracted_figures.append(fig_meta)

                    subfigs = parse_subfigs_fn(full_cap_text)
                    if subfigs and len(subfigs) >= 2:
                        w = fx1 - fx0
                        h = fy1 - fy0
                        is_horizontal = w > h * 1.1
                        for s_idx, (sub_tag, sub_desc) in enumerate(subfigs[:2]):
                            if is_horizontal:
                                mid_x = fx0 + w // 2
                                sbox = (fx0, fy0, mid_x + 10, fy1) if s_idx == 0 else (mid_x - 10, fy0, fx1, fy1)
                            else:
                                mid_y = fy0 + h // 2
                                sbox = (fx0, fy0, fx1, mid_y + 10) if s_idx == 0 else (fx0, mid_y - 10, fx1, fy1)
                            sub_crop = img_np[sbox[1] : sbox[3], sbox[0] : sbox[2]]
                            sub_pil = Image.fromarray(sub_crop)
                            if is_meaningful_fn(sub_pil):
                                sub_out_p, sub_out_fn = namer.next_path()
                                sub_pil.save(str(sub_out_p))
                                saved_image_paths.append(str(sub_out_p))

                                sub_meta = figure_factory_fn(
                                    figure_id=f"{fig_id}.{sub_tag}",
                                    figure_label=f"Figure {fig_id}({sub_tag})",
                                    subfigure_id=sub_tag,
                                    caption=sub_desc,
                                    page_number=page_number,
                                    image_path=str(sub_out_p),
                                    image_filename=sub_out_fn,
                                    bounding_box=sbox,
                                    source_type="scanned_crop",
                                    width=sub_pil.width,
                                    height=sub_pil.height,
                                    confidence=0.92,
                                    associated_keywords=extract_terms_fn(f"{full_cap_text} {sub_desc}"),
                                    context=f"Page {page_number} • Figure {fig_id}({sub_tag}): {sub_desc}",
                                )
                                extracted_figures.append(sub_meta)

        # 2. Florence-2 Visual Region Detection
        if self.visual_detector and self.visual_detector.is_available:
            try:
                florence_regions = self.visual_detector.detect_visual_boxes(pil_page_img)
                for yr in florence_regions:
                    yx0, yy0, yx1, yy1 = yr["bbox"]
                    overlap = False
                    for cx0, cy0, cx1, cy1 in claimed_boxes:
                        ox0 = max(yx0, cx0)
                        oy0 = max(yy0, cy0)
                        ox1 = min(yx1, cx1)
                        oy1 = min(yy1, cy1)
                        if (
                            ox1 > ox0
                            and oy1 > oy0
                            and (ox1 - ox0) * (oy1 - oy0)
                            > 0.35 * min((yx1 - yx0) * (yy1 - yy0), (cx1 - cx0) * (cy1 - cy0))
                        ):
                            overlap = True
                            break
                    if overlap:
                        continue

                    crop_pil = pil_page_img.crop((yx0, yy0, yx1, yy1))
                    if not is_meaningful_fn(crop_pil):
                        continue

                    claimed_boxes.append((yx0, yy0, yx1, yy1))
                    y_idx = len(extracted_figures) + 1
                    out_path, out_filename = namer.next_path()
                    crop_pil.save(str(out_path))
                    saved_image_paths.append(str(out_path))

                    near = caption_near((yx0, yy0, yx1, yy1))
                    fig_label = near.label if near else f"Page {page_number} visual {y_idx}"
                    fig_meta = figure_factory_fn(
                        figure_id=near.number if near else f"florence_{page_number}_{y_idx}",
                        figure_label=fig_label,
                        subfigure_id=None,
                        caption=near.text if near else None,
                        page_number=page_number,
                        image_path=str(out_path),
                        image_filename=out_filename,
                        bounding_box=(yx0, yy0, yx1, yy1),
                        source_type="florence_crop",
                        width=crop_pil.width,
                        height=crop_pil.height,
                        confidence=yr.get("confidence", 0.92),
                        associated_keywords=extract_terms_fn(f"{(near.text if near else '')} {yr.get('label') or ''}"),
                        context=f"Page {page_number} • Visual: {(near.text if near else fig_label)}",
                    )
                    extracted_figures.append(fig_meta)
            except (RuntimeError, ValueError, OSError) as exc:
                logger.error("Florence scanned-region detection failed on page %s: %s", page_number, exc)

        # 3. Standalone Visual Diagram & Activity Detection (OpenCV Contours)
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
            margin_x = int(W * 0.04)
            margin_y = int(H * 0.06)
            binary[:margin_y, :] = 0
            binary[H - margin_y :, :] = 0
            binary[:, :margin_x] = 0
            binary[:, W - margin_x :] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            diag_idx = 1
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cy < H * 0.07 or (cy + ch) > H * 0.93:
                    continue
                if cw < 120 or ch < 90 or cw * ch < 20_000:
                    continue
                if cw > W * 0.88 and ch > H * 0.88:
                    continue
                if (cw > 350 and ch < 65) or (ch > 350 and cw < 65):
                    continue
                if (cw / ch > 4.0) or (ch / cw > 4.0):
                    continue

                overlap = False
                for fx0, fy0, fx1, fy1 in claimed_boxes:
                    ox0 = max(cx, fx0)
                    oy0 = max(cy, fy0)
                    ox1 = min(cx + cw, fx1)
                    oy1 = min(cy + ch, fy1)
                    if ox1 > ox0 and oy1 > oy0:
                        int_area = (ox1 - ox0) * (oy1 - oy0)
                        if int_area > 0.35 * min(cw * ch, (fx1 - fx0) * (fy1 - fy0)):
                            overlap = True
                            break
                if overlap:
                    continue

                crop_img = img_np[cy : cy + ch, cx : cx + cw]
                crop_pil = Image.fromarray(crop_img)
                if not is_meaningful_fn(crop_pil):
                    continue

                near = caption_near((cx, cy, cx + cw, cy + ch))
                claimed_boxes.append((cx, cy, cx + cw, cy + ch))
                out_path, out_filename = namer.next_path()
                crop_pil.save(str(out_path))
                saved_image_paths.append(str(out_path))

                diag_label = near.label if near else f"Page {page_number} diagram {diag_idx}"
                diag_meta = figure_factory_fn(
                    figure_id=near.number if near else f"diag_{page_number}_{diag_idx}",
                    figure_label=diag_label,
                    subfigure_id=None,
                    caption=near.text if near else None,
                    page_number=page_number,
                    image_path=str(out_path),
                    image_filename=out_filename,
                    bounding_box=(cx, cy, cx + cw, cy + ch),
                    source_type="scanned_crop",
                    width=crop_pil.width,
                    height=crop_pil.height,
                    confidence=0.88,
                    associated_keywords=extract_terms_fn(near.text if near and near.text else ""),
                    context=f"Page {page_number} • Diagram: {(near.text if near else diag_label)}",
                )
                extracted_figures.append(diag_meta)
                diag_idx += 1
        except (cv2.error, ValueError) as exc:
            logger.error("OpenCV scanned-visual extraction failed on page %s: %s", page_number, exc)

        figure_dicts = [f.to_dict() if hasattr(f, "to_dict") else f for f in extracted_figures]
        sorted_refs = sorted(figure_refs_found)
        return saved_image_paths, figure_dicts, sorted_refs
