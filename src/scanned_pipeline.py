"""Scanned / Image PDF Processing Pipeline.

Specialized exclusively for scanned image pages, photographed documents, and OCR.
Handles:
- High-DPI page rendering to PIL Image
- EasyOCR text extraction (primary, 80+ languages) with Tesseract fallback
- Optical caption boundary localization
- OpenCV morphological contour & visual diagram extraction
- Scanned subfigure isolation
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import pymupdf as fitz
import numpy as np
from PIL import Image
import pytesseract

from src.ocr_engine import ocr_image

from src.storage import IMAGES_DIR

logger = logging.getLogger(__name__)


class ScannedPipeline:
    """Standalone extractor pipeline for Scanned (image-based) PDF pages using YOLOv8, OCR and OpenCV."""

    def __init__(
        self,
        ocr_dpi: int = 300,
        render_dpi: int = 200,
        tesseract_ok: bool = True,
        tesseract_msg: str = "",
        visual_detector: Optional[Any] = None,
        use_easyocr: bool = True,
        ocr_languages: Optional[list] = None,
    ):
        self.ocr_dpi = ocr_dpi
        self.render_dpi = render_dpi
        self.tesseract_ok = tesseract_ok
        self.tesseract_msg = tesseract_msg
        self.visual_detector = visual_detector
        self.use_easyocr = use_easyocr
        self.ocr_languages = ocr_languages or ["en"]

    def render_page_image(self, page: fitz.Page, dpi: int = 150) -> Image.Image:
        """Render a PyMuPDF page to a PIL Image at specified DPI."""
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def run_ocr(self, pil_img: Image.Image) -> Tuple[str, float, Optional[Dict[str, Any]], str]:
        """Perform OCR on a rendered page image using EasyOCR (primary) or Tesseract (fallback).

        EasyOCR provides better accuracy (~97%) and supports 80+ languages.
        Tesseract is used as fallback if EasyOCR is unavailable.
        Returns: (text, duration, ocr_data_dict, note)
        """
        ocr_start = time.perf_counter()

        # ── Primary: EasyOCR ─────────────────────────────────────────────────
        if self.use_easyocr:
            try:
                import easyocr
                easy_text = ocr_image(
                    pil_img,
                    prefer_easyocr=True,
                    languages=self.ocr_languages,
                    min_confidence=0.3,
                )
                ocr_duration = round(time.perf_counter() - ocr_start, 2)
                if easy_text.strip():
                    note = f"EasyOCR extracted {len(easy_text)} chars in {ocr_duration}s"
                    # Build a compatible ocr_data stub for downstream box-parsing
                    ocr_data = None  # EasyOCR path skips box-based caption detection
                    return easy_text, ocr_duration, ocr_data, note
            except Exception:
                pass  # Fall through to Tesseract

        # ── Fallback: Tesseract ───────────────────────────────────────────────
        if not self.tesseract_ok:
            return "", 0.0, None, f"Tesseract unavailable: {self.tesseract_msg}"

        try:
            ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            ocr_duration = round(time.perf_counter() - ocr_start, 2)

            lines_dict: Dict[Tuple[int, int, int], List[str]] = {}
            for i in range(len(ocr_data['text'])):
                t = ocr_data['text'][i].strip()
                if not t:
                    continue
                k = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
                lines_dict.setdefault(k, []).append(t)

            line_texts = [" ".join(words) for words in lines_dict.values()]
            ocr_text = "\n".join(line_texts).strip()
            cleaned_ocr = " ".join(ocr_text.split())

            if len(cleaned_ocr) > 0:
                note = f"Tesseract OCR extracted {len(cleaned_ocr)} chars in {ocr_duration}s"
            else:
                note = "Scanned/Image page with no readable text detected by OCR"

            return ocr_text, ocr_duration, ocr_data, note
        except Exception as e:
            ocr_duration = round(time.perf_counter() - ocr_start, 2)
            return "", ocr_duration, None, f"OCR Error: {e}"

    def _find_drawing_bbox_near_caption(
        self,
        img_np: np.ndarray,
        cap_box: List[int],
        line_boxes: List[List[int]],
        search_dir: str = 'above',
        max_search_dist: int = 600,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find the bounding box of a drawing / illustration near an OCR caption."""
        H, W, _ = img_np.shape
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        cx0, cy0, cx1, cy1 = cap_box

        if search_dir == 'above':
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
        ocr_data: Optional[Dict[str, Any]],
        is_meaningful_fn,
        parse_subfigs_fn,
        extract_terms_fn,
        figure_factory_fn,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Extract diagrams, figures and captions from a scanned page using YOLOv8, OCR data & OpenCV."""
        img_np = np.array(pil_page_img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        H, W, _ = img_np.shape

        lines: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        line_list: List[Dict[str, Any]] = []
        line_boxes: List[List[int]] = []

        if ocr_data and 'text' in ocr_data:
            # Parse OCR lines & bounding boxes
            for i in range(len(ocr_data['text'])):
                t = ocr_data['text'][i].strip()
                if not t:
                    continue
                k = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
                if k not in lines:
                    lines[k] = {
                        'text': [t],
                        'box': [
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['left'][i] + ocr_data['width'][i],
                            ocr_data['top'][i] + ocr_data['height'][i],
                        ],
                    }
                else:
                    lines[k]['text'].append(t)
                    lines[k]['box'][0] = min(lines[k]['box'][0], ocr_data['left'][i])
                    lines[k]['box'][1] = min(lines[k]['box'][1], ocr_data['top'][i])
                    lines[k]['box'][2] = max(lines[k]['box'][2], ocr_data['left'][i] + ocr_data['width'][i])
                    lines[k]['box'][3] = max(lines[k]['box'][3], ocr_data['top'][i] + ocr_data['height'][i])

            line_list = [{'text': ' '.join(v['text']), 'box': v['box']} for v in lines.values()]
            line_list.sort(key=lambda item: item['box'][1])
            line_boxes = [l['box'] for l in line_list]

        saved_image_paths: List[str] = []
        extracted_figures: List[Any] = []
        figure_refs_found: Set[str] = set()
        claimed_boxes: List[Tuple[int, int, int, int]] = []

        # 1. Caption-linked Scanned Figure Extraction
        for idx, l in enumerate(line_list):
            txt = l['text']
            m = re.search(r'(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?)\s*([0-9]+(?:\.[0-9]+)?[\w-]*)', txt)
            if not m:
                continue

            fig_id = m.group(1)
            figure_refs_found.add(fig_id)

            is_caption_start = bool(re.search(r'^\s*(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?)\s*[0-9]+', txt))
            if not is_caption_start and len(txt) < 80 and ('fig' in txt.lower() or 'figure' in txt.lower()):
                is_caption_start = True

            cap_box = list(l['box'])
            full_cap_text = txt

            if is_caption_start:
                for next_idx in range(idx + 1, min(idx + 4, len(line_list))):
                    next_l = line_list[next_idx]
                    if 0 <= next_l['box'][1] - cap_box[3] <= 35 and abs(next_l['box'][0] - cap_box[0]) <= 250:
                        if not re.search(r'^(?:Fig|Activity|[0-9]+\.[0-9]+)', next_l['text']):
                            full_cap_text += ' ' + next_l['text']
                            cap_box[0] = min(cap_box[0], next_l['box'][0])
                            cap_box[2] = max(cap_box[2], next_l['box'][2])
                            cap_box[3] = max(cap_box[3], next_l['box'][3])

            bbox = self._find_drawing_bbox_near_caption(img_np, cap_box, line_boxes, search_dir='above')
            if not bbox:
                bbox = self._find_drawing_bbox_near_caption(img_np, cap_box, line_boxes, search_dir='below')

            if bbox:
                fx0, fy0, fx1, fy1 = bbox
                if any(abs(fx0 - cx0) < 30 and abs(fy0 - cy0) < 30 for cx0, cy0, cx1, cy1 in claimed_boxes):
                    continue

                claimed_boxes.append((fx0, fy0, fx1, fy1))
                crop = img_np[fy0:fy1, fx0:fx1]
                pil_crop = Image.fromarray(crop)

                if is_meaningful_fn(pil_crop):
                    safe_fig_id = fig_id.replace('.', '_')
                    out_filename = f"{pdf_stem}_p{page_number}_fig{safe_fig_id}.png"
                    out_path = IMAGES_DIR / out_filename
                    pil_crop.save(str(out_path))
                    saved_image_paths.append(str(out_path))

                    clean_caption = re.sub(r'^(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?)\s*[0-9.]+\s*[:.\-]?\s*', '', full_cap_text).strip()
                    if not clean_caption:
                        clean_caption = full_cap_text

                    fig_meta = figure_factory_fn(
                        figure_id=fig_id,
                        figure_label=f"Figure {fig_id}",
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
                        associated_keywords=extract_terms_fn(f"{full_cap_text} {clean_caption}"),
                        context=f"Page {page_number} • Figure {fig_id}: {clean_caption}",
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
                            sub_crop = img_np[sbox[1]:sbox[3], sbox[0]:sbox[2]]
                            sub_pil = Image.fromarray(sub_crop)
                            if is_meaningful_fn(sub_pil):
                                sub_out_fn = f"{pdf_stem}_p{page_number}_fig{safe_fig_id}_{sub_tag}.png"
                                sub_out_p = IMAGES_DIR / sub_out_fn
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
                        if ox1 > ox0 and oy1 > oy0:
                            if (ox1 - ox0) * (oy1 - oy0) > 0.35 * min((yx1 - yx0) * (yy1 - yy0), (cx1 - cx0) * (cy1 - cy0)):
                                overlap = True
                                break
                    if overlap:
                        continue

                    crop_pil = pil_page_img.crop((yx0, yy0, yx1, yy1))
                    if not is_meaningful_fn(crop_pil):
                        continue

                    claimed_boxes.append((yx0, yy0, yx1, yy1))
                    y_idx = len(extracted_figures) + 1
                    out_filename = f"{pdf_stem}_p{page_number}_florence_fig_{y_idx}.png"
                    out_path = IMAGES_DIR / out_filename
                    crop_pil.save(str(out_path))
                    saved_image_paths.append(str(out_path))

                    florence_label = (yr.get("label") or "").strip()
                    nearby_lines = [
                        l['text'] for l in line_list
                        if abs(l['box'][1] - yy0) <= 200 or abs(l['box'][3] - yy1) <= 200
                    ]
                    desc_text = florence_label if florence_label else (" ".join(nearby_lines[:2]) if nearby_lines else f"Page {page_number} Visual")
                    fig_label = f"Figure ({florence_label[:45]})" if florence_label else f"Page {page_number} Visual"
                    fig_meta = figure_factory_fn(
                        figure_id=f"florence_{page_number}_{y_idx}",
                        figure_label=fig_label,
                        subfigure_id=None,
                        caption=desc_text,
                        page_number=page_number,
                        image_path=str(out_path),
                        image_filename=out_filename,
                        bounding_box=(yx0, yy0, yx1, yy1),
                        source_type="florence_crop",
                        width=crop_pil.width,
                        height=crop_pil.height,
                        confidence=yr.get("confidence", 0.92),
                        associated_keywords=extract_terms_fn(f"{desc_text} {florence_label}"),
                        context=f"Page {page_number} • Visual: {desc_text}",
                    )
                    extracted_figures.append(fig_meta)
            except Exception:
                logger.exception("Florence scanned-region detection failed on page %s", page_number)
                pass

        # 3. Standalone Visual Diagram & Activity Detection (OpenCV Contours)
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
            margin_x = int(W * 0.04)
            margin_y = int(H * 0.06)
            binary[:margin_y, :] = 0
            binary[H - margin_y:, :] = 0
            binary[:, :margin_x] = 0
            binary[:, W - margin_x:] = 0

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

                crop_img = img_np[cy:cy + ch, cx:cx + cw]
                crop_pil = Image.fromarray(crop_img)
                if not is_meaningful_fn(crop_pil):
                    continue

                nearby_lines = [
                    l['text'] for l in line_list
                    if abs(l['box'][1] - cy) <= 180 or abs(l['box'][3] - (cy + ch)) <= 180
                ]
                diag_label_text = ""
                for nl in nearby_lines:
                    if any(k in nl.lower() for k in ['activity', 'rule', 'thumb', 'left-hand', 'corkscrew', 'solenoid', 'circuit', 'magnet']):
                        diag_label_text = nl
                        break
                if not diag_label_text and nearby_lines:
                    diag_label_text = nearby_lines[0]

                if re.search(r'\.pdf|copilot|watermark', diag_label_text, re.IGNORECASE):
                    diag_label_text = f"Page {page_number} Visual Diagram"

                claimed_boxes.append((cx, cy, cx + cw, cy + ch))
                out_filename = f"{pdf_stem}_p{page_number}_diagram_{diag_idx}.png"
                out_path = IMAGES_DIR / out_filename
                crop_pil.save(str(out_path))
                saved_image_paths.append(str(out_path))

                desc_text = diag_label_text or f"Page {page_number} Visual Diagram"
                diag_meta = figure_factory_fn(
                    figure_id=f"diag_{page_number}_{diag_idx}",
                    figure_label=f"Page {page_number} Diagram",
                    subfigure_id=None,
                    caption=desc_text,
                    page_number=page_number,
                    image_path=str(out_path),
                    image_filename=out_filename,
                    bounding_box=(cx, cy, cx + cw, cy + ch),
                    source_type="scanned_crop",
                    width=crop_pil.width,
                    height=crop_pil.height,
                    confidence=0.88,
                    associated_keywords=extract_terms_fn(desc_text),
                    context=f"Page {page_number} • Diagram: {desc_text}",
                )
                extracted_figures.append(diag_meta)
                diag_idx += 1
        except Exception:
            logger.exception("OpenCV scanned-visual extraction failed on page %s", page_number)
            pass

        figure_dicts = [f.to_dict() if hasattr(f, "to_dict") else f for f in extracted_figures]
        sorted_refs = sorted(list(figure_refs_found))
        return saved_image_paths, figure_dicts, sorted_refs

