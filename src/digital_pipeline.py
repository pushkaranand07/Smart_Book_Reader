"""Digital PDF Processing Pipeline.

Specialized exclusively for digital/native PDFs with embedded text layers.
Handles:
- Direct native text & font block extraction via PyMuPDF
- Embedded raster image extraction (with quality & mask filtering)
- Vector drawing path detection & clustering
- Hard-boundary spatial stops preventing figure crop collisions
- Broadened caption & diagram label parsing (Figure, Activity, Table, Diagram, etc.)
- Adaptive obstacle-aware dynamic padding (5% relative, clamped 10-30pt)
- High-DPI rendered region crops (300 DPI)
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pymupdf as fitz
from PIL import Image

from src.storage import IMAGES_DIR

CAPTION_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?|Table|Tab\.?|Chart|Activity|Diagram|Exhibit)\s*'
    r'([0-9]+(?:\.[0-9]+)?[\w-]*)',
    re.IGNORECASE
)


def is_qr_code(pil_img: Image.Image, page_num: int) -> bool:
    """Identify square chapter QR codes or barcodes on page 1 / book headers."""
    w, h = pil_img.size
    aspect = w / h if h > 0 else 0
    if 0.85 <= aspect <= 1.15 and page_num == 1 and (w < 400 and h < 400):
        return True
    return False


def parse_subfigures_from_caption(caption_text: str) -> List[Tuple[str, str]]:
    """Parse subfigure tags and descriptions from a caption string."""
    matches = re.findall(r'\(([a-zA-Z0-9]+)\)\s*([^()]+)', caption_text)
    subfigs: List[Tuple[str, str]] = []
    if matches:
        for tag, text in matches:
            clean_tag = tag.strip().lower()
            clean_desc = text.strip().rstrip(".,;")
            clean_desc = re.sub(r'\s+and$', '', clean_desc, flags=re.IGNORECASE).strip()
            if clean_desc:
                subfigs.append((clean_tag, clean_desc))
    return subfigs


class DigitalPipeline:
    """Standalone extractor pipeline for Digital (text-based) PDF pages."""

    def __init__(
        self,
        render_dpi: int = 200,
        pad_pt: float = 20.0,
        yolo_detector: Optional[Any] = None,
    ):
        self.render_dpi = render_dpi
        self.pad_pt = pad_pt
        self.yolo_detector = yolo_detector

    def crop_page_region(
        self,
        page: fitz.Page,
        bbox: fitz.Rect,
        is_meaningful_fn,
        obstacles: Optional[List[fitz.Rect]] = None,
    ) -> Optional[Image.Image]:
        """Render page at specified DPI and crop the dynamic padded bounding box with obstacle avoidance."""
        page_rect = page.rect
        w = bbox.width
        h = bbox.height

        # Percentage-based padding clamped between 10pt and 30pt
        pad_x = max(10.0, min(30.0, w * 0.05))
        pad_y = max(10.0, min(30.0, h * 0.05))

        x0 = max(0.0, bbox.x0 - pad_x)
        y0 = max(0.0, bbox.y0 - pad_y)
        x1 = min(page_rect.width, bbox.x1 + pad_x)
        y1 = min(page_rect.height, bbox.y1 + pad_y)

        # Check obstacle collisions against adjacent body text paragraphs
        if obstacles:
            for obs in obstacles:
                # If obstacle is directly above
                if obs.y1 <= bbox.y0 and obs.y1 > y0:
                    y0 = max(0.0, obs.y1 + 2.0)
                # If obstacle is directly below
                if obs.y0 >= bbox.y1 and obs.y0 < y1:
                    y1 = min(page_rect.height, obs.y0 - 2.0)
                # If obstacle is directly to the left
                if obs.x1 <= bbox.x0 and obs.x1 > x0:
                    x0 = max(0.0, obs.x1 + 2.0)
                # If obstacle is directly to the right
                if obs.x0 >= bbox.x1 and obs.x0 < x1:
                    x1 = min(page_rect.width, obs.x0 - 2.0)

        if x1 <= x0 + 10 or y1 <= y0 + 10:
            return None

        zoom = self.render_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if is_meaningful_fn(img):
            return img
        return None

    def _extract_embedded_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_rect: fitz.Rect,
        page_number: int,
        is_meaningful_fn,
    ) -> List[Dict[str, Any]]:
        """Extract embedded raster images from digital page."""
        valid_images: List[Dict[str, Any]] = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
                image_bytes = base_img.get("image")
                if not image_bytes:
                    continue

                pil_img = Image.open(io.BytesIO(image_bytes))
                if pil_img.mode == "CMYK":
                    pil_img = pil_img.convert("RGB")

                if not is_meaningful_fn(pil_img):
                    continue
                if is_qr_code(pil_img, page_number):
                    continue

                rects = list(page.get_image_rects(xref))
                bbox = rects[0] if rects else fitz.Rect(0, 0, page_rect.width, page_rect.height)

                if (
                    bbox.width >= page_rect.width * 0.90
                    and bbox.height >= page_rect.height * 0.90
                ):
                    continue

                valid_images.append({
                    "xref": xref,
                    "img_idx": img_idx + 1,
                    "bbox": bbox,
                    "pil_img": pil_img,
                    "ext": base_img.get("ext", "png"),
                    "raw_bytes": image_bytes,
                })
            except Exception:
                continue

        return valid_images

    def _extract_vector_drawings(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
    ) -> List[fitz.Rect]:
        """Extract vector drawing boundaries, excluding margins and headers."""
        valid_drawings: List[fitz.Rect] = []
        try:
            raw_drawings = page.get_drawings()
            for d in raw_drawings:
                r = d["rect"]
                if r.x1 <= 0 or r.y1 <= 0 or r.x0 >= page_rect.width or r.y0 >= page_rect.height:
                    continue
                if r.width >= page_rect.width * 0.95 and r.height >= page_rect.height * 0.95:
                    continue
                if r.y0 <= 15 or r.y1 >= page_rect.height - 15:
                    continue
                valid_drawings.append(r)
        except Exception:
            pass
        return valid_drawings

    def _detect_captions(self, blocks: List[Any], clean_text_fn) -> List[Dict[str, Any]]:
        """Detect figure and diagram captions in digital text blocks."""
        captions: List[Dict[str, Any]] = []
        for b in blocks:
            if b[6] != 0:  # text blocks only
                continue
            raw_text = b[4].strip()
            cleaned = clean_text_fn(raw_text)

            if len(cleaned) <= 380:
                match = CAPTION_PATTERN.search(raw_text) or CAPTION_PATTERN.search(cleaned)
                if match:
                    fig_id = match.group(1)
                    matched_prefix = match.group(0).strip()
                    captions.append({
                        "fig_id": fig_id,
                        "prefix": matched_prefix,
                        "bbox": fitz.Rect(b[:4]),
                        "text": cleaned,
                        "raw_block": b,
                        "matched": True,
                    })
        return captions

    def _detect_subfig_labels(self, blocks: List[Any]) -> List[Dict[str, Any]]:
        """Detect standalone subfigure labels like (a), (b)."""
        subfig_labels: List[Dict[str, Any]] = []
        for b in blocks:
            txt = b[4].strip()
            for m in re.finditer(r'\(([a-zA-Z0-9]+)\)', txt):
                subfig_labels.append({
                    "label": m.group(1).lower(),
                    "bbox": fitz.Rect(b[:4]),
                })
        return subfig_labels

    def _extract_nearby_context(self, page_text: str, label_pattern: str) -> str:
        """Extract surrounding text sentences within 350 characters of a figure mention."""
        match = re.search(rf'\b{re.escape(label_pattern)}\b', page_text, re.IGNORECASE)
        if not match:
            return page_text[:400].replace("\n", " ").strip()
        pos = match.start()
        start = max(0, pos - 250)
        end = min(len(page_text), pos + 350)
        return page_text[start:end].replace("\n", " ").strip()

    def get_nearby_visuals(
        self,
        c_bbox: fitz.Rect,
        valid_drawings: List[fitz.Rect],
        valid_images: List[Dict[str, Any]],
        all_captions_on_page: List[Dict[str, Any]],
        max_vertical: float = 220.0,
        max_horizontal: float = 350.0,
    ) -> Tuple[List[fitz.Rect], List[Dict[str, Any]], Optional[float], Optional[float]]:
        """Collect visuals within 220pt constrained by hard above/below caption stop boundaries."""
        # Find hard stops (other captions on this page)
        above_stop = max(
            (cap["bbox"].y1 for cap in all_captions_on_page if cap["bbox"].y1 < c_bbox.y0),
            default=None
        )
        below_stop = min(
            (cap["bbox"].y0 for cap in all_captions_on_page if cap["bbox"].y0 > c_bbox.y1),
            default=None
        )

        fig_drawings: List[fitz.Rect] = []
        for r in valid_drawings:
            # Visual is above caption (caption sits below)
            if 0 <= (c_bbox.y0 - r.y1) <= max_vertical and abs(r.x0 - c_bbox.x0) <= max_horizontal:
                if above_stop is None or r.y0 >= above_stop:
                    fig_drawings.append(r)
            # Visual is below caption (caption sits above)
            elif 0 <= (r.y0 - c_bbox.y1) <= max_vertical and abs(r.x0 - c_bbox.x0) <= max_horizontal:
                if below_stop is None or r.y1 <= below_stop:
                    fig_drawings.append(r)

        fig_images: List[Dict[str, Any]] = []
        for img in valid_images:
            r = img["bbox"]
            # Image is above caption
            if 0 <= (c_bbox.y0 - r.y1) <= max_vertical and abs(r.x0 - c_bbox.x0) <= max_horizontal:
                if above_stop is None or r.y0 >= above_stop:
                    fig_images.append(img)
            # Image is below caption
            elif 0 <= (r.y0 - c_bbox.y1) <= max_vertical and abs(r.x0 - c_bbox.x0) <= max_horizontal:
                if below_stop is None or r.y1 <= below_stop:
                    fig_images.append(img)

        return fig_drawings, fig_images, above_stop, below_stop

    def extract_visuals(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
        pdf_stem: str,
        is_meaningful_fn,
        clean_text_fn,
        extract_terms_fn,
        figure_factory_fn,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Run digital visual extraction pipeline for a single page."""
        page_rect = page.rect
        blocks = page.get_text("blocks")
        full_page_text = page.get_text("text")

        valid_images = self._extract_embedded_images(doc, page, page_rect, page_number, is_meaningful_fn)
        valid_drawings = self._extract_vector_drawings(page, page_rect)
        captions = self._detect_captions(blocks, clean_text_fn)
        subfig_labels = self._detect_subfig_labels(blocks)

        figure_refs_found: Set[str] = set()
        for ref_match in re.finditer(
            r'(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?|Table|Activity|Chart)\s*([0-9]+(?:\.[0-9]+)?)',
            full_page_text,
            re.IGNORECASE,
        ):
            figure_refs_found.add(ref_match.group(1))

        # Collect non-caption body text blocks to act as obstacles against bleed-over
        caption_rects = [cap["bbox"] for cap in captions]
        body_obstacles: List[fitz.Rect] = []
        for b in blocks:
            if b[6] == 0:  # text block
                b_rect = fitz.Rect(b[:4])
                if len(b[4].strip()) > 80 and not any(b_rect.intersects(cr) for cr in caption_rects):
                    body_obstacles.append(b_rect)

        saved_image_paths: List[str] = []
        extracted_figures: List[Any] = []

        for cap in captions:
            c_id = cap["fig_id"]
            c_bbox = cap["bbox"]
            c_text = cap["text"]
            subfigs_in_cap = parse_subfigures_from_caption(c_text)
            figure_refs_found.add(c_id)

            fig_drawings, fig_images, above_stop, below_stop = self.get_nearby_visuals(
                c_bbox=c_bbox,
                valid_drawings=valid_drawings,
                valid_images=valid_images,
                all_captions_on_page=captions,
                max_vertical=220.0,
                max_horizontal=350.0,
            )

            all_elements = fig_drawings + [img["bbox"] for img in fig_images]
            if not all_elements:
                continue

            # Calculate base bounding box from detected elements, strictly respecting stops
            base_x0 = min(r.x0 for r in all_elements)
            base_y0 = min(r.y0 for r in all_elements)
            base_x1 = max(r.x1 for r in all_elements)
            base_y1 = max(r.y1 for r in all_elements)

            if above_stop is not None:
                base_y0 = max(base_y0, above_stop + 2.0)
            if below_stop is not None:
                base_y1 = min(base_y1, below_stop - 2.0)

            # Include caption block in compound union
            compound_bbox = fitz.Rect(
                min(base_x0, c_bbox.x0),
                min(base_y0, c_bbox.y0),
                max(base_x1, c_bbox.x1),
                max(base_y1, c_bbox.y1),
            )

            # Harvest internal text labels strictly inside the compound bbox
            labels_inside: List[str] = []
            for b in blocks:
                if b[6] == 0:
                    b_rect = fitz.Rect(b[:4])
                    if compound_bbox.contains(b_rect) and not b_rect.intersects(c_bbox):
                        txt = b[4].strip()
                        if 2 <= len(txt) <= 90:
                            labels_inside.append(txt)

            # Extract surrounding paragraph context
            surrounding_ctx = self._extract_nearby_context(full_page_text, f"Fig {c_id}")

            clean_caption_desc = re.sub(
                r'^(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?|Table|Tab\.?|Chart|Activity|Diagram|Exhibit)\s*[0-9.]+\s*[:.\-]?\s*',
                '',
                c_text,
                flags=re.IGNORECASE,
            ).strip()

            # Render complete high-DPI figure crop with dynamic padding and obstacle avoidance
            full_crop_img = self.crop_page_region(
                page, compound_bbox, is_meaningful_fn, obstacles=body_obstacles
            )
            if full_crop_img:
                safe_fig_id = c_id.replace('.', '_')
                out_filename = f"{pdf_stem}_p{page_number}_fig{safe_fig_id}.png"
                out_path = IMAGES_DIR / out_filename
                full_crop_img.save(str(out_path))
                saved_image_paths.append(str(out_path))

                source_type = (
                    "vector_region" if fig_drawings and not fig_images
                    else "embedded_image" if fig_images and not fig_drawings
                    else "mixed"
                )

                # Combine caption terms + internal labels + context terms for concept vector
                rich_term_source = f"{c_text} {' '.join(labels_inside)} {surrounding_ctx}"
                concept_keywords = extract_terms_fn(rich_term_source)

                fig_meta = figure_factory_fn(
                    figure_id=c_id,
                    figure_label=f"Figure {c_id}",
                    subfigure_id=None,
                    caption=clean_caption_desc or c_text,
                    page_number=page_number,
                    image_path=str(out_path),
                    image_filename=out_filename,
                    bounding_box=(compound_bbox.x0, compound_bbox.y0, compound_bbox.x1, compound_bbox.y1),
                    source_type=source_type,
                    width=full_crop_img.width,
                    height=full_crop_img.height,
                    confidence=0.95,
                    associated_keywords=concept_keywords,
                    labels_inside=labels_inside,
                    in_text_citations=[],
                    context=f"Page {page_number} • Figure {c_id}: {clean_caption_desc}",
                    surrounding_context=surrounding_ctx,
                )
                extracted_figures.append(fig_meta)

            # Subfigures crop if present (e.g. (a) and (b))
            if subfigs_in_cap and len(subfigs_in_cap) >= 2:
                w = compound_bbox.width
                h = compound_bbox.height
                sub_a_tag = subfigs_in_cap[0][0]
                sub_b_tag = subfigs_in_cap[1][0]
                lbl_a = next((s for s in subfig_labels if s["label"] == sub_a_tag and abs(s["bbox"].y0 - compound_bbox.y1) <= 60), None)
                lbl_b = next((s for s in subfig_labels if s["label"] == sub_b_tag and abs(s["bbox"].y0 - compound_bbox.y1) <= 60), None)

                is_horizontal = bool((lbl_a and lbl_b and abs(lbl_a["bbox"].x0 - lbl_b["bbox"].x0) > 40) or (w > h * 1.2))

                for s_idx, (sub_tag, sub_desc) in enumerate(subfigs_in_cap[:2]):
                    if is_horizontal:
                        mid_x = (compound_bbox.x0 + compound_bbox.x1) / 2.0
                        sub_box = fitz.Rect(compound_bbox.x0, compound_bbox.y0, mid_x + 5, compound_bbox.y1) if s_idx == 0 else fitz.Rect(mid_x - 5, compound_bbox.y0, compound_bbox.x1, compound_bbox.y1)
                    else:
                        mid_y = (compound_bbox.y0 + compound_bbox.y1) / 2.0
                        sub_box = fitz.Rect(compound_bbox.x0, compound_bbox.y0, compound_bbox.x1, mid_y + 5) if s_idx == 0 else fitz.Rect(compound_bbox.x0, mid_y - 5, compound_bbox.x1, compound_bbox.y1)

                    sub_crop = self.crop_page_region(page, sub_box, is_meaningful_fn, obstacles=body_obstacles)
                    if sub_crop:
                        safe_sub_id = f"{c_id}_{sub_tag}".replace('.', '_')
                        sub_out_filename = f"{pdf_stem}_p{page_number}_fig{safe_sub_id}.png"
                        sub_out_path = IMAGES_DIR / sub_out_filename
                        sub_crop.save(str(sub_out_path))
                        saved_image_paths.append(str(sub_out_path))

                        sub_meta = figure_factory_fn(
                            figure_id=f"{c_id}.{sub_tag}",
                            figure_label=f"Figure {c_id}({sub_tag})",
                            subfigure_id=sub_tag,
                            caption=sub_desc,
                            page_number=page_number,
                            image_path=str(sub_out_path),
                            image_filename=sub_out_filename,
                            bounding_box=(sub_box.x0, sub_box.y0, sub_box.x1, sub_box.y1),
                            source_type="rendered_crop",
                            width=sub_crop.width,
                            height=sub_crop.height,
                            confidence=0.92,
                            associated_keywords=extract_terms_fn(f"{c_text} {sub_desc}"),
                            labels_inside=[],
                            in_text_citations=[],
                            context=f"Page {page_number} • Figure {c_id}({sub_tag}): {sub_desc}",
                            surrounding_context=surrounding_ctx,
                        )
                        extracted_figures.append(sub_meta)

        figure_dicts = [f.to_dict() if hasattr(f, "to_dict") else f for f in extracted_figures]
        sorted_refs = sorted(list(figure_refs_found))
        return saved_image_paths, figure_dicts, sorted_refs
