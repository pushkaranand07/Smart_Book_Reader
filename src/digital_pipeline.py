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
import logging
import re
from typing import Any

import pymupdf as fitz
from PIL import Image, UnidentifiedImageError

from src.captions import FigureNamer, nearest_caption, parse_caption, text_without_symbol_fonts
from src.storage import IMAGES_DIR

logger = logging.getLogger(__name__)
_IMAGE_ERRORS = (ValueError, OSError, RuntimeError, UnidentifiedImageError)


def is_qr_code(pil_img: Image.Image, page_num: int) -> bool:
    """Identify square chapter QR codes or barcodes on page 1 / book headers."""
    w, h = pil_img.size
    aspect = w / h if h > 0 else 0
    return bool(0.85 <= aspect <= 1.15 and page_num == 1 and (w < 400 and h < 400))


def parse_subfigures_from_caption(caption_text: str) -> list[tuple[str, str]]:
    """Parse subfigure tags and descriptions from a caption string."""
    # Subfigure tags are single letters or small roman numerals: "(a)", "(b)", "(ii)". Not "(aq)" or "(s)" states.
    matches = re.findall(r"(?<![A-Za-z0-9+\-])\(([a-h]|i{1,3}|iv)\)\s*([^()]+)", caption_text)
    subfigs: list[tuple[str, str]] = []
    if matches:
        for tag, text in matches:
            clean_tag = tag.strip().lower()
            clean_desc = text.strip().rstrip(".,;")
            clean_desc = re.sub(r"\s+and$", "", clean_desc, flags=re.IGNORECASE).strip()
            if clean_desc:
                subfigs.append((clean_tag, clean_desc))
    return subfigs


class DigitalPipeline:
    """Standalone extractor pipeline for Digital (text-based) PDF pages."""

    def __init__(
        self,
        render_dpi: int = 200,
        pad_pt: float = 20.0,
        visual_detector: Any | None = None,
    ):
        self.render_dpi = render_dpi
        self.pad_pt = pad_pt
        self.visual_detector = visual_detector

    def crop_page_region(
        self,
        page: fitz.Page,
        bbox: fitz.Rect,
        is_meaningful_fn,
        obstacles: list[fitz.Rect] | None = None,
    ) -> Image.Image | None:
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
    ) -> list[dict[str, Any]]:
        """Extract embedded raster images from digital page."""
        valid_images: list[dict[str, Any]] = []
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

                if bbox.width >= page_rect.width * 0.90 and bbox.height >= page_rect.height * 0.90:
                    continue

                valid_images.append(
                    {
                        "xref": xref,
                        "img_idx": img_idx + 1,
                        "bbox": bbox,
                        "pil_img": pil_img,
                        "ext": base_img.get("ext", "png"),
                        "raw_bytes": image_bytes,
                    }
                )
            except _IMAGE_ERRORS as exc:
                logger.debug("skipping unreadable embedded image xref=%s on page %s: %s", xref, page_number, exc)
                continue

        return valid_images

    def _extract_vector_drawings(
        self,
        page: fitz.Page,
        page_rect: fitz.Rect,
    ) -> list[fitz.Rect]:
        """Extract vector drawing boundaries, excluding margins and headers."""
        valid_drawings: list[fitz.Rect] = []
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
        except (RuntimeError, ValueError) as exc:
            logger.warning("vector drawing extraction failed: %s", exc)
        return valid_drawings

    def _detect_captions(self, page: fitz.Page) -> list[dict[str, Any]]:
        """Find real captions: text blocks that START with "Fig./Figure/Table N(.N)".

        Spans set in symbol fonts (Wingdings bullets) are dropped before matching, and body text that merely
        mentions a figure is ignored.
        """
        captions: list[dict[str, Any]] = []
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except (RuntimeError, ValueError) as exc:
            logger.warning("caption detection failed: %s", exc)
            return captions
        for block in blocks:
            if block.get("type") != 0:
                continue
            text = text_without_symbol_fonts(block, caption_mode=True)
            if not text or len(text) > 380:
                continue
            match = parse_caption(text)
            if match is None:
                continue
            fig_id = f"{match.number}.{match.sub}" if match.sub else match.number
            label_text = f"{match.label} {match.text}" if match.text else match.label
            captions.append(
                {
                    "fig_id": fig_id,
                    "prefix": match.label,
                    "bbox": fitz.Rect(block["bbox"]),
                    "text": label_text,
                    "caption": match.text,
                    "match": match,
                }
            )
        return captions

    def _detect_subfig_labels(self, blocks: list[Any]) -> list[dict[str, Any]]:
        """Detect standalone subfigure labels like (a), (b)."""
        subfig_labels: list[dict[str, Any]] = []
        for b in blocks:
            txt = b[4].strip()
            for m in re.finditer(r"\(([a-zA-Z0-9]+)\)", txt):
                subfig_labels.append(
                    {
                        "label": m.group(1).lower(),
                        "bbox": fitz.Rect(b[:4]),
                    }
                )
        return subfig_labels

    def _extract_nearby_context(self, page_text: str, label_pattern: str) -> str:
        """Extract surrounding text sentences within 350 characters of a figure mention."""
        match = re.search(rf"\b{re.escape(label_pattern)}\b", page_text, re.IGNORECASE)
        if not match:
            return page_text[:400].replace("\n", " ").strip()
        pos = match.start()
        start = max(0, pos - 250)
        end = min(len(page_text), pos + 350)
        return page_text[start:end].replace("\n", " ").strip()

    def get_nearby_visuals(
        self,
        c_bbox: fitz.Rect,
        valid_drawings: list[fitz.Rect],
        valid_images: list[dict[str, Any]],
        all_captions_on_page: list[dict[str, Any]],
        max_vertical: float = 220.0,
        max_horizontal: float = 350.0,
    ) -> tuple[list[fitz.Rect], list[dict[str, Any]], float | None, float | None]:
        """Collect visuals within 220pt constrained by hard above/below caption stop boundaries."""
        # Find hard stops (other captions on this page)
        above_stop = max((cap["bbox"].y1 for cap in all_captions_on_page if cap["bbox"].y1 < c_bbox.y0), default=None)
        below_stop = min((cap["bbox"].y0 for cap in all_captions_on_page if cap["bbox"].y0 > c_bbox.y1), default=None)

        fig_drawings: list[fitz.Rect] = []
        for r in valid_drawings:
            # Visual is above caption (caption sits below)
            near_x = abs(r.x0 - c_bbox.x0) <= max_horizontal
            above = 0 <= (c_bbox.y0 - r.y1) <= max_vertical and near_x
            below = 0 <= (r.y0 - c_bbox.y1) <= max_vertical and near_x
            if (above and (above_stop is None or r.y0 >= above_stop)) or (
                not above and below and (below_stop is None or r.y1 <= below_stop)
            ):
                fig_drawings.append(r)

        fig_images: list[dict[str, Any]] = []
        for img in valid_images:
            r = img["bbox"]
            # Image is above caption
            near_x = abs(r.x0 - c_bbox.x0) <= max_horizontal
            above = 0 <= (c_bbox.y0 - r.y1) <= max_vertical and near_x
            below = 0 <= (r.y0 - c_bbox.y1) <= max_vertical and near_x
            if (above and (above_stop is None or r.y0 >= above_stop)) or (
                not above and below and (below_stop is None or r.y1 <= below_stop)
            ):
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
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        """Run digital visual extraction pipeline for a single page."""
        page_rect = page.rect
        blocks = page.get_text("blocks")
        full_page_text = page.get_text("text")

        valid_images = self._extract_embedded_images(doc, page, page_rect, page_number, is_meaningful_fn)
        valid_drawings = self._extract_vector_drawings(page, page_rect)
        captions = self._detect_captions(page)
        namer = FigureNamer(IMAGES_DIR, pdf_stem, page_number)
        subfig_labels = self._detect_subfig_labels(blocks)

        figure_refs_found: set[str] = set()
        for ref_match in re.finditer(
            r"(?:Fig(?:ure|\.)?|FIG(?:URE|\.)?|Table|Activity|Chart)\s*([0-9]+(?:\.[0-9]+)?)",
            full_page_text,
            re.IGNORECASE,
        ):
            figure_refs_found.add(ref_match.group(1))

        # Collect non-caption body text blocks to act as obstacles against bleed-over
        caption_rects = [cap["bbox"] for cap in captions]
        body_obstacles: list[fitz.Rect] = []
        for b in blocks:
            if b[6] == 0:  # text block
                b_rect = fitz.Rect(b[:4])
                if len(b[4].strip()) > 80 and not any(b_rect.intersects(cr) for cr in caption_rects):
                    body_obstacles.append(b_rect)

        saved_image_paths: list[str] = []
        extracted_figures: list[Any] = []

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
            labels_inside: list[str] = []
            for b in blocks:
                if b[6] == 0:
                    b_rect = fitz.Rect(b[:4])
                    if compound_bbox.contains(b_rect) and not b_rect.intersects(c_bbox):
                        txt = b[4].strip()
                        if 2 <= len(txt) <= 90:
                            labels_inside.append(txt)

            # Extract surrounding paragraph context
            surrounding_ctx = self._extract_nearby_context(full_page_text, f"Fig {c_id}")

            clean_caption_desc = cap["caption"]

            # Render complete high-DPI figure crop with dynamic padding and obstacle avoidance
            full_crop_img = self.crop_page_region(page, compound_bbox, is_meaningful_fn, obstacles=body_obstacles)
            if full_crop_img:
                out_path, out_filename = namer.next_path()
                full_crop_img.save(str(out_path))
                saved_image_paths.append(str(out_path))

                source_type = (
                    "vector_region"
                    if fig_drawings and not fig_images
                    else "embedded_image"
                    if fig_images and not fig_drawings
                    else "mixed"
                )

                # Combine caption terms + internal labels + context terms for concept vector
                rich_term_source = f"{c_text} {' '.join(labels_inside)} {surrounding_ctx}"
                concept_keywords = extract_terms_fn(rich_term_source)

                fig_meta = figure_factory_fn(
                    figure_id=c_id,
                    figure_label=cap["prefix"],
                    subfigure_id=None,
                    caption=clean_caption_desc,
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
                    context=f"Page {page_number} • {cap['prefix']}: {clean_caption_desc or ''}",
                    surrounding_context=surrounding_ctx,
                )
                extracted_figures.append(fig_meta)

            # Subfigures crop if present (e.g. (a) and (b))
            if subfigs_in_cap and len(subfigs_in_cap) >= 2:
                w = compound_bbox.width
                h = compound_bbox.height
                sub_a_tag = subfigs_in_cap[0][0]
                sub_b_tag = subfigs_in_cap[1][0]
                lbl_a = next(
                    (
                        s
                        for s in subfig_labels
                        if s["label"] == sub_a_tag and abs(s["bbox"].y0 - compound_bbox.y1) <= 60
                    ),
                    None,
                )
                lbl_b = next(
                    (
                        s
                        for s in subfig_labels
                        if s["label"] == sub_b_tag and abs(s["bbox"].y0 - compound_bbox.y1) <= 60
                    ),
                    None,
                )

                is_horizontal = bool(
                    (lbl_a and lbl_b and abs(lbl_a["bbox"].x0 - lbl_b["bbox"].x0) > 40) or (w > h * 1.2)
                )

                for s_idx, (sub_tag, sub_desc) in enumerate(subfigs_in_cap[:2]):
                    if is_horizontal:
                        mid_x = (compound_bbox.x0 + compound_bbox.x1) / 2.0
                        sub_box = (
                            fitz.Rect(compound_bbox.x0, compound_bbox.y0, mid_x + 5, compound_bbox.y1)
                            if s_idx == 0
                            else fitz.Rect(mid_x - 5, compound_bbox.y0, compound_bbox.x1, compound_bbox.y1)
                        )
                    else:
                        mid_y = (compound_bbox.y0 + compound_bbox.y1) / 2.0
                        sub_box = (
                            fitz.Rect(compound_bbox.x0, compound_bbox.y0, compound_bbox.x1, mid_y + 5)
                            if s_idx == 0
                            else fitz.Rect(compound_bbox.x0, mid_y - 5, compound_bbox.x1, compound_bbox.y1)
                        )

                    sub_crop = self.crop_page_region(page, sub_box, is_meaningful_fn, obstacles=body_obstacles)
                    if sub_crop:
                        sub_out_path, sub_out_filename = namer.next_path()
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

        # -------------------------------------------------------------
        # Extract Remaining Standalone Valid Embedded Images (Zero-Context Visuals)
        # -------------------------------------------------------------
        claimed_rects = [
            fitz.Rect(f["bounding_box"]) if isinstance(f, dict) else fitz.Rect(f.bounding_box)
            for f in extracted_figures
        ]

        for v_img in valid_images:
            r = v_img["bbox"]
            if any(r.intersects(cr) or cr.contains(r) for cr in claimed_rects):
                continue

            pil_img = v_img["pil_img"]
            img_idx = v_img["img_idx"]

            # Only a real caption next to the image counts; never borrow a nearby paragraph.
            caption_match = nearest_caption(
                (r.x0, r.y0, r.x1, r.y1),
                [((c["bbox"].x0, c["bbox"].y0, c["bbox"].x1, c["bbox"].y1), c["match"]) for c in captions],
                max_gap=40.0,
            )
            label_text = caption_match.label if caption_match else f"Page {page_number} image {img_idx}"
            caption_text = caption_match.text if caption_match else None

            out_path, out_filename = namer.next_path()
            pil_img.save(str(out_path))
            saved_image_paths.append(str(out_path))

            fig_meta = figure_factory_fn(
                figure_id=caption_match.number if caption_match else f"p{page_number}_img{img_idx}",
                figure_label=label_text[:60],
                subfigure_id=None,
                caption=caption_text,
                page_number=page_number,
                image_path=str(out_path),
                image_filename=out_filename,
                bounding_box=(r.x0, r.y0, r.x1, r.y1),
                source_type="embedded_image",
                width=pil_img.width,
                height=pil_img.height,
                confidence=0.85,
                associated_keywords=extract_terms_fn(f"{caption_text or ''} {full_page_text[:300]}"),
                labels_inside=[],
                in_text_citations=[],
                context=f"Page {page_number} • Visual: {caption_text or label_text}",
                surrounding_context=full_page_text[:300],
            )
            extracted_figures.append(fig_meta)
            claimed_rects.append(r)

        figure_dicts = [f.to_dict() if hasattr(f, "to_dict") else f for f in extracted_figures]
        sorted_refs = sorted(figure_refs_found)
        return saved_image_paths, figure_dicts, sorted_refs
