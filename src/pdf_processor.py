"""PDF Processing & Two-Lane Pipeline Orchestrator.

Orchestrates document classification and routes each page strictly to:
1. Digital Pipeline (`DigitalPipeline`): For native text PDFs using PyMuPDF vector & embedded visuals.
2. Scanned Pipeline (`ScannedPipeline`): For scanned/image pages using Tesseract OCR & OpenCV contour extraction.
"""

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pymupdf as fitz
from PIL import Image, ImageStat

from src.digital_pipeline import DigitalPipeline, parse_subfigures_from_caption
from src.ocr_config import configure_tesseract
from src.scanned_pipeline import ScannedPipeline
from src.storage import IMAGES_DIR, ensure_directories, sanitize_filename
from src.yolo_detector import YOLOVisualDetector

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Constants & Quality Thresholds ---
_MIN_IMAGE_PIXELS = 5_000          # Min pixel area for content images
_MIN_STDDEV = 2.0                  # Min stddev to reject blank / solid-color masks
_MIN_TINT_STDDEV_SUM = 12.0        # Min total channel variance to reject uncaptioned tint/background boxes
_RENDER_DPI = 150                  # DPI for high-quality diagram/vector crops (optimized for speed)
_PAD_PT = 20.0                     # Padding around bounding box in points


# --- Shared Data Structures ---

@dataclass
class ExtractedFigure:
    """Represents a rich visual figure extracted from a PDF document."""
    figure_id: str
    figure_label: str
    page_number: int
    image_path: str
    image_filename: str
    bounding_box: Tuple[float, float, float, float]
    source_type: str
    width: int
    height: int
    confidence: float
    subfigure_id: Optional[str] = None
    caption: Optional[str] = None
    description: Optional[str] = None
    relevance_score: float = 0.0
    associated_keywords: List[str] = field(default_factory=list)
    labels_inside: List[str] = field(default_factory=list)
    in_text_citations: List[str] = field(default_factory=list)
    context: Optional[str] = None
    surrounding_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageResult:
    """Represents the complete processed result for a single PDF page."""
    page_number: int
    page_type: str                          # "Digital" or "Scanned"
    text: str
    char_count: int
    word_count: int
    ocr_applied: bool
    ocr_time_sec: float
    confidence_note: str
    image_bytes: Optional[bytes] = None      # PNG preview thumbnail bytes
    images: List[str] = field(default_factory=list)              # List of saved image paths
    figures: List[Dict[str, Any]] = field(default_factory=list)  # Detailed extracted figure dicts
    figure_references: List[str] = field(default_factory=list)   # IDs referenced on this page

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("image_bytes", None)
        return d

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# --- Shared Helper Functions ---

def clean_repeated_lines(raw_text: str) -> str:
    """Deduplicate repeated adjacent lines commonly produced by desktop publishing exports."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    unique_lines: List[str] = []
    for line in lines:
        if not unique_lines or line != unique_lines[-1]:
            unique_lines.append(line)
    return " ".join(unique_lines)


def is_meaningful_content_image(pil_img: Image.Image) -> bool:
    """Return True if image has real visual detail (not a spacer, solid mask, or flat tint box)."""
    w, h = pil_img.size
    if w * h < _MIN_IMAGE_PIXELS:
        return False
    try:
        rgb = pil_img.convert("RGB")
        stat = ImageStat.Stat(rgb)
        if all(s < _MIN_STDDEV for s in stat.stddev):
            return False
        if sum(stat.stddev) < _MIN_TINT_STDDEV_SUM:
            return False
    except Exception:
        pass
    return True


def extract_meaningful_terms(text: str) -> List[str]:
    """Extract normalized search keywords from a query or text passage."""
    stopwords = {
        "what", "is", "the", "of", "and", "in", "to", "a", "an", "for", "on",
        "how", "do", "does", "did", "with", "from", "by", "as", "or", "which", "where",
        "about", "show", "describe", "explain", "give", "diagram", "figure", "image",
        "their", "them", "they", "this", "that", "these", "those", "are", "was", "were",
        "can", "could", "would", "should", "tell", "me", "you", "your", "its", "each",
        "some", "such", "also", "into", "over", "after", "before", "between", "draw",
        "illustrate", "represented", "representing", "schematic",
    }
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = [tok.strip() for tok in cleaned.split() if tok.strip()]
    meaningful = [t for t in tokens if len(t) > 2 and t not in stopwords]
    return meaningful


# --- Main PDF Processor Orchestrator ---

class PDFProcessor:
    """Two-Lane PDF Processing Orchestrator with YOLOv8 Visual Analysis.

    Inspects each page to determine if it is Digital (contains embedded font/char stream)
    or Scanned (raster image only), then executes the dedicated pipeline without mixing up extraction logic.
    """

    def __init__(
        self,
        min_char_threshold: int = 40,
        ocr_dpi: int = 150,
        render_dpi: int = _RENDER_DPI,
    ):
        self.min_char_threshold = min_char_threshold
        self.ocr_dpi = ocr_dpi
        self.render_dpi = render_dpi
        self.tesseract_ok, self.tesseract_msg = configure_tesseract()
        self.yolo_detector = YOLOVisualDetector()
        ensure_directories()

        # Initialize the two independent pipelines with YOLO detector integration
        self.digital_pipeline = DigitalPipeline(
            render_dpi=self.render_dpi,
            pad_pt=_PAD_PT,
            yolo_detector=self.yolo_detector,
        )
        self.scanned_pipeline = ScannedPipeline(
            ocr_dpi=self.ocr_dpi,
            render_dpi=self.render_dpi,
            tesseract_ok=self.tesseract_ok,
            tesseract_msg=self.tesseract_msg,
            yolo_detector=self.yolo_detector,
        )

    def get_page_preview_bytes(self, page: fitz.Page, dpi: int = 120) -> bytes:
        """Get PNG preview bytes of a page for UI rendering."""
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")

    def process_page(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
        pdf_stem: str,
    ) -> PageResult:
        """Inspect page format and route cleanly to Digital or Scanned pipeline."""
        extracted_text = page.get_text("text").strip()
        cleaned_text = " ".join(extracted_text.split())
        meaningful_char_count = len(cleaned_text)

        # -------------------------------------------------------------
        # LANE 1: DIGITAL PIPELINE (Native PDF text & vector graphics)
        # -------------------------------------------------------------
        if meaningful_char_count >= self.min_char_threshold:
            saved_images, figure_data, figure_refs = self.digital_pipeline.extract_visuals(
                doc=doc,
                page=page,
                page_number=page_number,
                pdf_stem=pdf_stem,
                is_meaningful_fn=is_meaningful_content_image,
                clean_text_fn=clean_repeated_lines,
                extract_terms_fn=extract_meaningful_terms,
                figure_factory_fn=ExtractedFigure,
            )
            words = [w for w in cleaned_text.split() if w]
            return PageResult(
                page_number=page_number,
                page_type="Digital",
                text=extracted_text,
                char_count=len(extracted_text),
                word_count=len(words),
                ocr_applied=False,
                ocr_time_sec=0.0,
                confidence_note=f"Embedded digital text found ({meaningful_char_count} chars)",
                image_bytes=None,
                images=saved_images,
                figures=figure_data,
                figure_references=figure_refs,
            )

        # -------------------------------------------------------------
        # LANE 2: SCANNED PIPELINE (OCR & OpenCV morphology)
        # -------------------------------------------------------------
        pil_img = self.scanned_pipeline.render_page_image(page, dpi=self.ocr_dpi)
        ocr_text, ocr_duration, ocr_data, confidence_note = self.scanned_pipeline.run_ocr(pil_img)

        saved_images, figure_data, figure_refs = self.scanned_pipeline.extract_visuals(
            page=page,
            page_number=page_number,
            pdf_stem=pdf_stem,
            pil_page_img=pil_img,
            ocr_data=ocr_data,
            is_meaningful_fn=is_meaningful_content_image,
            parse_subfigs_fn=parse_subfigures_from_caption,
            extract_terms_fn=extract_meaningful_terms,
            figure_factory_fn=ExtractedFigure,
        )

        final_text = ocr_text if ocr_text else extracted_text
        words = [w for w in final_text.split() if w]

        return PageResult(
            page_number=page_number,
            page_type="Scanned",
            text=final_text,
            char_count=len(final_text),
            word_count=len(words),
            ocr_applied=True,
            ocr_time_sec=ocr_duration,
            confidence_note=confidence_note,
            image_bytes=None,
            images=saved_images,
            figures=figure_data,
            figure_references=figure_refs,
        )

    def process_pdf(
        self,
        pdf_path: str | Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """Process an entire PDF file concurrently across multiple threads."""
        path_obj = Path(pdf_path)
        doc = fitz.open(str(path_obj))
        total_pages = len(doc)
        doc.close()
        pdf_stem = sanitize_filename(path_obj.stem)

        start_time = time.perf_counter()
        pages_results: List[PageResult] = []

        if total_pages <= 1:
            doc = fitz.open(str(path_obj))
            try:
                res = self.process_page(doc, doc[0], 1, pdf_stem)
                pages_results.append(res)
                if progress_callback:
                    progress_callback(1, 1, "Analyzing page 1 of 1...")
            finally:
                doc.close()
        else:
            num_workers = min(max_workers, total_pages, os.cpu_count() or 4)
            completed_count = 0

            def _process_single_page(p_num: int) -> PageResult:
                t_doc = fitz.open(str(path_obj))
                try:
                    t_page = t_doc[p_num - 1]
                    return self.process_page(t_doc, t_page, p_num, pdf_stem)
                finally:
                    t_doc.close()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_page, p): p for p in range(1, total_pages + 1)}
                for future in as_completed(futures):
                    res = future.result()
                    pages_results.append(res)
                    completed_count += 1
                    if progress_callback:
                        progress_callback(
                            completed_count,
                            total_pages,
                            f"Analyzing pages ({completed_count}/{total_pages} processed)...",
                        )

            pages_results.sort(key=lambda r: r.page_number)

        digital_count = sum(1 for r in pages_results if r.page_type == "Digital")
        scanned_count = len(pages_results) - digital_count
        total_duration = round(time.perf_counter() - start_time, 2)

        return {
            "filename": path_obj.name,
            "filepath": str(path_obj.resolve()),
            "total_pages": total_pages,
            "digital_pages": digital_count,
            "scanned_pages": scanned_count,
            "processing_time_sec": total_duration,
            "pages": pages_results,
        }


def process_book(
    pdf_path: str | Path,
    min_char_threshold: int = 40,
    ocr_dpi: int = 150,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Process a PDF book, handle OCR and visual figure extraction, and persist cache."""
    from src.storage import save_extraction_results

    processor = PDFProcessor(
        min_char_threshold=min_char_threshold,
        ocr_dpi=ocr_dpi,
    )
    results = processor.process_pdf(pdf_path, progress_callback=progress_callback)

    pages_data = [page.to_dict() for page in results["pages"]]

    export_dict = {
        "filename": results["filename"],
        "filepath": results["filepath"],
        "total_pages": results["total_pages"],
        "digital_pages": results["digital_pages"],
        "scanned_pages": results["scanned_pages"],
        "processing_time_sec": results["processing_time_sec"],
        "threshold_used": min_char_threshold,
        "pages": pages_data,
    }
    save_extraction_results(results["filename"], export_dict)
    return export_dict


# --- Multi-Signal Relevance Scoring & Anti-Uncertainty Engine ---

_PROCESS_KEYWORDS = {
    "transport", "exchange", "mechanism", "cycle", "pathway", "movement",
    "reaction", "flow", "transpiration", "circulation", "transfer", "enter", "enters"
}
_STRUCTURE_KEYWORDS = {
    "structure", "system", "anatomy", "cross-section", "sectional", "diagram",
    "parts", "internal", "view"
}


def score_figure_relevance(
    figure: Dict[str, Any],
    query_terms: List[str],
    query_raw: str,
    page_text: str,
) -> Tuple[float, Dict[str, float]]:
    """Calculate multi-signal relevance score for a figure using composite weights.

    Formula:
    S = 0.40 * S_semantic + 0.25 * S_caption + 0.20 * S_citation + 0.10 * S_context + 0.05 * S_type
    Returns (final_score [0..100], component_breakdown_dict)
    """
    if not query_terms:
        return 0.0, {}

    caption = (figure.get("caption") or "").lower()
    label = (figure.get("figure_label") or "").lower()
    fig_id = (figure.get("figure_id") or "").lower()
    keywords = [k.lower() for k in figure.get("associated_keywords", [])]
    labels_inside = [lbl.lower() for lbl in figure.get("labels_inside", [])]
    surrounding_ctx = (figure.get("surrounding_context") or figure.get("context") or "").lower()

    n_terms = len(query_terms)

    # 1. Semantic / Concept Similarity (40%)
    # Matches against distilled keywords, subfigure terms, and internal diagram labels
    concept_pool = set(keywords)
    for lbl in labels_inside:
        for t in extract_meaningful_terms(lbl):
            concept_pool.add(t)

    concept_hits = sum(1 for term in query_terms if term in concept_pool or any(term in k for k in concept_pool))
    s_semantic = concept_hits / max(1, n_terms)

    # 2. Caption Similarity (25%)
    # Direct hits in figure caption or label
    caption_hits = 0
    for term in query_terms:
        if term in caption or term in label or term == fig_id or f"fig {term}" in label:
            caption_hits += 1

    s_caption = caption_hits / max(1, n_terms)
    # Exact phrase substring bonus in caption
    if len(query_raw) > 6 and query_raw.lower() in caption:
        s_caption = min(1.0, s_caption + 0.4)

    # 3. In-Text Citation Proximity (20%)
    # Looks for sentence-level co-occurrence of query terms and Fig X.Y
    s_citation = 0.0
    clean_id = fig_id.split('.')[0] + '.' + fig_id.split('.')[1] if '.' in fig_id else fig_id
    if clean_id:
        fig_pat = rf'Fig(?:ure|\.)?\s*{re.escape(clean_id)}'
        cit_matches = list(re.finditer(fig_pat, page_text, re.IGNORECASE))
        if cit_matches:
            hit_count = 0
            for cm in cit_matches:
                span_start = max(0, cm.start() - 150)
                span_end = min(len(page_text), cm.end() + 150)
                citation_window = page_text[span_start:span_end].lower()
                for term in query_terms:
                    if term in citation_window:
                        hit_count += 1
            s_citation = min(1.0, hit_count / max(1, n_terms))

    # 4. Surrounding Context (10%)
    context_hits = sum(1 for term in query_terms if term in surrounding_ctx)
    s_context = context_hits / max(1, n_terms)

    # 5. Structural Intent Alignment (5%)
    query_is_process = any(w in query_raw.lower() for w in _PROCESS_KEYWORDS)
    query_is_struct = any(w in query_raw.lower() for w in _STRUCTURE_KEYWORDS)

    fig_text_full = f"{caption} {label} {' '.join(labels_inside)}"
    fig_is_process = any(w in fig_text_full for w in _PROCESS_KEYWORDS)
    fig_is_struct = any(w in fig_text_full for w in _STRUCTURE_KEYWORDS)

    s_type = 0.5  # neutral baseline
    if query_is_process and fig_is_process:
        s_type = 1.0
    elif query_is_struct and fig_is_struct:
        s_type = 1.0
    elif query_is_process and fig_is_struct and not fig_is_process:
        s_type = 0.0

    # Composite weighted formula scaled to 0-100
    composite = (
        0.40 * s_semantic +
        0.25 * s_caption +
        0.20 * s_citation +
        0.10 * s_context +
        0.05 * s_type
    ) * 100.0

    components = {
        "semantic": round(s_semantic, 2),
        "caption": round(s_caption, 2),
        "citation": round(s_citation, 2),
        "context": round(s_context, 2),
        "type": round(s_type, 2),
    }

    # If absolutely zero signal matched anywhere, score is zero
    if concept_hits == 0 and caption_hits == 0 and s_citation == 0:
        return 0.0, components

    return round(composite, 2), components


def find_figures_for_query(
    page_results: List[Any],
    query: str,
    top_k: int = 4,
    min_score: float = 20.0,
) -> Tuple[List[Dict[str, Any]], float, bool]:
    """Retrieve and rank all relevant figures across the book.

    Returns:
        (ranked_figures, confidence_margin, is_ambiguous)
    """
    query_terms = extract_meaningful_terms(query)
    if not query_terms:
        return [], 0.0, False

    scored_figures: List[Tuple[float, Dict[str, Any]]] = []

    for page in page_results:
        p_text = page["text"] if isinstance(page, dict) else page.text
        p_figs = page["figures"] if isinstance(page, dict) else page.figures

        for fig in p_figs:
            fig_copy = dict(fig) if isinstance(fig, dict) else fig.to_dict()
            rel_score, components = score_figure_relevance(fig_copy, query_terms, query, p_text)
            if rel_score >= min_score:
                fig_copy["relevance_score"] = rel_score
                fig_copy["relevance_components"] = components
                scored_figures.append((rel_score, fig_copy))

    # Sort descending by score
    scored_figures.sort(key=lambda item: item[0], reverse=True)

    if not scored_figures:
        return [], 0.0, False

    # Evaluate confidence margin between top 2 distinct figures
    margin = 100.0
    if len(scored_figures) >= 2:
        margin = scored_figures[0][0] - scored_figures[1][0]

    is_ambiguous = (len(scored_figures) >= 2 and margin < 15.0 and scored_figures[0][0] >= 30.0)

    max_score = scored_figures[0][0]
    filtered = [fig for s, fig in scored_figures if s >= max(min_score, max_score * 0.45)]
    return filtered[:top_k], round(margin, 2), is_ambiguous