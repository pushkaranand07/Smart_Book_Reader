from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def make_canonical_id(document_id: str, page_number: int, kind: str, index: int) -> str:
    return f"doc:{document_id}:page:{page_number}:{kind}:{index}"


def _figure_reference_from_legacy(raw_figure: Dict[str, Any]) -> Optional[str]:
    for candidate in [
        raw_figure.get("figure_id"),
        raw_figure.get("figure_label"),
        raw_figure.get("caption"),
        raw_figure.get("description"),
        raw_figure.get("surrounding_context"),
        raw_figure.get("context"),
    ]:
        ref = parse_explicit_reference(str(candidate or ""))
        if ref:
            return ref
    return None


def _is_valid_legacy_figure(raw_figure: Dict[str, Any]) -> bool:
    if not isinstance(raw_figure, dict):
        return False

    caption = str(raw_figure.get("caption") or "")
    label = str(raw_figure.get("figure_label") or "")
    raw_id = str(raw_figure.get("figure_id") or "")

    if raw_id.lower() in {"none", "null", "nan", ""} and not label and not caption:
        return False

    # OCR often misclassifies long explanatory sentences as figures. Those should be filtered
    # unless they include an explicit figure/table reference.
    if not _figure_reference_from_legacy(raw_figure) and len(caption) > 150:
        return False

    if _figure_reference_from_legacy(raw_figure):
        return True

    if raw_id and not re.search(r"\s", raw_id) and re.search(r"\d", raw_id):
        return True

    return bool(label.strip()) or bool(caption.strip())


def parse_explicit_reference(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"Figure\s+(\d+(?:\.\d+)?(?:\.[a-zA-Z])?)",
        r"Fig\.?\s*(\d+(?:\.\d+)?(?:\.[a-zA-Z])?)",
        r"Table\s+(\d+(?:\.\d+)?(?:\.[a-zA-Z])?)",
        r"Activity\s+(\d+(?:\.\d+)?(?:\.[a-zA-Z])?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _looks_like_activity_label(raw_label: str, raw_caption: str) -> bool:
    combined = f"{raw_label} {raw_caption}".lower()
    if "activity" not in combined:
        return False
    return not re.search(r"\b(?:figure|fig\.?|table)\s+\d", combined, flags=re.IGNORECASE)


def _normalize_legacy_figure_id(raw_figure: Dict[str, Any]) -> Optional[str]:
    raw_id = str(raw_figure.get("figure_id") or "").strip()
    raw_label = str(raw_figure.get("figure_label") or "").strip()
    raw_caption = str(raw_figure.get("caption") or "").strip()

    if _looks_like_activity_label(raw_label, raw_caption):
        return None

    for candidate in [raw_id, raw_label, raw_caption]:
        if not candidate:
            continue
        candidate = candidate.strip()
        if candidate.startswith("."):
            continue

        cleaned = re.sub(r"^(?:Figure|Fig\.?|Table|Activity)\s*", "", candidate, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned.startswith("."):
            continue

        match = re.search(r"(\d+(?:\.\d+)?(?:\.[a-zA-Z])?)", cleaned)
        if not match:
            continue

        value = match.group(1)
        if value.startswith("."):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?(?:\.[a-zA-Z])?", value):
            return value

    return None


@dataclass
class Caption:
    id: str
    page_id: str
    text: str
    bbox: Tuple[float, float, float, float]
    attached_to: Optional[str] = None
    explicit_reference: Optional[str] = None
    section_id: Optional[str] = None
    confidence: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Figure:
    id: str
    page_id: str
    figure_type: str
    bbox: Tuple[float, float, float, float]
    caption_id: Optional[str] = None
    image_source: Optional[str] = None
    extracted_text: Optional[str] = None
    labels_inside: List[str] = field(default_factory=list)
    surrounding_context: Optional[str] = None
    section_id: Optional[str] = None
    reading_order_index: int = 0
    confidence: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    id: str
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    id: str
    document_id: str
    page_number: int
    page_type: str
    width: Optional[float] = None
    height: Optional[float] = None
    raw_text: str = ""
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    captions: List[Caption] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    reading_order: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    id: str
    source_path: str
    filename: str
    pages: List[Page]
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def infer_figure_type(raw_figure: Dict[str, Any]) -> str:
    text = " ".join(
        [
            str(raw_figure.get("figure_label") or ""),
            str(raw_figure.get("caption") or ""),
            str(raw_figure.get("source_type") or ""),
        ]
    ).lower()

    if "table" in text:
        return "table"
    if "formula" in text or "equation" in text:
        return "formula"
    return "figure"


def normalize_legacy_page_result(document: NormalizedDocument, page_result: Dict[str, Any]) -> Page:
    page_number = int(page_result.get("page_number", 1))
    page_id = f"doc:{document.id}:page:{page_number}"

    page = Page(
        id=page_id,
        document_id=document.id,
        page_number=page_number,
        page_type=str(page_result.get("page_type", "Digital")),
        raw_text=str(page_result.get("text", "")),
        provenance={
            "source": "legacy_page_result",
            "ocr_applied": bool(page_result.get("ocr_applied", False)),
            "page_type": page_result.get("page_type"),
        },
    )

    deduped_figures: Dict[str, Dict[str, Any]] = {}
    for raw_fig in page_result.get("figures", []) or []:
        if not isinstance(raw_fig, dict):
            continue
        normalized_fig_id = _normalize_legacy_figure_id(raw_fig)
        if not normalized_fig_id:
            continue
        if normalized_fig_id not in deduped_figures:
            deduped_figures[normalized_fig_id] = raw_fig
            continue

        merged = deduped_figures[normalized_fig_id]
        for field in ["caption", "figure_label", "description", "image_path", "context", "surrounding_context", "source_type"]:
            new_value = raw_fig.get(field)
            if new_value and not merged.get(field):
                merged[field] = new_value

        existing_labels = list(merged.get("labels_inside", []) or [])
        for label in list(raw_fig.get("labels_inside", []) or []):
            if label not in existing_labels:
                existing_labels.append(label)
        merged["labels_inside"] = existing_labels

        existing_caption = str(merged.get("caption") or "")
        new_caption = str(raw_fig.get("caption") or "")
        if existing_caption and new_caption and new_caption not in existing_caption:
            merged["caption"] = f"{existing_caption} {new_caption}".strip()
        elif not existing_caption and new_caption:
            merged["caption"] = new_caption

    for idx, raw_fig in enumerate(deduped_figures.values(), start=1):
        raw_caption = str(raw_fig.get("caption") or raw_fig.get("figure_label") or "")
        raw_label = str(raw_fig.get("figure_label") or "")
        normalized_fig_id = _normalize_legacy_figure_id(raw_fig)
        if not normalized_fig_id:
            continue

        fig_id = make_canonical_id(document.id, page_number, "figure", idx)
        explicit_reference = parse_explicit_reference(raw_label) or parse_explicit_reference(raw_caption) or parse_explicit_reference(str(raw_fig.get("surrounding_context") or ""))
        figure = Figure(
            id=fig_id,
            page_id=page_id,
            figure_type=infer_figure_type(raw_fig),
            bbox=tuple(raw_fig.get("bounding_box", (0.0, 0.0, 0.0, 0.0))),
            caption_id=None,
            image_source=raw_fig.get("image_path"),
            extracted_text=raw_caption,
            labels_inside=list(raw_fig.get("labels_inside", []) or []),
            surrounding_context=raw_fig.get("surrounding_context") or raw_fig.get("context"),
            reading_order_index=idx,
            confidence=float(raw_fig.get("confidence", 0.0) or 0.0),
            provenance={
                "raw_figure_id": raw_fig.get("figure_id"),
                "normalized_figure_id": normalized_fig_id,
                "source_type": raw_fig.get("source_type"),
                "parser": "legacy_figure_extractor",
            },
        )

        page.figures.append(figure)

        if raw_caption:
            caption_id = make_canonical_id(document.id, page_number, "caption", idx)
            caption = Caption(
                id=caption_id,
                page_id=page_id,
                text=raw_caption,
                bbox=tuple(raw_fig.get("bounding_box", (0.0, 0.0, 0.0, 0.0))),
                attached_to=figure.id,
                explicit_reference=explicit_reference,
                confidence=figure.confidence,
                provenance={"source": "legacy_caption"},
            )
            figure.caption_id = caption_id
            page.captions.append(caption)

            page.relationships.append(
                Relationship(
                    id=f"{caption.id}->{figure.id}",
                    source_id=caption.id,
                    target_id=figure.id,
                    relation_type="caption_of",
                    confidence=figure.confidence,
                    metadata={"source": "legacy_layout"},
                )
            )

        if explicit_reference:
            page.relationships.append(
                Relationship(
                    id=f"{page.id}->{figure.id}:ref:{explicit_reference}",
                    source_id=page_id,
                    target_id=figure.id,
                    relation_type="references",
                    confidence=0.85,
                    metadata={"explicit_reference": explicit_reference},
                )
            )

    existing_references = {
        caption.explicit_reference
        for caption in page.captions
        if caption.explicit_reference
    }
    for ref_index, reference in enumerate(page_result.get("figure_references", []) or [], start=1):
        explicit_reference = parse_explicit_reference(f"Figure {reference}") or str(reference).strip()
        if not explicit_reference or explicit_reference in existing_references:
            continue
        caption_id = make_canonical_id(document.id, page_number, "reference", ref_index)
        page.captions.append(
            Caption(
                id=caption_id,
                page_id=page_id,
                text=f"Figure {explicit_reference}",
                bbox=(0.0, 0.0, 0.0, 0.0),
                explicit_reference=explicit_reference,
                confidence=0.0,
                provenance={"source": "page_reference"},
            )
        )
        existing_references.add(explicit_reference)

    return page


def normalize_legacy_document(
    document_id: str,
    source_path: str,
    filename: str,
    legacy_pages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> NormalizedDocument:
    """Convert legacy PageResult-style data into the internal normalized graph model."""
    document = NormalizedDocument(
        id=document_id,
        source_path=source_path,
        filename=filename,
        pages=[],
        metadata=metadata or {},
        provenance=provenance or {},
    )

    for page_result in legacy_pages or []:
        if not isinstance(page_result, dict):
            continue
        document.pages.append(normalize_legacy_page_result(document, page_result))

    return document


def normalized_document_to_dict(document: NormalizedDocument) -> Dict[str, Any]:
    """Serialize a normalized document into a JSON-friendly dict for persistence.

    The downstream QA layer expects figure entries to expose the stable keys used by the app:
    `figure_id`, `figure_label`, `caption`, `page_number`, `source_type`, and `image_path`.
    """
    serialized_pages = []
    for page in document.pages:
        page_dict = asdict(page)
        page_dict["text"] = page.raw_text
        page_dict["char_count"] = len(page.raw_text)
        page_dict["word_count"] = len(page.raw_text.split())
        figures_out = []
        seen_ids = set()
        for fig in page.figures:
            caption = next((c.text for c in page.captions if c.attached_to == fig.id), fig.extracted_text or "")
            explicit_ref = next((c.explicit_reference for c in page.captions if c.attached_to == fig.id and c.explicit_reference), None)
            fig_id = explicit_ref or fig.id.split(":figure:")[-1] if fig.id.startswith("doc:") else fig.id
            fig_label = f"Figure {explicit_ref}" if explicit_ref else (fig.extracted_text or caption or fig.id)
            if fig_id in seen_ids:
                continue
            seen_ids.add(fig_id)
            figures_out.append(
                {
                    "figure_id": fig_id,
                    "figure_label": fig_label,
                    "caption": caption,
                    "page_number": page.page_number,
                    "source_type": fig.provenance.get("source_type", page.page_type),
                    "labels_inside": fig.labels_inside,
                    "surrounding_context": fig.surrounding_context,
                    "image_path": fig.image_source,
                    "bounding_box": fig.bbox,
                    "confidence": fig.confidence,
                    "figure_type": fig.figure_type,
                    "subfigure_id": None,
                }
            )
        page_dict["figures"] = figures_out
        page_dict["images"] = [
            figure["image_path"]
            for figure in figures_out
            if figure.get("image_path")
        ]
        serialized_pages.append(page_dict)
    return {
        "id": document.id,
        "source_path": document.source_path,
        "filename": document.filename,
        "metadata": document.metadata,
        "provenance": document.provenance,
        "pages": serialized_pages,
    }
