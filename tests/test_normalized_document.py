import tempfile

import fitz

from src.normalized_document import (
    Caption,
    Figure,
    NormalizedDocument,
    Page,
    Relationship,
    make_canonical_id,
    normalize_legacy_document,
    normalize_legacy_page_result,
    parse_explicit_reference,
)
from src.pdf_processor import process_book
from src.pdf_processor import is_figure_or_table


def test_make_canonical_id():
    assert make_canonical_id("doc-1", 5, "figure", 2) == "doc:doc-1:page:5:figure:2"


def test_parse_explicit_reference():
    assert parse_explicit_reference("Figure 5.2") == "5.2"
    assert parse_explicit_reference("Fig. 12.1") == "12.1"
    assert parse_explicit_reference("Table 3") == "3"


def test_activity_regions_are_not_visual_figures():
    assert not is_figure_or_table({
        "figure_id": ".4",
        "figure_label": "Figure .4",
        "caption": "Activity 5.4",
        "source_type": "vector_region",
    })


def test_normalize_legacy_page_result_builds_graph():
    page_result = {
        "page_number": 4,
        "page_type": "Digital",
        "text": "Let us now see what actually happens during the process of photosynthesis. Figure 5.2 shows a leaf test.",
        "figures": [
            {
                "figure_id": "5.2",
                "figure_label": "Figure 5.2",
                "caption": "Variegated leaf (a) before and (b) after starch test",
                "bounding_box": (10, 20, 100, 200),
                "source_type": "vector_region",
                "confidence": 0.95,
                "labels_inside": ["Leaf", "Starch"],
                "surrounding_context": "Figure 5.2 shows a leaf test.",
            }
        ],
    }

    doc = NormalizedDocument(
        id="doc-1",
        source_path="/tmp/sample.pdf",
        filename="sample.pdf",
        pages=[],
        metadata={},
        provenance={},
    )

    page = normalize_legacy_page_result(doc, page_result)

    assert isinstance(page, Page)
    assert page.page_number == 4
    assert len(page.figures) == 1
    assert page.figures[0].figure_type == "figure"
    assert len(page.captions) == 1
    assert page.captions[0].explicit_reference == "5.2"
    assert any(r.relation_type == "caption_of" for r in page.relationships)
    assert any(r.relation_type == "references" for r in page.relationships)


def test_normalize_legacy_document_and_process_book_expose_adapter_payload(tmp_path):
    pdf_path = tmp_path / "sample_adapter.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Figure 5.2 shows a leaf test.")
    doc.save(pdf_path)
    doc.close()

    legacy_doc = normalize_legacy_document(
        document_id="doc-1",
        source_path=str(pdf_path),
        filename=pdf_path.name,
        legacy_pages=[
            {
                "page_number": 1,
                "page_type": "Digital",
                "text": "Figure 5.2 shows a leaf test.",
                "figures": [
                    {
                        "figure_id": "5.2",
                        "figure_label": "Figure 5.2",
                        "caption": "Variegated leaf before and after starch test",
                        "bounding_box": (10, 20, 100, 200),
                        "source_type": "vector_region",
                        "confidence": 0.95,
                        "labels_inside": ["Leaf", "Starch"],
                        "surrounding_context": "Figure 5.2 shows a leaf test.",
                    }
                ],
            }
        ],
    )

    assert legacy_doc.pages[0].captions[0].explicit_reference == "5.2"

    result = process_book(pdf_path, min_char_threshold=1, ocr_dpi=72)
    assert "normalized_document" in result
    assert result["normalized_document"]["pages"][0]["page_number"] == 1
    assert result["normalized_document"]["pages"][0]["captions"][0]["explicit_reference"] == "5.2"
