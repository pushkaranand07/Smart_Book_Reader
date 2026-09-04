from src.normalized_document import normalize_legacy_document, normalized_document_to_dict
from src.normalized_document import normalize_legacy_document
from src.qa_engine import answer_question


def test_offline_qa_returns_grounded_page_and_evidence():
    book = {
        "pages": [
            {
                "page_number": 1,
                "page_type": "Digital",
                "text": "Respiration is the process that releases energy from food.",
                "figures": [],
                "images": [],
            },
            {
                "page_number": 2,
                "page_type": "Digital",
                "text": "Photosynthesis uses sunlight to make food.",
                "figures": [],
                "images": [],
            },
        ]
    }

    result = answer_question("What is respiration?", book, api_key=None, top_k=1)

    assert result["is_sufficient"] is True
    assert result["source_pages"] == [1]
    assert "respiration" in result["answer"].lower()


def test_normalized_document_exposes_real_figure_ids_for_qa():
    legacy_pages = [
        {
            "page_number": 4,
            "page_type": "Digital",
            "text": "Figure 5.2 and Figure 5.1 appear together.",
            "figures": [
                {
                    "figure_id": "synthesis. The following events occur during this process –",
                    "figure_label": "Figure synthesis. The following events occur during this process –",
                    "caption": "Let us now see what actually happens during the process of photosynthesis. The following events occur during this process –",
                    "source_type": "vector_region",
                },
                {
                    "figure_id": "5.2",
                    "figure_label": "Figure 5.2",
                    "caption": "Variegated leaf (a) before and (b) after starch test",
                    "source_type": "vector_region",
                },
                {
                    "figure_id": ".1",
                    "figure_label": "Figure .1",
                    "caption": "Activity 5.1",
                    "source_type": "vector_region",
                },
            ],
        }
    ]

    normalized = normalize_legacy_document("doc-1", "example.pdf", "example.pdf", legacy_pages)
    exported = normalized_document_to_dict(normalized)
    exported_ids = [fig.get("figure_id") for page in exported["pages"] for fig in page["figures"]]

    assert "5.2" in exported_ids
    assert "5.1" not in exported_ids
    assert all(fid not in {"synthesis. The following events occur during this process –", ".1", "Figure .1"} for fid in exported_ids if fid is not None)


def test_explicit_figure_reference_selects_single_target_figure():
    book = {
        "pages": [
            {
                "page_number": 5,
                "page_type": "Digital",
                "text": "Figure 5.1 and Figure 5.2 appear together.",
                "figures": [
                    {
                        "figure_id": "5.1",
                        "figure_label": "Figure 5.1",
                        "caption": "Leaf anatomy diagram",
                        "page_number": 5,
                        "source_type": "vector_region",
                        "labels_inside": ["Leaf", "vein"],
                    },
                    {
                        "figure_id": "5.2",
                        "figure_label": "Figure 5.2",
                        "caption": "Leaf test procedure",
                        "page_number": 5,
                        "source_type": "vector_region",
                        "labels_inside": ["test", "starch"],
                    },
                ],
                "images": [],
            }
        ],
        "normalized_document": {
            "pages": [
                {
                    "page_number": 5,
                    "page_type": "Digital",
                    "raw_text": "Figure 5.1 and Figure 5.2 appear together.",
                    "figures": [
                        {
                            "figure_id": "5.1",
                            "figure_label": "Figure 5.1",
                            "caption": "Leaf anatomy diagram",
                            "page_number": 5,
                            "source_type": "vector_region",
                            "labels_inside": ["Leaf", "vein"],
                        },
                        {
                            "figure_id": "5.2",
                            "figure_label": "Figure 5.2",
                            "caption": "Leaf test procedure",
                            "page_number": 5,
                            "source_type": "vector_region",
                            "labels_inside": ["test", "starch"],
                        },
                    ],
                    "captions": [
                        {"text": "Leaf anatomy diagram", "explicit_reference": "5.1"},
                        {"text": "Leaf test procedure", "explicit_reference": "5.2"},
                    ],
                    "relationships": [{"relation_type": "references"}],
                }
            ]
        },
    }

    result = answer_question("What is shown in Figure 5.2?", book, api_key=None, top_k=5)

    assert result["source_pages"] == [5]
    assert [f.get("figure_id") for f in result["figures"]] == ["5.2"]


def test_normalized_pages_are_authoritative_over_legacy_pages():
    book = {
        "pages": [
            {
                "page_number": 1,
                "page_type": "Digital",
                "text": "Legacy text that should not be retrieved.",
                "figures": [],
                "images": [],
            }
        ],
        "normalized_document": {
            "pages": [
                {
                    "page_number": 2,
                    "page_type": "Digital",
                    "raw_text": "Normalized text about cellular respiration.",
                    "figures": [],
                    "captions": [],
                    "relationships": [],
                }
            ]
        },
    }

    result = answer_question("What is cellular respiration?", book, api_key=None, top_k=1)

    assert result["source_pages"] == [2]
    assert "Normalized text" in result["answer"]


def test_normalized_document_deduplicates_same_figure_reference():
    legacy_pages = [
        {
            "page_number": 4,
            "page_type": "Digital",
            "text": "Figure 5.2 and Figure 5.1 appear together.",
            "figures": [
                {
                    "figure_id": "5.2",
                    "figure_label": "Figure 5.2",
                    "caption": "before",
                    "source_type": "vector_region",
                },
                {
                    "figure_id": "5.2",
                    "figure_label": "Figure 5.2",
                    "caption": "after starch test",
                    "source_type": "vector_region",
                },
                {
                    "figure_id": "5.1",
                    "figure_label": "Figure 5.1",
                    "caption": "Cross-section of a leaf",
                    "source_type": "vector_region",
                },
            ],
        }
    ]

    normalized = normalize_legacy_document("doc-2", "example.pdf", "example.pdf", legacy_pages)
    exported = normalized_document_to_dict(normalized)
    figure_ids = [fig.get("figure_id") for page in exported["pages"] for fig in page["figures"]]

    assert figure_ids.count("5.2") == 1
    assert "5.1" in figure_ids
