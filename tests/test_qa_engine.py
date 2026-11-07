from src.normalized_document import normalize_legacy_document, normalized_document_to_dict
from src.qa_engine import _clean_answer_format, answer_question


def test_clean_answer_format_removes_inline_heading_markers():
    assert _clean_answer_format("Intro ### Chemical Equation\n### Details") == "Intro\n\nChemical Equation\nDetails"


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
    assert all(
        fid not in {"synthesis. The following events occur during this process –", ".1", "Figure .1"}
        for fid in exported_ids
        if fid is not None
    )


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


def test_clear_figure_winner_excludes_other_figures_from_same_page():
    book = {
        "pages": [
            {
                "page_number": 1,
                "page_type": "Digital",
                "text": "The setup for burning a magnesium ribbon in air is shown in Figure 1.1. Figure 1.2 shows a different experiment.",
                "figures": [
                    {
                        "figure_id": "1.1",
                        "figure_label": "Figure 1.1",
                        "caption": "Burning of a magnesium ribbon in air and collection of magnesium oxide in a watch-glass",
                        "page_number": 1,
                        "source_type": "vector_region",
                    },
                    {
                        "figure_id": "1.2",
                        "figure_label": "Figure 1.2",
                        "caption": "Formation of hydrogen gas by the action of dilute sulphuric acid on zinc",
                        "page_number": 1,
                        "source_type": "vector_region",
                    },
                ],
                "images": [],
            }
        ],
    }

    result = answer_question(
        "What are the components and products shown in the setup for burning a magnesium ribbon in air?",
        book,
        api_key=None,
        top_k=1,
    )

    assert [figure.get("figure_id") for figure in result["figures"]] == ["1.1"]


def test_equation_question_does_not_attach_unrequested_page_figures():
    book = {
        "pages": [
            {
                "page_number": 1,
                "page_type": "Digital",
                "text": "The formation of magnesium oxide is described here. Figure 1.2 shows hydrogen gas formation. Figure 1.3 shows slaked lime formation.",
                "figures": [
                    {
                        "figure_id": "1.2",
                        "figure_label": "Figure 1.2",
                        "caption": "Formation of hydrogen gas by the action of dilute sulphuric acid on zinc",
                        "page_number": 1,
                        "source_type": "vector_region",
                    },
                    {
                        "figure_id": "1.3",
                        "figure_label": "Figure 1.3",
                        "caption": "Formation of slaked lime by the reaction of calcium oxide with water",
                        "page_number": 1,
                        "source_type": "vector_region",
                    },
                ],
                "images": [],
            }
        ],
    }

    result = answer_question(
        "Write the balanced chemical equation for the formation of magnesium oxide.", book, api_key=None, top_k=1
    )

    assert result["figures"] == []
    assert result["images"] == []


def test_visual_request_still_retrieves_explicit_figure():
    book = {
        "pages": [
            {
                "page_number": 1,
                "page_type": "Digital",
                "text": "The setup is shown in Figure 1.1.",
                "figures": [
                    {
                        "figure_id": "1.1",
                        "figure_label": "Figure 1.1",
                        "caption": "Burning magnesium ribbon in air",
                        "page_number": 1,
                        "source_type": "vector_region",
                    }
                ],
                "images": [],
            }
        ],
    }

    result = answer_question("What is shown in Figure 1.1?", book, api_key=None, top_k=1)

    assert [figure["figure_id"] for figure in result["figures"]] == ["1.1"]


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
