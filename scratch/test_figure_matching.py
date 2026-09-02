"""Verification test for multi-signal figure scoring and disambiguation."""

from src.pdf_processor import score_figure_relevance, find_figures_for_query, extract_meaningful_terms

def run_tests():
    # Mock figure dataset representing NCERT Class 10 Life Processes figures
    mock_pages = [
        {
            "page_number": 12,
            "text": "Figure 5.9 shows the human respiratory system with trachea, bronchi, and lungs with alveolar sacs.",
            "figures": [
                {
                    "figure_id": "5.9",
                    "figure_label": "Figure 5.9",
                    "page_number": 12,
                    "caption": "Human respiratory system",
                    "labels_inside": ["Nasal passage", "Mouth cavity", "Pharynx", "Larynx", "Trachea", "Lungs", "Ribs", "Diaphragm", "Alveoli"],
                    "associated_keywords": ["human", "respiratory", "system", "nasal", "larynx", "trachea", "lungs", "alveoli"],
                    "surrounding_context": "Air enters the body through the nostrils... within the lungs, the passage divides into smaller and smaller tubes...",
                    "image_path": "data/images/fig_5_9.png"
                }
            ]
        },
        {
            "page_number": 14,
            "text": "Oxygen-rich blood from the lungs comes to the thin-walled upper chamber of the heart. Schematic representation of transport and exchange of oxygen and carbon dioxide is shown in Fig. 5.11.",
            "figures": [
                {
                    "figure_id": "5.10",
                    "figure_label": "Figure 5.10",
                    "page_number": 14,
                    "caption": "Schematic sectional view of the human heart",
                    "labels_inside": ["Vena cava", "Right atrium", "Right ventricle", "Septum", "Left ventricle", "Left atrium", "Aorta", "Pulmonary arteries"],
                    "associated_keywords": ["heart", "sectional", "view", "atrium", "ventricle", "aorta", "pulmonary"],
                    "surrounding_context": "The heart is a muscular organ which is as big as our fist...",
                    "image_path": "data/images/fig_5_10.png"
                },
                {
                    "figure_id": "5.11",
                    "figure_label": "Figure 5.11",
                    "page_number": 14,
                    "caption": "Schematic representation of transport and exchange of oxygen and carbon dioxide",
                    "labels_inside": ["Lungs", "Pulmonary vein", "Left atrium", "Body organs", "Capillaries", "Vena cava", "Pulmonary artery"],
                    "associated_keywords": ["transport", "exchange", "oxygen", "carbon", "dioxide", "blood", "lungs", "heart", "circulation"],
                    "surrounding_context": "Oxygen enters the blood in the lungs. The separation of the right side and the left side of the heart is useful...",
                    "image_path": "data/images/fig_5_11.png"
                }
            ]
        },
        {
            "page_number": 17,
            "text": "Movement of water during transpiration in a tree is illustrated in Fig. 5.12.",
            "figures": [
                {
                    "figure_id": "5.12",
                    "figure_label": "Figure 5.12",
                    "page_number": 17,
                    "caption": "Movement of water during transpiration in a tree",
                    "labels_inside": ["Transpiration", "Xylem vessels", "Roots", "Water absorption"],
                    "associated_keywords": ["movement", "water", "transpiration", "tree", "xylem", "absorption"],
                    "surrounding_context": "Water is moved upwards through xylem tissue by transpiration pull...",
                    "image_path": "data/images/fig_5_12.png"
                }
            ]
        }
    ]

    print("=== TEST 1: Process Query: 'Oxygen enters the blood in the lungs' ===")
    q1 = "Oxygen enters the blood in the lungs"
    ranked_1, margin_1, is_amb_1 = find_figures_for_query(mock_pages, q1)
    print(f"Top Figure: {ranked_1[0]['figure_id']} ({ranked_1[0]['caption']})")
    print(f"Score: {ranked_1[0]['relevance_score']} | Breakdown: {ranked_1[0]['relevance_components']}")
    if len(ranked_1) > 1:
        print(f"Runner-up: {ranked_1[1]['figure_id']} ({ranked_1[1]['caption']}) - Score: {ranked_1[1]['relevance_score']}")
    print(f"Margin: {margin_1} | Ambiguous: {is_amb_1}")
    assert ranked_1[0]["figure_id"] == "5.11", f"Expected 5.11, got {ranked_1[0]['figure_id']}"
    print(">>> PASS: 5.11 correctly ranked above 5.9 and 5.10!\n")

    print("=== TEST 2: Process Query: 'Movement of water during transpiration' ===")
    q2 = "Movement of water during transpiration"
    ranked_2, margin_2, is_amb_2 = find_figures_for_query(mock_pages, q2)
    print(f"Top Figure: {ranked_2[0]['figure_id']} ({ranked_2[0]['caption']})")
    print(f"Score: {ranked_2[0]['relevance_score']}")
    assert ranked_2[0]["figure_id"] == "5.12", f"Expected 5.12, got {ranked_2[0]['figure_id']}"
    print(">>> PASS: 5.12 correctly ranked #1!\n")

    print("=== TEST 3: Structural Query: 'Human respiratory system' ===")
    q3 = "Human respiratory system diagram"
    ranked_3, margin_3, is_amb_3 = find_figures_for_query(mock_pages, q3)
    print(f"Top Figure: {ranked_3[0]['figure_id']} ({ranked_3[0]['caption']})")
    print(f"Score: {ranked_3[0]['relevance_score']}")
    assert ranked_3[0]["figure_id"] == "5.9", f"Expected 5.9, got {ranked_3[0]['figure_id']}"
    print(">>> PASS: 5.9 correctly ranked #1 for anatomical structure query!\n")

    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
