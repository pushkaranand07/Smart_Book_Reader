from PIL import Image

import src.florence_detector as detector_module


def test_detector_accepts_picture_and_rejects_unrelated_labels(monkeypatch):
    detector = detector_module.FlorenceVisualDetector.__new__(
        detector_module.FlorenceVisualDetector
    )
    detector.is_available = True
    image = Image.new("RGB", (100, 100), "white")

    monkeypatch.setattr(
        detector_module,
        "generate_detections",
        lambda *args, **kwargs: {
            "bboxes": [(10, 10, 80, 80), (0, 0, 100, 100)],
            "labels": ["Picture", "person"],
        },
    )

    boxes = detector.detect_visual_boxes(image, min_box_area=1, max_page_coverage=0.9)

    assert len(boxes) == 1
    assert boxes[0]["label"] == "Picture"
    assert boxes[0]["bbox"] == (10, 10, 80, 80)
