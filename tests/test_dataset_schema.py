import json

from PIL import Image

from src.layout.dataset_schema import validate_dataset


def test_manifest_validation_accepts_valid_record(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "id": "valid",
        "dataset": "DocLayNet",
        "split": "train",
        "source_id": "source-a",
        "image_path": str(image_path),
        "image_filename": image_path.name,
        "width": 100,
        "height": 100,
        "annotations": [{"category": "Picture", "bbox": [10, 10, 80, 80]}],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = validate_dataset(manifest, check_images=True)

    assert report.status == "PASS"
    assert report.annotation_counts["Picture"] == 1


def test_manifest_validation_rejects_invalid_box(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "id": "invalid",
        "dataset": "DocLayNet",
        "split": "train",
        "source_id": "source-a",
        "image_path": str(image_path),
        "width": 100,
        "height": 100,
        "annotations": [{"category": "Picture", "bbox": [80, 80, 10, 10]}],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = validate_dataset(manifest, check_images=True)

    assert report.status == "PASS"
    assert report.invalid_boxes == 1
