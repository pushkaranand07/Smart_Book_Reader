import json

import src.storage as storage
from src.storage import get_cache_key


def test_cache_key_changes_when_pdf_content_changes():
    first = get_cache_key("book.pdf", 40, "hash-a", 150)
    second = get_cache_key("book.pdf", 40, "hash-b", 150)

    assert first != second


def test_cache_key_changes_when_ocr_dpi_changes():
    first = get_cache_key("book.pdf", 40, "hash-a", 150)
    second = get_cache_key("book.pdf", 40, "hash-a", 300)

    assert first != second


def test_save_extraction_results_does_not_rewrite_matching_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "EXTRACTED_DIR", tmp_path / "extracted")
    data = {
        "threshold_used": 40,
        "ocr_dpi": 150,
        "source_file_hash": "hash-a",
        "pages": [],
    }

    path = storage.save_extraction_results("book.pdf", data)
    first_content = path.read_text(encoding="utf-8")
    storage.save_extraction_results("book.pdf", data)

    assert path.read_text(encoding="utf-8") == first_content
    assert json.loads(first_content)["cache_key"] == get_cache_key("book.pdf", 40, "hash-a", 150)
