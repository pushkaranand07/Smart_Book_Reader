from src.storage import get_cache_key


def test_cache_key_changes_when_pdf_content_changes():
    first = get_cache_key("book.pdf", 40, "hash-a", 150)
    second = get_cache_key("book.pdf", 40, "hash-b", 150)

    assert first != second


def test_cache_key_changes_when_ocr_dpi_changes():
    first = get_cache_key("book.pdf", 40, "hash-a", 150)
    second = get_cache_key("book.pdf", 40, "hash-a", 300)

    assert first != second
