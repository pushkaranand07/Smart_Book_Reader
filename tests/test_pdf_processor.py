from PIL import Image

from src.pdf_processor import PageResult, PDFProcessor, clean_repeated_lines, is_meaningful_content_image


def test_clean_repeated_lines_removes_adjacent_duplicates():
    assert clean_repeated_lines("Title\nTitle\nBody") == "Title Body"


def test_meaningful_image_rejects_flat_image_and_accepts_detail():
    flat = Image.new("RGB", (100, 100), "white")
    detailed = Image.new("RGB", (100, 100), "white")
    for x in range(20, 80):
        for y in range(20, 80):
            detailed.putpixel((x, y), (0, 0, 0))

    assert not is_meaningful_content_image(flat)
    assert is_meaningful_content_image(detailed)


def test_process_pdf_keeps_other_pages_when_one_page_fails(tmp_path, monkeypatch):
    pdf_path = tmp_path / "partial.pdf"
    document = __import__("fitz").open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    processor = PDFProcessor()

    def fake_process_page(doc, page, page_number, pdf_stem):
        if page_number == 2:
            raise RuntimeError("synthetic page failure")
        return PageResult(
            page_number=page_number,
            page_type="Digital",
            text="healthy page",
            char_count=12,
            word_count=2,
            ocr_applied=False,
            ocr_time_sec=0.0,
            confidence_note="Embedded digital text found",
        )

    monkeypatch.setattr(processor, "process_page", fake_process_page)
    result = processor.process_pdf(pdf_path, max_workers=2)

    assert len(result["pages"]) == 2
    assert result["pages"][0].status == "ok"
    assert result["pages"][1].status == "failed"
    assert result["pages"][1].errors[0]["code"] == "PAGE_PROCESSING_FAILED"
