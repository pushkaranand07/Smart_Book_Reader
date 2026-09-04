from PIL import Image

from src.pdf_processor import clean_repeated_lines, is_meaningful_content_image


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
