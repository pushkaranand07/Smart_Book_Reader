import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_DIR = DATA_DIR / "extracted"
PAGES_DIR = DATA_DIR / "pages"
IMAGES_DIR = DATA_DIR / "images"

# Bump this version whenever extraction, padding, or matching logic changes to auto-invalidate stale caches
PIPELINE_VERSION = "v3.1"


def get_cache_key(file_name: str, min_char_threshold: int) -> str:
    """Generate MD5 hash of filename, threshold, and pipeline version for cache invalidation."""
    raw = f"{file_name}:{min_char_threshold}:{PIPELINE_VERSION}"
    return hashlib.md5(raw.encode()).hexdigest()


def ensure_directories() -> None:
    """Ensure data/uploads, data/extracted, data/pages, and data/images directories exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def clear_cache() -> None:
    """Purge extracted JSONs and cropped images to force a fresh re-extraction."""
    for folder in [EXTRACTED_DIR, IMAGES_DIR]:
        if folder.exists():
            shutil.rmtree(folder)
    ensure_directories()


def sanitize_filename(name: str) -> str:
    """Sanitize filename to avoid invalid path characters."""
    return re.sub(r'[\\/*?:"<>| ]', "_", name)


def save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    """Save uploaded PDF bytes to data/uploads/ and return the absolute path."""
    ensure_directories()
    safe_name = sanitize_filename(filename)
    target_path = UPLOADS_DIR / safe_name
    with open(target_path, "wb") as f:
        f.write(file_bytes)
    return target_path


def save_extraction_results(filename: str, data: Dict[str, Any]) -> Path:
    """Save extraction metadata and text to data/extracted/<name>_extracted.json."""
    ensure_directories()
    safe_base = Path(sanitize_filename(filename)).stem
    threshold = data.get("threshold_used", 40)
    data["cache_key"] = get_cache_key(filename, threshold)
    data["pipeline_version"] = PIPELINE_VERSION

    json_path = EXTRACTED_DIR / f"{safe_base}_extracted.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return json_path


def load_extraction_results(filename: str, min_char_threshold: int = 40) -> Optional[Dict[str, Any]]:
    """Load previously saved extraction results if valid and not stale."""
    safe_base = Path(sanitize_filename(filename)).stem
    json_path = EXTRACTED_DIR / f"{safe_base}_extracted.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            expected_key = get_cache_key(filename, min_char_threshold)
            # Stale cache check: if cache key does not match current pipeline version and threshold, reject it
            if data.get("cache_key") == expected_key:
                return data
        except Exception:
            return None
    return None
