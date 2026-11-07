"""Book service: upload storage, loading processed books, and building API views."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pymupdf
from fastapi import UploadFile

from api.config import Settings
from api.db import Book
from api.errors import ApiError
from api.schemas import BookDetail, FigureOut, PageError, PageOut
from api.security import safe_data_path, sign_path

logger = logging.getLogger(__name__)
PDF_MAGIC = b"%PDF-"
CHUNK = 1024 * 1024


def store_upload(file: UploadFile, settings: Settings) -> tuple[str, int, Path]:
    """Stream an upload to uploads/<sha256>.pdf, enforcing size and magic bytes. Returns (sha, size, path)."""
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(dir=settings.uploads_dir, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            first = True
            while chunk := file.file.read(CHUNK):
                if first:
                    if not chunk.startswith(PDF_MAGIC):
                        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "The file is not a PDF.")
                    first = False
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise ApiError(413, "FILE_TOO_LARGE", f"Uploads are limited to {settings.max_upload_mb} MB.")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "The file is empty.")
        sha = digest.hexdigest()
        final = settings.uploads_dir / f"{sha}.pdf"
        if final.exists():
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(final)
        return sha, size, final
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def count_pages(path: Path) -> int:
    try:
        with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
            return int(doc.page_count)
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise ApiError(422, "INVALID_PDF", "The PDF could not be opened.") from exc


def extraction_path(settings: Settings, sha: str) -> Path:
    return settings.data_dir / "extracted" / f"{sha}_extracted.json"


class BookDataCache:
    """LRU of parsed extraction JSON keyed by content hash (shared by all owners of that content)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, settings: Settings, sha: str) -> dict[str, Any]:
        with self._lock:
            if sha in self._items:
                self._items.move_to_end(sha)
                return self._items[sha]
        path = extraction_path(settings, sha)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ApiError(409, "BOOK_NOT_READY", "The book has not finished processing.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("corrupt extraction for %s: %s", sha, exc)
            raise ApiError(500, "EXTRACTION_UNREADABLE", "The processed book could not be read.") from exc
        if not isinstance(data, dict):
            raise ApiError(500, "EXTRACTION_UNREADABLE", "The processed book could not be read.")
        with self._lock:
            self._items[sha] = data
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return data

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def pages_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = data.get("normalized_document") or {}
    pages = normalized.get("pages") or data.get("pages") or []
    return [p for p in pages if isinstance(p, dict)]


def figure_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """figure_key -> figure dict. Keys are stable positions, never filesystem paths."""
    out: dict[str, dict[str, Any]] = {}
    for page in pages_of(data):
        for idx, fig in enumerate(page.get("figures") or [], start=1):
            if isinstance(fig, dict):
                out[f"p{page.get('page_number')}-{idx}"] = fig
    return out


def _clean_caption(fig: dict[str, Any]) -> str | None:
    caption = fig.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        return None
    return caption.strip()


def figure_out(settings: Settings, book_id: str, key: str, fig: dict[str, Any]) -> FigureOut:
    has_image = bool(fig.get("image_path")) and safe_data_path(settings, str(fig["image_path"])) is not None
    figure_id = str(fig.get("figure_id") or key)
    return FigureOut(
        figure_key=key,
        figure_id=figure_id,
        label=str(fig.get("figure_label") or f"Figure {figure_id}"),
        caption=_clean_caption(fig),
        page_number=int(fig.get("page_number") or key.split("-")[0].lstrip("p") or 0),
        image_url=sign_path(settings, f"/api/books/{book_id}/figures/{key}/image") if has_image else None,
    )


def book_detail(settings: Settings, book: Book, data: dict[str, Any]) -> BookDetail:
    pages: list[PageOut] = []
    for page in pages_of(data):
        number = int(page.get("page_number") or 0)
        text = str(page.get("text") or page.get("raw_text") or "")
        figures = [
            figure_out(settings, book.id, f"p{number}-{idx}", fig)
            for idx, fig in enumerate(page.get("figures") or [], start=1)
            if isinstance(fig, dict)
        ]
        pages.append(
            PageOut(
                page_number=number,
                page_type=str(page.get("page_type") or "Digital"),
                status=str(page.get("status") or "ok"),
                text=text,
                word_count=len(text.split()),
                errors=[
                    PageError(code=str(e.get("code", "")), message=str(e.get("message", "")))
                    for e in page.get("errors") or []
                    if isinstance(e, dict)
                ],
                figures=figures,
            )
        )
    return BookDetail(
        book_id=book.id,
        filename=book.filename,
        status=book.status,
        total_pages=book.total_pages,
        created_at=book.created_at,
        digital_pages=book.digital_pages,
        scanned_pages=book.scanned_pages,
        file_url=sign_path(settings, f"/api/books/{book.id}/file"),
        pages=pages,
    )
