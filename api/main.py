from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pdf_processor import process_book
from src.normalized_document import normalize_legacy_document, normalized_document_to_dict
from src.qa_engine import answer_question
from src.storage import (
    DATA_DIR,
    EXTRACTED_DIR,
    hash_file,
    load_extraction_results,
    save_uploaded_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS: dict[str, dict[str, Any]] = {}


def _cors_origins() -> list[str]:
    """Read approved browser origins from the deployment environment."""
    configured = os.environ.get("CORS_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]

app = FastAPI(title="Smart Book Reader API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    api_key: str | None = None


def _media_url(path: str) -> str:
    try:
        relative = Path(path).resolve().relative_to(DATA_DIR.resolve())
    except (OSError, ValueError):
        return path
    return "/media/" + relative.as_posix()


def _public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg")):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return _media_url(str(candidate))
    return value


def _public_book(book: dict[str, Any]) -> dict[str, Any]:
    return _public_value(book)


def _refresh_normalized_document(book: dict[str, Any]) -> dict[str, Any]:
    pages = book.get("pages", [])
    if not isinstance(pages, list):
        return book
    book["normalized_document"] = normalized_document_to_dict(
        normalize_legacy_document(
            document_id=str(book.get("book_id") or Path(book.get("filename", "book")).stem),
            source_path=str(book.get("filepath", "")),
            filename=str(book.get("filename", "book.pdf")),
            legacy_pages=pages,
            provenance={"source": "api_cache_migration"},
        )
    )
    return book


def _book_id(filename: str, content_hash: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(filename).stem).strip("-").lower() or "book"
    return f"{stem}-{content_hash[:10]}"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "smart-book-reader"}


@app.post("/api/books/upload")
def upload_book(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    content = file.file.read()
    saved_path = save_uploaded_file(content, file.filename)
    content_hash = hash_file(saved_path)
    book_id = _book_id(file.filename, content_hash)

    book_data = load_extraction_results(
        saved_path.name,
        file_hash=content_hash,
    )
    if book_data is None:
        try:
            book_data = process_book(saved_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {exc}") from exc
    else:
        book_data = _refresh_normalized_document(book_data)

    book_data["book_id"] = book_id
    BOOKS[book_id] = book_data
    return _public_book(book_data)


@app.get("/api/books/{book_id}")
def get_book(book_id: str) -> dict[str, Any]:
    book = BOOKS.get(book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book is not loaded in this API session.")
    return _public_book(book)


@app.post("/api/books/{book_id}/questions")
def ask_question(book_id: str, request: QuestionRequest) -> dict[str, Any]:
    book = BOOKS.get(book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book is not loaded in this API session.")
    try:
        result = answer_question(
            query=request.question,
            book_data=book,
            api_key=request.api_key or os.environ.get("GEMINI_API_KEY"),
            top_k=5,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question processing failed: {exc}") from exc
    return _public_book(result)
