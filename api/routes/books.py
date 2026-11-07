"""Book upload, listing, detail, questions and signed media."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from api.auth import Principal, principal_from_request, require_principal
from api.books import BookDataCache, book_detail, count_pages, figure_index, figure_out, store_upload
from api.config import Settings, get_settings
from api.db import Book, Job, get_session, new_id
from api.errors import ApiError, ErrorResponse, not_found
from api.jobs import dispatch
from api.limits import limiter
from api.schemas import AnswerOut, BookDetail, BookList, BookSummary, CitationOut, QuestionIn, UploadAccepted
from api.security import safe_data_path, verify

router = APIRouter(prefix="/api", tags=["books"])
book_cache = BookDataCache(size=get_settings().book_cache_size)
ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 404, 409, 413, 415, 422, 429)
}


def _owned_book(session: Session, book_id: str, owner_id: str) -> Book:
    book = session.get(Book, book_id)
    if book is None or book.owner_id != owner_id:
        # Same response for "missing" and "someone else's" so ids cannot be probed.
        raise not_found("BOOK_NOT_FOUND", "Book not found.")
    return book


def _summary(book: Book) -> BookSummary:
    return BookSummary(
        book_id=book.id,
        filename=book.filename,
        status=book.status,
        total_pages=book.total_pages,
        created_at=book.created_at,
    )


@router.post("/books", status_code=202, response_model=UploadAccepted, responses=ERRORS)
@limiter.limit(lambda: get_settings().rate_limit_upload)
def upload_book(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(require_principal),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> UploadAccepted:
    sha, size, path = store_upload(file, settings)
    pages = count_pages(path)
    if pages > settings.max_pages:
        raise ApiError(422, "TOO_MANY_PAGES", f"PDFs are limited to {settings.max_pages} pages (this one has {pages}).")
    filename = (file.filename or "document.pdf")[:255]

    existing = session.exec(select(Book).where(Book.owner_id == principal.owner_id, Book.sha256 == sha)).first()
    if existing is not None and existing.status != "failed":
        job = session.exec(select(Job).where(Job.book_id == existing.id).order_by(Job.created_at.desc())).first()  # type: ignore[attr-defined]
        if job is not None:
            return UploadAccepted(book_id=existing.id, job_id=job.id, status=job.status)

    book = existing or Book(
        id=new_id("bk"), owner_id=principal.owner_id, sha256=sha, filename=filename, size_bytes=size, total_pages=pages
    )
    book.status, book.error = "queued", None
    job = Job(id=new_id("job"), book_id=book.id, owner_id=principal.owner_id)
    session.add(book)
    session.add(job)
    try:
        session.commit()
    except IntegrityError as exc:  # concurrent duplicate upload by the same owner
        session.rollback()
        raise ApiError(409, "DUPLICATE_UPLOAD", "This file is already being processed.") from exc
    dispatch(job.id, settings)
    session.refresh(job)
    return UploadAccepted(book_id=book.id, job_id=job.id, status=job.status)


@router.get("/books", response_model=BookList, responses=ERRORS)
def list_books(principal: Principal = Depends(require_principal), session: Session = Depends(get_session)) -> BookList:
    books = session.exec(
        select(Book).where(Book.owner_id == principal.owner_id).order_by(Book.created_at.desc())  # type: ignore[attr-defined]
    ).all()
    return BookList(books=[_summary(b) for b in books])


@router.get("/books/{book_id}", response_model=BookDetail, responses=ERRORS)
def get_book(
    book_id: str,
    principal: Principal = Depends(require_principal),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> BookDetail:
    book = _owned_book(session, book_id, principal.owner_id)
    if book.status != "ready":
        return BookDetail(**_summary(book).model_dump(), digital_pages=0, scanned_pages=0, file_url=None, pages=[])
    return book_detail(settings, book, book_cache.get(settings, book.sha256))


@router.post("/books/{book_id}/questions", response_model=AnswerOut, responses=ERRORS)
@limiter.limit(lambda: get_settings().rate_limit_ask)
def ask_question(
    request: Request,
    book_id: str,
    body: QuestionIn,
    principal: Principal = Depends(require_principal),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> AnswerOut:
    from src.qa_engine import answer_question

    book = _owned_book(session, book_id, principal.owner_id)
    if book.status != "ready":
        raise ApiError(409, "BOOK_NOT_READY", "The book has not finished processing.")
    data = book_cache.get(settings, book.sha256)
    result = answer_question(
        body.question, {**data, "book_id": book.sha256, "filename": book.filename}, api_key=settings.gemini_api_key
    )
    figures_by_ref = {
        (int(f.get("page_number") or 0), str(f.get("figure_id"))): k for k, f in figure_index(data).items()
    }
    index = figure_index(data)
    figures = []
    for fig in result.get("figures", []):
        key = figures_by_ref.get((int(fig.get("page_number") or 0), str(fig.get("figure_id"))))
        if key is not None:
            figures.append(figure_out(settings, book.id, key, index[key]))
    return AnswerOut(
        answer=result["answer"],
        extractive_answer=result.get("extractive_answer"),
        is_sufficient=bool(result["is_sufficient"]),
        confidence=float(result.get("confidence") or 0.0),
        mode=result.get("mode", "offline"),
        citations=[CitationOut(**c) for c in result.get("citations", [])] if result["is_sufficient"] else [],
        figures=figures,
    )


def _media_access(
    request: Request, settings: Settings, session: Session, book_id: str, path: str, exp: int | None, sig: str | None
) -> Book:
    """Allow a valid signed URL, or an authenticated owner. Everything else is 401/404."""
    book = session.get(Book, book_id)
    if verify(settings, path, exp, sig):
        if book is None:
            raise not_found("BOOK_NOT_FOUND", "Book not found.")
        return book
    principal = principal_from_request(request, settings)
    if principal is None:
        raise ApiError(401, "UNAUTHORIZED", "A valid signed URL or API key is required.")
    return _owned_book(session, book_id, principal.owner_id)


@router.get("/books/{book_id}/figures/{figure_key}/image", responses=ERRORS)
def figure_image(
    request: Request,
    book_id: str,
    figure_key: str,
    exp: int | None = None,
    sig: str | None = None,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> FileResponse:
    path = f"/api/books/{book_id}/figures/{figure_key}/image"
    book = _media_access(request, settings, session, book_id, path, exp, sig)
    fig = figure_index(book_cache.get(settings, book.sha256)).get(figure_key)
    image = safe_data_path(settings, str(fig.get("image_path"))) if fig and fig.get("image_path") else None
    if image is None:
        raise not_found("FIGURE_NOT_FOUND", "Figure not found.")
    return FileResponse(image, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/books/{book_id}/file", responses=ERRORS)
def book_file(
    request: Request,
    book_id: str,
    exp: int | None = None,
    sig: str | None = None,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> FileResponse:
    book = _media_access(request, settings, session, book_id, f"/api/books/{book_id}/file", exp, sig)
    pdf = settings.uploads_dir / f"{book.sha256}.pdf"
    if not pdf.is_file():
        raise not_found("FILE_NOT_FOUND", "File not found.")
    return FileResponse(pdf, media_type="application/pdf", filename=book.filename)
