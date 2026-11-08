"""Background processing of uploaded books."""

from __future__ import annotations

import logging
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import pymupdf
from sqlmodel import Session, select

from api.config import Settings, get_settings
from api.db import Book, Job, get_engine, utcnow

logger = logging.getLogger(__name__)

_executor_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_process_executor: ProcessPoolExecutor | None = None
PROCESSING_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    MemoryError,
    pymupdf.FileDataError,
)


def _update(job_id: str, **fields: Any) -> None:
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = utcnow()
        session.add(job)
        session.commit()


def _set_book(book_id: str, **fields: Any) -> None:
    with Session(get_engine()) as session:
        book = session.get(Book, book_id)
        if book is None:
            return
        for key, value in fields.items():
            setattr(book, key, value)
        book.updated_at = utcnow()
        session.add(book)
        session.commit()


def run_processing(job_id: str, settings: Settings | None = None) -> None:
    """Process one uploaded book end to end and record status/progress in the registry."""
    from src.pdf_processor import process_book
    from src.qa.engine import get_engine as get_qa_engine
    from src.storage import PIPELINE_VERSION, load_extraction_results

    settings = settings or get_settings()
    os.environ.setdefault("QA_INDEX_DIR", str(settings.index_dir))
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        book = session.get(Book, job.book_id) if job else None
        if job is None or book is None:
            logger.error("job %s or its book is missing", job_id)
            return
        book_id, sha = book.id, book.sha256

    finished = False
    try:
        _update(job_id, status="running", progress=0.02, message="Starting")
        _set_book(book_id, status="processing")
        pdf_path = settings.uploads_dir / f"{sha}.pdf"
        data = load_extraction_results(pdf_path.name, file_hash=sha)
        if data is None:

            def progress(done: int, total: int, message: str) -> None:
                _update(job_id, progress=round(0.05 + 0.85 * done / max(1, total), 3), message=message)

            data = process_book(pdf_path, progress_callback=progress)
        _update(job_id, progress=0.92, message="Indexing for questions")
        pages = (data.get("normalized_document") or {}).get("pages") or data.get("pages") or []
        get_qa_engine().index_for(sha, pages)
        _set_book(
            book_id,
            status="ready",
            total_pages=int(data.get("total_pages") or len(pages)),
            digital_pages=int(data.get("digital_pages") or 0),
            scanned_pages=int(data.get("scanned_pages") or 0),
            pipeline_version=str(data.get("pipeline_version") or PIPELINE_VERSION),
            error=None,
        )
        _update(job_id, status="succeeded", progress=1.0, message="Ready")
        finished = True
    except PROCESSING_ERRORS as exc:
        logger.exception("processing failed for job %s", job_id)
        _set_book(book_id, status="failed", error=str(exc)[:500])
        _update(job_id, status="failed", message="Processing failed", error=str(exc)[:500])
        finished = True
    finally:
        if not finished:
            # Unexpected error type: record it, then let it propagate to the worker's own logging.
            _set_book(book_id, status="failed", error="unexpected processing error")
            _update(job_id, status="failed", message="Processing failed", error="unexpected processing error")


def _thread_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="book-worker")
        return _executor


def lower_process_priority() -> None:
    """Run book processing below normal CPU priority so question answering stays responsive.

    OCR and embedding use every core; at normal priority a large upload made questions take minutes.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            below_normal = 0x00004000
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)
        else:
            os.nice(10)
    except (OSError, AttributeError) as exc:
        logger.warning("could not lower worker priority: %s", exc)


def _run_in_worker_process(job_id: str) -> None:
    run_processing(job_id)


def _worker_process_executor() -> ProcessPoolExecutor:
    global _process_executor
    with _executor_lock:
        if _process_executor is None:
            _process_executor = ProcessPoolExecutor(max_workers=1, initializer=lower_process_priority)
        return _process_executor


def shutdown_executors() -> None:
    global _process_executor, _executor
    with _executor_lock:
        if _process_executor is not None:
            _process_executor.shutdown(wait=False, cancel_futures=True)
            _process_executor = None
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None


def dispatch(job_id: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if settings.jobs_mode == "immediate":
        run_processing(job_id, settings)
    elif settings.jobs_mode == "process":
        _worker_process_executor().submit(_run_in_worker_process, job_id)
    elif settings.jobs_mode == "thread":
        _thread_executor().submit(run_processing, job_id, settings)
    else:
        from api.worker import process_job

        process_job(job_id)


def resume_pending_jobs(settings: Settings | None = None) -> int:
    """Re-dispatch jobs left queued/running by a previous process (thread/process modes have no durable queue)."""
    settings = settings or get_settings()
    if settings.jobs_mode not in {"thread", "process"}:
        return 0
    with Session(get_engine()) as session:
        pending = session.exec(select(Job).where(Job.status.in_(["queued", "running"]))).all()  # type: ignore[attr-defined]
        ids = [job.id for job in pending]
    for job_id in ids:
        _update(job_id, status="queued", message="Re-queued after restart")
        dispatch(job_id, settings)
    return len(ids)
