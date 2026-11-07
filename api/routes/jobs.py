"""Job status polling."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from api.auth import Principal, require_principal
from api.db import Job, get_session
from api.errors import ErrorResponse, not_found
from api.schemas import JobOut

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobOut, responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
def get_job(
    job_id: str, principal: Principal = Depends(require_principal), session: Session = Depends(get_session)
) -> JobOut:
    job = session.get(Job, job_id)
    if job is None or job.owner_id != principal.owner_id:
        raise not_found("JOB_NOT_FOUND", "Job not found.")
    return JobOut(
        job_id=job.id,
        book_id=job.book_id,
        status=job.status,  # type: ignore[arg-type]
        progress=min(1.0, max(0.0, job.progress)),
        message=job.message,
        error=job.error,
    )
