"""Response and request models (the v2 API contract in PLAN.md)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

BookStatus = Literal["queued", "processing", "ready", "failed"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class UploadAccepted(BaseModel):
    book_id: str
    job_id: str
    status: JobStatus


class JobOut(BaseModel):
    job_id: str
    book_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    error: str | None = None


class BookSummary(BaseModel):
    book_id: str
    filename: str
    status: BookStatus
    total_pages: int
    created_at: datetime


class BookList(BaseModel):
    books: list[BookSummary]


class PageError(BaseModel):
    code: str
    message: str


class FigureOut(BaseModel):
    figure_key: str
    figure_id: str
    label: str
    caption: str | None
    page_number: int
    image_url: str | None


class PageOut(BaseModel):
    page_number: int
    page_type: str
    status: str
    text: str
    word_count: int
    errors: list[PageError]
    figures: list[FigureOut]


class BookDetail(BookSummary):
    digital_pages: int
    scanned_pages: int
    file_url: str | None
    pages: list[PageOut]


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    page_number: int
    section: str | None
    chunk_id: str
    snippet: str
    figure_ids: list[str]


class AnswerOut(BaseModel):
    answer: str
    extractive_answer: str | None
    is_sufficient: bool
    confidence: float
    mode: Literal["offline", "gemini"]
    citations: list[CitationOut]
    figures: list[FigureOut]


class HealthOut(BaseModel):
    status: Literal["ok"]


class ReadyOut(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]
