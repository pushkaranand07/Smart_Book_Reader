"""SQLite book/job registry (SQLModel)."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import UniqueConstraint, event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from api.config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class Book(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("owner_id", "sha256", name="uq_owner_sha"),)

    id: str = Field(primary_key=True)
    owner_id: str = Field(index=True)
    sha256: str = Field(index=True)
    filename: str
    size_bytes: int
    status: str = "queued"  # queued | processing | ready | failed
    total_pages: int = 0
    digital_pages: int = 0
    scanned_pages: int = 0
    pipeline_version: str = ""
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    book_id: str = Field(index=True)
    owner_id: str = Field(index=True)
    status: str = "queued"  # queued | running | succeeded | failed
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


_engine_lock = threading.Lock()
_engine: Engine | None = None


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def get_engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            settings = get_settings()
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            url = settings.db_url
            connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
            _engine = create_engine(url, connect_args=connect_args)
            if url.startswith("sqlite"):
                event.listen(_engine, "connect", _sqlite_pragmas)
            SQLModel.metadata.create_all(_engine)
        return _engine


def reset_engine() -> None:
    """Drop the cached engine (tests switch data directories between cases)."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
