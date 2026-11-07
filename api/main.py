"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from api.config import Settings, get_settings
from api.db import get_engine
from api.errors import ErrorBody, ErrorResponse, install_error_handlers
from api.jobs import resume_pending_jobs
from api.limits import limiter
from api.observability import RequestIdMiddleware, configure_logging, current_request_id, install_metrics
from api.routes import books, health, jobs

logger = logging.getLogger(__name__)
MULTIPART_OVERHEAD = 1024 * 1024


class BodySizeLimitMiddleware:
    """Reject oversized requests from Content-Length before reading the body (413)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") == "POST":
            length = dict(scope.get("headers") or []).get(b"content-length")
            if length is not None and length.isdigit() and int(length) > self.max_bytes:
                from fastapi.responses import JSONResponse

                body = ErrorResponse(
                    error=ErrorBody(
                        code="FILE_TOO_LARGE",
                        message=f"Uploads are limited to {self.max_bytes // (1024 * 1024)} MB.",
                        request_id=current_request_id(),
                    )
                )
                response = JSONResponse(status_code=413, content=body.model_dump(), headers={"Connection": "close"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _warm_models() -> None:
    from api.routes.health import mark_qa_models
    from src.qa.backends import TransformerBackend, get_backend

    backend = get_backend()
    mark_qa_models("loading")
    try:
        if isinstance(backend, TransformerBackend):
            backend.warm_up()
        mark_qa_models("loaded")
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        logger.error("QA model warm-up failed: %s", exc)
        mark_qa_models("failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    os.environ.setdefault("QA_INDEX_DIR", str(settings.index_dir))
    os.environ.setdefault("FLORENCE_MODE", settings.florence_mode)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        get_engine()
        resumed = resume_pending_jobs(settings)
        if resumed:
            logger.info("re-queued %d unfinished jobs", resumed)
        if settings.warm_models_on_start:
            threading.Thread(target=_warm_models, name="model-warmup", daemon=True).start()
        yield

    app = FastAPI(title="Smart Book Reader API", version="2.0.0", lifespan=lifespan)
    app.state.limiter = limiter
    install_error_handlers(app)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(books.router)
    install_metrics(app)
    return app


app = create_app()
