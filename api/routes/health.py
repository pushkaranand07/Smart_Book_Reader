"""Liveness and readiness probes."""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from api.config import Settings, get_settings
from api.db import get_engine
from api.schemas import HealthOut, ReadyOut

router = APIRouter(tags=["health"])
_model_state = {"qa": "lazy"}


def mark_qa_models(state: str) -> None:
    _model_state["qa"] = state


def florence_check(settings: Settings) -> tuple[bool, str]:
    from src.layout.paths import FLORENCE_LAYOUT_DIR

    adapter = FLORENCE_LAYOUT_DIR / "best" / "adapter_model.safetensors"
    if settings.florence_mode == "disabled":
        return True, "disabled"
    if adapter.is_file():
        return True, "fine-tuned adapter present"
    if settings.florence_mode == "base":
        return True, "base model (no fine-tuned adapter)"
    return False, f"missing fine-tuned weights at {adapter}"


@router.get("/healthz", response_model=HealthOut)
def healthz() -> HealthOut:
    return HealthOut(status="ok")


@router.get("/readyz", response_model=ReadyOut, responses={503: {"model": ReadyOut}})
def readyz(settings: Settings = Depends(get_settings)) -> JSONResponse:
    checks: dict[str, str] = {}
    ok = True
    try:
        with Session(get_engine()) as session:
            session.exec(text("SELECT 1"))  # type: ignore[call-overload]
        checks["database"] = "ok"
    except SQLAlchemyError as exc:
        ok, checks["database"] = False, f"error: {exc.__class__.__name__}"
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.data_dir, delete=True):
            pass
        checks["storage"] = "ok"
    except OSError as exc:
        ok, checks["storage"] = False, f"error: {exc.strerror}"
    florence_ok, checks["florence"] = florence_check(settings)
    ok &= florence_ok
    checks["qa_models"] = _model_state["qa"]
    if _model_state["qa"] == "failed":
        ok = False
    if settings.auth_mode == "api_key" and not settings.api_keys:
        ok, checks["auth"] = False, "no API_KEYS configured"
    else:
        checks["auth"] = settings.auth_mode
    checks["jobs"] = settings.jobs_mode
    azure = settings.azure_config
    checks["llm"] = (
        f"azure_openai ({azure.deployment})" if azure else ("gemini" if settings.gemini_api_key else "offline only")
    )
    checks["pid"] = str(os.getpid())
    body = ReadyOut(status="ready" if ok else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())
