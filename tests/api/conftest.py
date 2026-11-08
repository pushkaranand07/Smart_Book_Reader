"""Fixtures for API tests: isolated data dir, fresh registry, fake (fast) PDF processing."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

KEYS = {"alice": "key-alice-123", "bob": "key-bob-456"}

PAGE_TEXT = (
    "1.1 Corrosion\nCorrosion is the slow destruction of metals by air and moisture. "
    "Rusting of iron is a common example of corrosion. Figure 1.1 shows rusted iron nails."
)


def make_pdf(pages: int = 2, text: str = "Hello") -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{text} page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def fake_process_book(data_dir: Path) -> Callable[..., dict[str, Any]]:
    def _process(pdf_path: Path, progress_callback: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
        sha = Path(pdf_path).stem
        image_dir = data_dir / "images" / sha
        image_dir.mkdir(parents=True, exist_ok=True)
        image = image_dir / "p1_fig1.png"
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), (200, 100, 50)).save(buf, format="PNG")
        image.write_bytes(buf.getvalue())
        if progress_callback:
            progress_callback(1, 1, "Analyzing page 1 of 1")
        figure = {
            "figure_id": "1.1",
            "figure_label": "Figure 1.1",
            "caption": "Rusted iron nails",
            "page_number": 1,
            "image_path": str(image),
            "source_type": "embedded_image",
        }
        result = {
            "filename": f"{sha}.pdf",
            "total_pages": 1,
            "digital_pages": 1,
            "scanned_pages": 0,
            "pipeline_version": "test",
            "normalized_document": {
                "pages": [
                    {"page_number": 1, "page_type": "Digital", "text": PAGE_TEXT, "status": "ok", "figures": [figure]}
                ]
            },
        }
        out = data_dir / "extracted" / f"{sha}_extracted.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result), encoding="utf-8")
        return result

    return _process


@pytest.fixture
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    env = {
        "DATA_DIR": str(tmp_path / "data"),
        "AUTH_MODE": "api_key",
        "API_KEYS": ",".join(f"{owner}:{key}" for owner, key in KEYS.items()),
        "JOBS_MODE": "immediate",
        "MAX_UPLOAD_MB": "1",
        "MAX_PAGES": "5",
        "MEDIA_SIGNING_SECRET": "test-secret",
        "LOG_JSON": "false",
        "RATE_LIMIT_ASK": "100/minute",
        "RATE_LIMIT_UPLOAD": "100/minute",
        "FLORENCE_MODE": "disabled",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME", "OPENAI_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    return env


@pytest.fixture
def make_client(api_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[], TestClient]]:
    import src.pdf_processor
    import src.storage
    from api import config, db
    from api.limits import limiter

    data_dir = Path(api_env["DATA_DIR"])
    monkeypatch.setattr(src.pdf_processor, "process_book", fake_process_book(data_dir))
    monkeypatch.setattr(src.storage, "load_extraction_results", lambda *a, **k: None)
    clients: list[TestClient] = []

    def _make() -> TestClient:
        """Build a fresh app instance (simulates a process restart: new settings, engine and caches)."""
        config.get_settings.cache_clear()
        db.reset_engine()
        from api.routes import books

        books.book_cache.clear()
        limiter.reset()
        from api.main import create_app

        client = TestClient(create_app())
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
    config.get_settings.cache_clear()
    db.reset_engine()


@pytest.fixture
def client(make_client: Callable[[], TestClient]) -> TestClient:
    return make_client()


def auth(owner: str = "alice") -> dict[str, str]:
    return {"X-API-Key": KEYS[owner]}


def upload(client: TestClient, content: bytes | None = None, name: str = "book.pdf", owner: str = "alice") -> Any:
    return client.post(
        "/api/books", files={"file": (name, content or make_pdf(), "application/pdf")}, headers=auth(owner)
    )
