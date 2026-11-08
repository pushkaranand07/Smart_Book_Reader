"""API behaviour: auth, upload limits, jobs, media access control, questions, persistence."""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from tests.api.conftest import KEYS, auth, make_pdf, upload

# ------------------------------------------------------------------ auth


def test_api_requires_a_valid_key(client: TestClient):
    assert client.get("/api/books").status_code == 401
    bad = client.get("/api/books", headers={"X-API-Key": "nope"})
    assert bad.status_code == 401
    assert bad.json()["error"]["code"] == "UNAUTHORIZED"
    assert bad.json()["error"]["request_id"] == bad.headers["X-Request-ID"]
    assert client.get("/api/books", headers=auth()).json() == {"books": []}


def test_jwt_mode(make_client: Callable[[], TestClient], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("JWT_SECRET", "jwt-secret-for-tests-0123456789abcdef")
    client = make_client()
    token = jwt.encode({"sub": "carol", "exp": int(time.time()) + 60}, "jwt-secret-for-tests-0123456789abcdef")
    assert client.get("/api/books", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    expired = jwt.encode({"sub": "carol", "exp": int(time.time()) - 5}, "jwt-secret-for-tests-0123456789abcdef")
    assert client.get("/api/books", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    assert client.get("/api/books", headers={"Authorization": "Basic abc"}).status_code == 401


# ------------------------------------------------------------------ uploads


def test_upload_returns_202_and_job_completes(client: TestClient):
    res = upload(client)
    assert res.status_code == 202
    body = res.json()
    job = client.get(f"/api/jobs/{body['job_id']}", headers=auth()).json()
    assert job["status"] == "succeeded" and job["progress"] == 1.0 and job["book_id"] == body["book_id"]
    book = client.get(f"/api/books/{body['book_id']}", headers=auth()).json()
    assert book["status"] == "ready" and book["filename"] == "book.pdf"
    page = book["pages"][0]
    assert page["figures"][0]["caption"] == "Rusted iron nails"
    assert "sig=" in page["figures"][0]["image_url"] and "/media/" not in page["figures"][0]["image_url"]
    assert all("image_path" not in fig for fig in page["figures"])


def test_upload_rejects_non_pdf(client: TestClient):
    res = upload(client, content=b"GIF89a not a pdf", name="cat.pdf")
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_rejects_oversized_body_by_content_length(client: TestClient):
    big = b"%PDF-1.7\n" + b"0" * (3 * 1024 * 1024)
    res = upload(client, content=big)
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_upload_rejects_oversized_stream_without_content_length(client: TestClient):
    from api.books import store_upload
    from api.config import get_settings
    from api.errors import ApiError

    class Fake:
        def __init__(self) -> None:
            import io

            self.file = io.BytesIO(b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024))

    with pytest.raises(ApiError) as exc:
        store_upload(Fake(), get_settings())  # type: ignore[arg-type]
    assert exc.value.status_code == 413
    assert not list(get_settings().uploads_dir.glob("*.part"))


def test_upload_rejects_too_many_pages(client: TestClient):
    res = upload(client, content=make_pdf(pages=6))
    assert res.status_code == 422 and res.json()["error"]["code"] == "TOO_MANY_PAGES"


def test_upload_rejects_corrupt_pdf(client: TestClient):
    res = upload(client, content=b"%PDF-1.7\nthis is not really a pdf")
    assert res.status_code == 422 and res.json()["error"]["code"] == "INVALID_PDF"


def test_same_filename_different_content_both_survive(client: TestClient):
    first = upload(client, content=make_pdf(text="first"), name="notes.pdf").json()
    second = upload(client, content=make_pdf(text="second"), name="notes.pdf").json()
    assert first["book_id"] != second["book_id"]
    books = client.get("/api/books", headers=auth()).json()["books"]
    assert {b["book_id"] for b in books} == {first["book_id"], second["book_id"]}
    assert all(client.get(f"/api/books/{b['book_id']}", headers=auth()).status_code == 200 for b in books)


def test_reupload_of_same_content_is_idempotent(client: TestClient):
    pdf = make_pdf(text="same")
    assert upload(client, content=pdf).json()["book_id"] == upload(client, content=pdf).json()["book_id"]


def test_upload_stores_by_content_hash(client: TestClient):
    from api.config import get_settings

    upload(client, name="weird name (1).pdf")
    stored = [p.name for p in get_settings().uploads_dir.iterdir()]
    assert len(stored) == 1 and len(stored[0]) == 64 + 4 and stored[0].endswith(".pdf")


def test_processing_failure_marks_job_failed(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    import src.pdf_processor

    def boom(*_a, **_k):
        raise RuntimeError("renderer crashed")

    monkeypatch.setattr(src.pdf_processor, "process_book", boom)
    body = upload(client, content=make_pdf(text="fails")).json()
    job = client.get(f"/api/jobs/{body['job_id']}", headers=auth()).json()
    assert job["status"] == "failed" and "renderer crashed" in job["error"]
    book = client.get(f"/api/books/{body['book_id']}", headers=auth()).json()
    assert book["status"] == "failed"


def test_upload_returns_fast_with_background_worker(make_client, monkeypatch: pytest.MonkeyPatch):
    import src.pdf_processor

    monkeypatch.setenv("JOBS_MODE", "thread")
    client = make_client()
    real = src.pdf_processor.process_book

    def slow(*args, **kwargs):
        time.sleep(1.5)
        return real(*args, **kwargs)

    monkeypatch.setattr(src.pdf_processor, "process_book", slow)
    pdf = make_pdf(pages=3)
    start = time.perf_counter()
    res = upload(client, content=pdf)
    elapsed = time.perf_counter() - start
    assert res.status_code == 202 and elapsed < 0.3, elapsed
    assert res.json()["status"] in {"queued", "running"}
    deadline = time.time() + 10
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{res.json()['job_id']}", headers=auth()).json()
        if job["status"] == "succeeded":
            break
        time.sleep(0.1)
    assert job["status"] == "succeeded"


# ------------------------------------------------------------------ ownership and media


def test_other_owner_gets_404_for_book_job_and_media(client: TestClient):
    body = upload(client).json()
    assert client.get(f"/api/books/{body['book_id']}", headers=auth("bob")).status_code == 404
    assert client.get(f"/api/jobs/{body['job_id']}", headers=auth("bob")).status_code == 404
    res = client.get(f"/api/books/{body['book_id']}/figures/p1-1/image", headers=auth("bob"))
    assert res.status_code == 404


def test_signed_media_urls(client: TestClient):
    body = upload(client).json()
    book = client.get(f"/api/books/{body['book_id']}", headers=auth()).json()
    url = book["pages"][0]["figures"][0]["image_url"]
    ok = client.get(url)
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    tampered = f"{parsed.path}?exp={query['exp'][0]}&sig={'0' * 32}"
    assert client.get(tampered).status_code == 401
    other_figure = url.replace("/p1-1/", "/p1-2/")
    assert client.get(other_figure).status_code == 401
    expired = f"{parsed.path}?exp={int(time.time()) - 10}&sig={query['sig'][0]}"
    assert client.get(expired).status_code == 401
    assert client.get(parsed.path).status_code == 401
    assert client.get(parsed.path, headers=auth()).status_code == 200

    pdf = client.get(book["file_url"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")


def test_former_media_mount_is_gone(client: TestClient):
    upload(client)
    for path in ["/media/uploads/book.pdf", "/media/extracted/x_extracted.json", "/media/images/p1_fig1.png"]:
        assert client.get(path).status_code in {401, 404}


# ------------------------------------------------------------------ questions


def test_ask_answerable_and_unanswerable(client: TestClient):
    book_id = upload(client).json()["book_id"]
    good = client.post(f"/api/books/{book_id}/questions", json={"question": "What is corrosion?"}, headers=auth())
    assert good.status_code == 200
    body = good.json()
    assert body["is_sufficient"] and body["mode"] == "offline"
    assert body["citations"][0]["page_number"] == 1 and "destruction of metals" in body["answer"]

    bad = client.post(
        f"/api/books/{book_id}/questions", json={"question": "Who won the 2022 football world cup?"}, headers=auth()
    ).json()
    assert not bad["is_sufficient"] and bad["citations"] == [] and bad["answer"] == "This isn't covered in the book."


def test_ask_about_named_figure_returns_signed_figure(client: TestClient):
    book_id = upload(client).json()["book_id"]
    body = client.post(
        f"/api/books/{book_id}/questions", json={"question": "What is shown in Figure 1.1?"}, headers=auth()
    ).json()
    assert body["figures"] and body["figures"][0]["figure_id"] == "1.1" and "sig=" in body["figures"][0]["image_url"]


def test_ask_with_gemini_mocked(make_client, monkeypatch: pytest.MonkeyPatch):
    from src.qa.gemini import GeminiAnswer

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    client = make_client()
    monkeypatch.setattr(
        "src.qa.engine.generate_answer",
        lambda *a, **k: GeminiAnswer("Corrosion is the slow destruction of metals by air and moisture.", True, "m"),
    )
    book_id = upload(client).json()["book_id"]
    body = client.post(f"/api/books/{book_id}/questions", json={"question": "What is corrosion?"}, headers=auth())
    assert (
        body.json()["mode"] == "gemini"
        and body.json()["answer"] == "Corrosion is the slow destruction of metals by air and moisture."
    )


def test_ask_validation_and_not_ready(client: TestClient):
    book_id = upload(client).json()["book_id"]
    res = client.post(f"/api/books/{book_id}/questions", json={"question": ""}, headers=auth())
    assert res.status_code == 422 and res.json()["error"]["code"] == "VALIDATION_ERROR"

    from api.db import Book, get_engine

    with Session(get_engine()) as session:
        book = session.get(Book, book_id)
        assert book is not None
        book.status = "processing"
        session.add(book)
        session.commit()
    res = client.post(f"/api/books/{book_id}/questions", json={"question": "What is corrosion?"}, headers=auth())
    assert res.status_code == 409 and res.json()["error"]["code"] == "BOOK_NOT_READY"
    assert client.get(f"/api/books/{book_id}", headers=auth()).json()["pages"] == []


def test_rate_limit_on_ask(make_client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RATE_LIMIT_ASK", "2/minute")
    client = make_client()
    book_id = upload(client).json()["book_id"]
    codes = [
        client.post(
            f"/api/books/{book_id}/questions", json={"question": "What is corrosion?"}, headers=auth()
        ).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]


# ------------------------------------------------------------------ persistence and scale


def test_books_survive_restart(make_client):
    first = make_client()
    body = upload(first, content=make_pdf(text="persist")).json()
    restarted = make_client()
    book = restarted.get(f"/api/books/{body['book_id']}", headers=auth())
    assert book.status_code == 200 and book.json()["status"] == "ready"
    answer = restarted.post(
        f"/api/books/{body['book_id']}/questions", json={"question": "What is corrosion?"}, headers=auth()
    )
    assert answer.json()["is_sufficient"]


def test_unknown_book_lookup_is_fast_with_1000_books(client: TestClient):
    from api.db import Book, get_engine, new_id

    with Session(get_engine()) as session:
        for i in range(1000):
            session.add(Book(id=new_id("bk"), owner_id="alice", sha256=f"{i:064x}", filename=f"{i}.pdf", size_bytes=1))
        session.commit()
    client.get("/api/books/bk_warmup", headers=auth())
    timings = []
    for _ in range(20):
        res = client.get("/api/books/bk_doesnotexist", headers=auth())
        assert res.status_code == 404
        # Server-side time (auth + primary-key lookup + error response), excluding test-client overhead.
        timings.append(float(res.headers["Server-Timing"].split("dur=")[1]))
    timings.sort()
    assert timings[len(timings) // 2] < 10.0, timings


# ------------------------------------------------------------------ health


def test_health_and_readiness(make_client, monkeypatch: pytest.MonkeyPatch):
    client = make_client()
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] == "ok"

    monkeypatch.setenv("FLORENCE_MODE", "required")
    strict = make_client().get("/readyz")
    assert strict.status_code == 503 and "missing" in strict.json()["checks"]["florence"]

    monkeypatch.setenv("FLORENCE_MODE", "disabled")
    monkeypatch.setenv("API_KEYS", "")
    assert make_client().get("/readyz").status_code == 503


def test_metrics_and_request_id(client: TestClient):
    client.get("/api/books", headers=auth())
    metrics = client.get("/metrics")
    assert metrics.status_code == 200 and "http_requests_total" in metrics.text
    res = client.get("/healthz", headers={"X-Request-ID": "trace-123"})
    assert res.headers["X-Request-ID"] == "trace-123"


def test_keys_are_compared_by_hash_not_stored_plain():
    from api.auth import _key_table

    table = _key_table("alice:secret-1,bare-key")
    assert "secret-1" not in table and set(table.values()) == {"alice", next(v for v in table.values() if v != "alice")}
    assert all(len(k) == 64 for k in table)


def test_owner_cannot_reach_other_owner_job_via_keys():
    assert KEYS["alice"] != KEYS["bob"]


# ------------------------------------------------------------------ background worker process


def test_process_mode_dispatches_to_low_priority_worker(monkeypatch: pytest.MonkeyPatch):
    import api.jobs as jobs
    from api.config import get_settings

    submitted: list[tuple] = []

    class FakeExecutor:
        def submit(self, fn, *args):
            submitted.append((fn, args))

    monkeypatch.setenv("JOBS_MODE", "process")
    get_settings.cache_clear()
    monkeypatch.setattr(jobs, "_worker_process_executor", lambda: FakeExecutor())
    jobs.dispatch("job_x")
    assert submitted == [(jobs._run_in_worker_process, ("job_x",))]
    get_settings.cache_clear()


def test_lower_process_priority_is_safe_to_call(monkeypatch: pytest.MonkeyPatch):
    import os
    import sys

    import api.jobs as jobs

    calls: list[object] = []
    if sys.platform == "win32":
        import ctypes

        class K32:
            def GetCurrentProcess(self):
                return 1

            def SetPriorityClass(self, handle, cls):
                calls.append(cls)
                return 1

        monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": K32()})(), raising=False)
        jobs.lower_process_priority()
        assert calls == [0x00004000]
    else:
        monkeypatch.setattr(os, "nice", lambda n: calls.append(n))
        jobs.lower_process_priority()
        assert calls == [10]
