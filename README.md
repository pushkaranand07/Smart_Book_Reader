# Smart Book Reader

Upload a textbook, paper or scanned PDF. Smart Book Reader extracts the text (native or OCR), finds figures and
their real captions, and answers questions about the book with citations to the pages it used. It works fully
offline; Azure OpenAI or Gemini are optional upgrades for answer wording, and every LLM sentence is checked
against the book before it is shown.

## Architecture

```mermaid
flowchart LR
  UI[React + TypeScript SPA] -->|X-API-Key / JWT| API[FastAPI]
  API --> DB[(SQLite registry: books, jobs)]
  API -->|enqueue| Q[(huey queue on SQLite)]
  Q --> W[Worker]
  W --> P[PDFProcessor]
  P -->|native text| D[Digital pipeline: PyMuPDF text, images, vectors, captions]
  P -->|image pages| S[Scanned pipeline: RapidOCR, Tesseract fallback, OpenCV regions]
  D --> N[Normalized document: pages, figures, captions, relationships]
  S --> N
  N --> FS[(data/: uploads/&lt;sha256&gt;.pdf, extracted/, images/&lt;doc&gt;/p{page}_fig{n}.png)]
  W -->|build index| IDX[(QA index cache)]
  API -->|signed, expiring URLs| FS
  API --> QA[QA: chunk, BM25 + bge-small, RRF, MiniLM rerank, roberta-squad2 reader]
  QA -. optional .-> G[Azure OpenAI / Gemini, grounded by NLI check]
```

| Layer | Code | Notes |
| --- | --- | --- |
| Frontend | `frontend/src` | TypeScript, TanStack Query, lazy markdown/KaTeX, Vitest + Playwright |
| API | `api/` | Auth, uploads, jobs, signed media, typed errors, metrics |
| Processing | `src/pdf_processor.py`, `src/digital_pipeline.py`, `src/scanned_pipeline.py`, `src/captions.py` | Per-page routing; a failing page never aborts the book |
| QA | `src/qa/` | Retrieval, reranking, "not in this book" detection, offline + Gemini answers |
| Training | `scripts/`, `src/layout/` | Florence-2 layout fine-tuning ([docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md)) |

## Quick start (local development)

Requirements: Python 3.13, Node 24+, Tesseract (optional fallback OCR).

```bash
python -m venv .venv && . .venv/Scripts/activate         # Windows Git Bash; use .venv/bin/activate on Linux/macOS
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 torchvision==0.22.0
pip install -r requirements/dev.txt
cp .env.example .env                                       # set API_KEYS, MEDIA_SIGNING_SECRET
python scripts/fetch_models.py qa                          # ~700 MB of QA/OCR models, once
uvicorn api.main:app --port 8000                           # JOBS_MODE=process: low-priority worker, no extra command

cd frontend && npm ci && npm run dev                       # http://127.0.0.1:5173 (proxies /api to :8000)
```

Open the app, click **API key** and paste a key from `API_KEYS`, then upload a PDF.

## Docker

```bash
cp .env.example .env            # set API_KEYS and MEDIA_SIGNING_SECRET
docker compose up --build       # api :8000, worker, frontend :5173
```

The API image is multi-stage, runs as a non-root user, uses CPU-only torch and bakes the QA/OCR models in.
`api` and `worker` share the `data` volume (SQLite registry and queue), so no Redis is needed.

## Configuration

Every setting is documented in [.env.example](.env.example). The important ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTH_MODE` | `api_key` | `api_key`, `jwt`, or `none` (local only) |
| `API_KEYS` | (empty) | `owner:key` pairs; `/readyz` fails if empty in `api_key` mode |
| `MEDIA_SIGNING_SECRET` | random per process | Signs figure/PDF URLs; set it explicitly in production |
| `MAX_UPLOAD_MB`, `MAX_PAGES` | `50`, `1500` | Upload limits (413 / 422) |
| `JOBS_MODE` | `process` | `process` (low-priority worker started by the API), `huey` (separate worker service), `thread`, `immediate` (tests) |
| `FLORENCE_MODE` | `disabled` | `base` uses Florence-2-base; `required` makes `/readyz` fail without fine-tuned weights |
| `FLORENCE_HF_REPO`, `HF_TOKEN` | (empty) | Where the entrypoint downloads the fine-tuned adapter from |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` | (empty) | Optional Azure OpenAI answers; used when all three are set (wins over Gemini). `OPENAI_API_VERSION`, `AZURE_OPENAI_REASONING_EFFORT` tune it |
| `GEMINI_API_KEY` | (empty) | Optional; the app is complete without it |
| `PORT` | `8000` | Same default in the Dockerfile, compose, Vite proxy and nginx |

## API

All `/api` routes need `X-API-Key` (or `Authorization: Bearer <jwt>`). Errors are always
`{"error": {"code", "message", "request_id"}}`.

| Method | Path | Result |
| --- | --- | --- |
| POST | `/api/books` | `202 {book_id, job_id, status}`; 413 too large, 415 not a PDF, 422 too many pages |
| GET | `/api/jobs/{job_id}` | status, progress 0..1, message, error |
| GET | `/api/books`, `/api/books/{id}` | your books only (404 for anyone else's) |
| POST | `/api/books/{id}/questions` | answer, extractive_answer, is_sufficient, confidence, mode, citations, figures |
| GET | figure/PDF URLs from the responses | signed, expire after `MEDIA_URL_TTL_SECONDS` |
| GET | `/healthz`, `/readyz`, `/metrics` | liveness, readiness (DB, storage, models, Florence, auth), Prometheus |

## Security model

- No static file serving: the old public `/media` mount is gone. Figures and PDFs are served only through
  ownership-checked routes or HMAC-signed, expiring URLs. Raw extraction JSON is never served.
- API keys are stored only as SHA-256 hashes and compared in constant time; JWT mode requires `sub` and `exp`.
- Uploads are streamed with a size cap (also rejected early from `Content-Length`), checked for the `%PDF-` magic
  bytes and a page limit, and stored as `uploads/<sha256>.pdf` (the original name lives only in the database).
- Book and job IDs are random; lookups for another owner's resources return 404, not 403.
- Rate limits on upload and ask (per credential, else per IP). Request IDs on every response and log line.

## Quality evidence

QA (real models, `make eval`, golden set of 67 questions over 7 books, thresholds tuned on a separate dev set):

| Metric | Result | Target |
| --- | --- | --- |
| Retrieval recall@5 | 1.000 | >= 0.90 |
| Unanswerable detection | 14/14 = 1.000 | >= 0.90 |
| False refusals | 4/53 = 0.075 | <= 0.10 |
| Answers that echo the question | 0 | 0 |
| Answers with citations | 100% | 100% |
| Median answer latency (CPU) | ~1.5-2.3 s | |

Captions (`make eval-captions`): 88/88 correct on a labeled set of 88 figures from six digital chapters and a
scanned sample; scanned pages are read by RapidOCR, not the Tesseract fallback.

Code: `ruff check`/`ruff format` clean, `mypy --strict` clean on `api/` and the core QA modules, 86 Python tests,
90% coverage on `api/` + `src/qa/`, 46 frontend unit tests, Playwright e2e with axe (0 violations).

## Testing

```bash
make test           # Python unit + API tests (offline lexical QA backend, no model downloads)
make cov            # with coverage, fails under 85%
make lint typecheck
make eval-ci        # QA eval on the committed fixture book (real models)
make eval           # full golden set (needs cached books in data/extracted)
make eval-captions  # caption precision (needs the local PDFs in data/uploads)
cd frontend && npm run lint && npm run typecheck && npm test && npm run build && npm run test:e2e
```

## Deployment

- **Railway (API):** `railway.toml` builds `Dockerfile.api` and health-checks `/readyz`. Set `API_KEYS`,
  `MEDIA_SIGNING_SECRET`, `CORS_ORIGINS`, attach a volume at `/app/data`. Use `JOBS_MODE=process` on one service,
  or add a worker service (`/entrypoint.sh worker`, `JOBS_MODE=huey`) on the same volume.
- **Vercel / nginx (frontend):** build with `VITE_API_URL` pointing at the API, or serve behind nginx which proxies
  `/api` to `api:8000` (see `frontend/nginx.conf`).

## Licenses

All new dependencies are MIT/Apache/BSD, except: **PyMuPDF is AGPL-3.0** (already used before this work) and
`@axe-core/playwright` (dev-only) is MPL-2.0. If AGPL is a problem for a hosted deployment, the permissive
replacement is `pypdfium2` (rendering) + `pdfplumber` (text/boxes); see [PLAN.md](PLAN.md).
