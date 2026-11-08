# Final Report: Production Hardening

Branch `production-hardening` is fully merged into `main` (`b5cd4e2` and later). CI on `main` is green: backend, frontend and Docker jobs.

## Scores by area: before → after

The "after" scores are my honest assessment. An area only gets 10 when every criterion is met *and* the
evidence is independent of me. Several criteria below were verified only on test sets I built myself.

| Area | Before | After | Why it is not higher |
| --- | --- | --- | --- |
| QA quality | 3 | **8** | 3 answerable questions still refused; LLM answers on famous books get trimmed to 1-2 sentences by the grounding filter; eval set is mine |
| API security and robustness | 3 | **8.5** | Single-host SQLite; `AUTH_MODE=none` exists for local use and is unsafe if deployed |
| PDF extraction (digital) | 7.5 | **8.5** | Caption labels were derived from pipeline output, then verified against raw PDF text, which is weaker than hand labels |
| Scanned / OCR | 5.5 | **7** | Only a 3-page scanned sample was labeled; drop caps (the decorative first letter of a chapter) get misread by OCR; no learned layout model |
| Frontend | 4 | **8.5** | Lighthouse was run before the last small change (`robots.txt`) and not re-run |
| Deployment | n/a | **7** | `docker compose up` never run end to end (no Docker on the dev machine); Railway untested; Florence `/readyz` check unreachable (no weights exist) |
| Code quality and tests | 5 | **8.5** | Legacy pipeline modules are excluded from `mypy --strict` |
| **Overall** | **5** | **8** | |

## Evidence per criterion

### Phase 1: QA (`make eval`, real models, 78 questions over 7 books, thresholds tuned only on the separate dev set)

| Criterion | Target | Result |
| --- | --- | --- |
| Retrieval recall@5 | ≥ 0.90 | **1.000** |
| Unanswerable detection | ≥ 0.90 | **17/17 = 1.000** |
| False refusals | ≤ 10% | **3/61 = 4.9%** (was 8.2% before the comparison-question fix) |
| Answers echoing the question | 0 | **0** |
| Answers with citations | 100% | **100%** |
| Latency (CPU, offline) | n/a | p50 **1.3 s**, p95 **3.3 s** |

- Hybrid retrieval: BM25 + `bge-small-en-v1.5` fused with RRF, then `ms-marco-MiniLM-L6-v2` reranks the top 20.
- "Not in this book" detection: the reranker threshold combined with the `roberta-base-squad2` null-answer margin.
- Comparison questions ("difference between X and Y") are split per concept. All 8 comparison questions in the golden set are answered, covering both sides.
- LLM answers are verified sentence by sentence with `nli-deberta-v3-small`, and unsupported sentences are removed. Found live: gpt-5 explained why magnesium ribbon is cleaned, which the chapter never states. It is now refused.
- The Gemini path is covered by mocked tests (success, self-check, retry, error fallback, hard 60 s deadline).
- The prompt is document-agnostic.

### Phase 2: API (`tests/api`, 29 tests)

| Criterion | Result |
| --- | --- |
| Unauthenticated GET on former `/media` URLs returns 401/404 | **404**; figures and PDFs only via ownership checks or HMAC-signed, expiring URLs |
| 200 MB upload rejected with 413 | **413** from `Content-Length` before the body is read, also enforced while streaming |
| Upload returns in < 300 ms | **0.12-0.37 s** measured live (test asserts < 0.3 s with a slow fake processor) |
| Books survive restart | **Yes** (test + live restart, interrupted job resumed) |
| Unknown-ID lookup < 10 ms at 1,000 books | **~6 ms** server-side (`Server-Timing`) |
| Same-name files both survive | **Yes**, stored as `uploads/<sha256>.pdf` |

- Book processing runs in a low-priority worker process. Questions asked while a 580-page book was processing took **8-15 s**, versus **134.7 s** before this fix.

### Phase 3: Extraction and OCR (`make eval-captions`)

- **Caption precision 88/88** on 88 figures (six digital chapters plus a scanned sample); **0** bullet glyphs or filename junk in captions.
- Scanned sample read by **RapidOCR** (quality 0.97-0.98), not the Tesseract fallback.
- Filenames are `images/<doc>/p{page}_fig{n}.png`. Figures without a real caption get `caption: null`.

### Phase 4: Frontend

- TypeScript strict, TanStack Query, lazy-loaded markdown/KaTeX, lightbox and full-text panels.
- **Initial JS 88.64 kB gzip** (limit 200 kB), no Vite warnings.
- ESLint and Prettier clean. **46** Vitest tests.
- Playwright end-to-end passes **mocked** and **against the real backend**.
- axe: **0 violations**. Lighthouse accessibility **100**.

### Phase 5: Deployment

- **API image 1,583 MB** (limit 2.5 GB), frontend image **21 MB**, measured in CI.
- The CI container smoke test passes: healthz, readyz, 401 without a key, 404 on `/media`.
- Multi-stage, non-root image with CPU-only torch; models baked in at build time.
- Split pinned requirements, `.env.example`, a single `PORT` setting.
- GitHub Actions: lint, types, tests with an 85% coverage gate, QA eval, frontend, Docker.

### Phase 6: Code quality

- `ruff check` / `ruff format`: **574 → 0**.
- `mypy --strict` on `api/` and the QA package: **0 errors** (32 files).
- Bare `except Exception`: **38 → 0**. The single catch-all is the API's last-resort handler, which logs and returns a typed 500.
- **106** Python tests. Coverage on `api/` + `src/qa/`: **89.8%** (CI gate 85%).
- pre-commit hooks; README rewritten.

## Not reached, and why

1. **Florence fine-tuned model.** `model/florence_layout/` was empty, so there were no weights to ship or load. The download hook (`FLORENCE_HF_REPO`) and the `/readyz` check exist, but "readyz green with Florence loaded" cannot be shown.
2. **`docker compose up` from a clean clone.** Docker is not installed on the dev machine. CI builds both images and smoke-tests the API container, but has never run the three-container compose stack.
3. **Railway deploy.** Configured (`railway.toml`) but never deployed.
4. **PyMuPDF is AGPL-3.0.** It is flagged, and a permissive replacement (pypdfium2 + pdfplumber) is proposed, but it was not migrated because that touches every pipeline module.
5. **Scanned-page layout.** Captions on scanned pages use OCR line geometry plus regex rules, not a learned layout model.
6. **Azure OpenAI.** It was tested live with gpt-5 but deliberately **not committed**: the key was temporary, at your request. The engine accepts any `LLMProvider`, so adding Azure later is a small change.
7. **Open oddities I could not explain:**
   - One full test-suite run stalled past 10 minutes once; it did not reproduce in 3 reruns.
   - One Azure call took about 1 hour. Likely laptop sleep; it is now capped by the 60 s deadline.
   - Backend start takes about 4 minutes, because the model library checks Hugging Face online for models already on disk. Setting `HF_HUB_OFFLINE=1` would likely fix it, but I have not tested it.

## New dependencies and licenses

**Python (runtime):**

- Retrieval and models: bm25s (MIT), sentence-transformers (Apache-2.0), transformers (Apache-2.0), torch CPU (BSD-3), peft (Apache-2.0), timm (Apache-2.0), einops (MIT)
- OCR and images: rapidocr (Apache-2.0), onnxruntime (MIT), opencv-python-headless (Apache-2.0)
- API and storage: sqlmodel (MIT), SQLAlchemy (MIT), pydantic-settings (MIT), PyJWT (MIT)
- Jobs and limits: huey (MIT), slowapi (MIT), limits (MIT), tenacity (Apache-2.0)
- Observability: structlog (MIT/Apache-2.0), prometheus-fastapi-instrumentator (ISC), prometheus_client (Apache-2.0)

**Python (dev):** pytest (MIT), pytest-cov (MIT), ruff (MIT), mypy (MIT), pre-commit (MIT).

**Models:**

- BAAI/bge-small-en-v1.5 (MIT)
- cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0)
- deepset/roberta-base-squad2 (CC-BY-4.0)
- cross-encoder/nli-deberta-v3-small (Apache-2.0)
- PP-OCR models via RapidOCR (Apache-2.0)

**Frontend:**

- Runtime: @tanstack/react-query, react, react-dom, react-markdown, remark-math, rehype-katex, katex (all MIT); @fontsource fonts (OFL-1.1)
- Dev: typescript, vite, vitest, jsdom, testing-library, eslint, prettier (MIT/Apache-2.0); @axe-core/playwright (**MPL-2.0**, dev-only)

**Pre-existing, flagged:** PyMuPDF (**AGPL-3.0**).

## Commands

```bash
# run locally (backend on PORT from .env; JOBS_MODE=process starts the worker automatically)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 torchvision==0.22.0
pip install -r requirements/dev.txt
cp .env.example .env              # set API_KEYS + MEDIA_SIGNING_SECRET (or AUTH_MODE=none for local-only use)
python scripts/fetch_models.py qa
uvicorn api.main:app --port 8000
cd frontend && npm ci && VITE_API_PORT=8000 npm run dev    # http://127.0.0.1:5173

# test
make lint typecheck test cov
make eval-ci                      # fixture book, real models
make eval                         # golden set (needs data/extracted)
make eval-captions                # needs local PDFs
cd frontend && npm run lint && npm run typecheck && npm test && npm run build && npm run test:e2e

# deploy locally with Docker
docker compose up --build         # api :8000, worker, frontend :5173
```
