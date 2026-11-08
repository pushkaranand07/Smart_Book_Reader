"""Download model weights.

    python scripts/fetch_models.py qa         # embedder, reranker, reader (+ RapidOCR models); used at image build
    python scripts/fetch_models.py florence   # fine-tuned Florence adapter from FLORENCE_HF_REPO (HF_TOKEN if private)

Florence: if FLORENCE_HF_REPO is unset the step is skipped; with FLORENCE_MODE=required, /readyz then reports the
missing weights and returns 503 instead of silently running without them.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch_models")


def fetch_qa() -> None:
    from huggingface_hub import snapshot_download

    from src.qa.backends import EMBED_MODEL, NLI_MODEL, READER_MODEL, RERANK_MODEL

    for repo in (EMBED_MODEL, RERANK_MODEL, READER_MODEL, NLI_MODEL):
        log.info("downloading %s", repo)
        snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "merges.txt", "vocab*"])
    try:
        from rapidocr import RapidOCR

        RapidOCR()  # downloads its ONNX detection/recognition models on first construction
        log.info("RapidOCR models ready")
    except (ImportError, OSError, RuntimeError) as exc:
        log.warning("RapidOCR model download skipped: %s", exc)


def fetch_florence() -> int:
    repo = os.environ.get("FLORENCE_HF_REPO", "").strip()
    target = ROOT / "model" / "florence_layout" / "best"
    if (target / "adapter_model.safetensors").is_file():
        log.info("Florence adapter already present at %s", target)
        return 0
    if not repo:
        log.info("FLORENCE_HF_REPO not set; skipping Florence adapter download")
        return 0
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        snapshot_download(
            repo,
            revision=os.environ.get("FLORENCE_HF_REVISION") or None,
            token=os.environ.get("HF_TOKEN") or None,
            local_dir=target,
        )
    except (HfHubHTTPError, OSError) as exc:
        log.error("Florence adapter download from %s failed: %s", repo, exc)
        return 1
    log.info("Florence adapter downloaded to %s", target)
    return 0


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "qa"
    if what == "qa":
        fetch_qa()
        return 0
    if what == "florence":
        return fetch_florence()
    log.error("unknown target %r (use qa or florence)", what)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
