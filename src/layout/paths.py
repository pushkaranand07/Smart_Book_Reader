"""Project path resolution — works on Windows local and Google Colab."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "training_data"
MANIFEST_PATH = TRAINING_DATA_DIR / "dataset_manifest.jsonl"
IMAGES_DIR = TRAINING_DATA_DIR / "images"
MODEL_DIR = PROJECT_ROOT / "model"
FLORENCE_LAYOUT_DIR = MODEL_DIR / "florence_layout"
RUNS_DIR = PROJECT_ROOT / "runs"
EVAL_DIR = PROJECT_ROOT / "evaluation"


def resolve_image_path(image_path: str, manifest_dir: Path | None = None) -> Path:
    """Resolve a manifest image_path to an existing file. Raises FileNotFoundError if missing."""
    p = Path(image_path)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            PROJECT_ROOT / p,
            Path.cwd() / p,
        ])
        if manifest_dir:
            candidates.append(manifest_dir / p.name)
            candidates.append(manifest_dir / "images" / p.name)

    candidates.append(IMAGES_DIR / p.name)

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        f"Image not found: '{image_path}'. Expected under {IMAGES_DIR}"
    )
