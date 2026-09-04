"""
build_doclaynet.py — Download DocLayNet and convert to Florence Picture-detection format.

Dataset: docling-project/DocLayNet (or ds4sd/DocLayNet) on Hugging Face
  - ~80k real document pages (financial, scientific, manuals, patents, etc.)
  - Human-annotated layout boxes including Picture
  - Official train / val / test splits (NO synthetic leakage)

This script:
  1. Streams DocLayNet from Hugging Face (no full 30GB download required)
  2. By default keeps Picture pages; with --include-negative also keeps No-Picture pages
  3. Converts COCO xywh boxes -> xyxy pixels
  4. Writes data/training_data/{images,dataset_manifest.jsonl}
  5. Labels used for Florence: Picture only (Caption/Table/Text ignored)

Usage:
    .\\venv\\Scripts\\python.exe scripts\\dataset\\build_doclaynet.py --custom-split --include-negative --max-train 30000 --max-val 10000 --max-test 5000
  .\\venv\\Scripts\\python.exe scripts\\dataset\\build_doclaynet.py --pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.paths import PROJECT_ROOT as ROOT, TRAINING_DATA_DIR, IMAGES_DIR, MANIFEST_PATH

# DocLayNet COCO category ids
CATEGORY_ID_TO_NAME = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}
PICTURE_ID = 7


def coco_xywh_to_xyxy(
    bbox: List[float],
    width: int,
    height: int,
    bbox_format: str = "auto",
) -> Optional[List[float]]:
    """Convert a box to clamped xyxy pixels.

    DocLayNet-v1.1 often stores absolute xyxy; classic COCO uses xywh.
    Heuristic: if x+w and y+h fit the page, treat as xywh; otherwise xyxy.
    """
    if len(bbox) != 4:
        return None
    a, b, c, d = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Normalized 0-1
    if max(a, b, c, d) <= 1.5:
        a, b, c, d = a * width, b * height, c * width, d * height

    if bbox_format == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
    elif bbox_format == "xywh":
        x1, y1, x2, y2 = a, b, a + c, b + d
    else:
        looks_xywh = (a + c) <= width * 1.02 and (b + d) <= height * 1.02 and c > 0 and d > 0
        looks_xyxy = c > a and d > b

        if looks_xywh and not (looks_xyxy and (c > width * 0.5 or d > height * 0.5) and (a + c) > width):
            x1, y1, x2, y2 = a, b, a + c, b + d
        elif looks_xyxy:
            x1, y1, x2, y2 = a, b, c, d
        else:
            x1, y1, x2, y2 = a, b, a + c, b + d

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _item_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("metadata")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str) and meta.strip():
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _to_pil_rgb(image_obj: Any):
    from PIL import Image
    import io

    if image_obj is None:
        return None
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict):
        raw = image_obj.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = image_obj.get("path")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(image_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    return None


def _is_picture_category(cid: Any, label: Any = "") -> bool:
    if str(label) == "Picture":
        return True
    if cid is None:
        return False
    cid_i = int(cid)
    # COCO 1-based id 7, or HF ClassLabel 0-based index 6
    return cid_i in (PICTURE_ID, PICTURE_ID - 1)


def extract_picture_boxes(item: Dict[str, Any], width: int, height: int) -> List[Dict[str, Any]]:
    """Extract Picture annotations from various DocLayNet HF schemas."""
    pictures: List[Dict[str, Any]] = []

    # Schema A: parallel category_id + bboxes lists (DocLayNet-v1.1 style)
    if "category_id" in item and "bboxes" in item:
        cats = item["category_id"]
        boxes = item["bboxes"]
        if isinstance(cats, list) and isinstance(boxes, list):
            for cid, bbox in zip(cats, boxes):
                if not _is_picture_category(cid):
                    continue
                xyxy = coco_xywh_to_xyxy(list(bbox), width, height, bbox_format="xyxy")
                if xyxy:
                    pictures.append({"category": "Picture", "bbox": xyxy})
            if pictures:
                return pictures

    # Schema used by pierreguillou/DocLayNet-small processed annotations.
    categories = item.get("categories")
    block_boxes = item.get("bboxes_block")
    if isinstance(categories, list) and isinstance(block_boxes, list):
        for category, bbox in zip(categories, block_boxes):
            if not _is_picture_category(None, category):
                continue
            xyxy = coco_xywh_to_xyxy(list(bbox), width, height, bbox_format="xywh")
            if xyxy:
                pictures.append({"category": "Picture", "bbox": xyxy})
        if pictures:
            return _dedupe_annotations(pictures)

    # Schema B: objects list (ds4sd / docling-project DocLayNet)
    objects = item.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label") or obj.get("category") or ""
            if not _is_picture_category(obj.get("category_id"), label):
                continue
            bbox = obj.get("bbox") or obj.get("box")
            if not bbox or len(bbox) != 4:
                continue
            xyxy = coco_xywh_to_xyxy(list(bbox), width, height, bbox_format="xyxy")
            if xyxy:
                pictures.append({"category": "Picture", "bbox": xyxy})
        if pictures:
            return pictures

    # Schema C: bboxes_block / segments / annotations
    for key in ("bboxes_block", "segments", "annotations"):
        raw = item.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        for b in raw:
            if not isinstance(b, dict):
                continue
            label = b.get("label") or b.get("category") or ""
            cid = b.get("category_id")
            if cid is not None and not label:
                label = CATEGORY_ID_TO_NAME.get(int(cid), CATEGORY_ID_TO_NAME.get(int(cid) + 1, ""))
            if not _is_picture_category(cid, label):
                continue
            bbox = b.get("bbox") or b.get("box")
            if not bbox or len(bbox) != 4:
                continue
            xyxy = coco_xywh_to_xyxy(list(bbox), width, height)
            if xyxy:
                pictures.append({"category": "Picture", "bbox": xyxy})
        if pictures:
            return pictures

    return pictures


def _dedupe_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for annotation in annotations:
        key = (annotation.get("category"), tuple(annotation.get("bbox", [])))
        if key in seen:
            continue
        seen.add(key)
        unique.append(annotation)
    return unique


def load_small_doclaynet_split(split: str):
    """Load the 1% processed archive without relying on a dataset script."""
    from huggingface_hub import hf_hub_download

    split_name = "val" if split in ("val", "validation") else split
    cache_dir = ROOT / "data" / ".cache" / "doclaynet-small"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id="pierreguillou/DocLayNet-small",
            repo_type="dataset",
            filename="data/dataset_small.zip",
        )
    )
    extracted_root = cache_dir / "small_dataset"
    if not extracted_root.exists():
        print(f"  Extracting {archive.name} ...", flush=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache_dir)

    split_root = extracted_root / split_name
    if not split_root.exists():
        raise RuntimeError(f"Small DocLayNet split not found: {split_name}")

    def items():
        annotations_dir = split_root / "annotations"
        images_dir = split_root / "images"
        for annotation_path in sorted(annotations_dir.glob("*.json")):
            with annotation_path.open("r", encoding="utf-8") as f:
                annotation = json.load(f)
            image_path = images_dir / f"{annotation_path.stem}.png"
            if not image_path.exists():
                continue
            form = annotation.get("form", [])
            metadata = annotation.get("metadata", {})
            yield {
                "categories": [entry.get("category") for entry in form],
                "bboxes_block": [entry.get("box") for entry in form],
                "image": {"path": str(image_path)},
                "metadata": metadata,
                "id": annotation_path.stem,
            }

    return items(), "pierreguillou/DocLayNet-small"


def load_doclaynet_split(split: str, pilot: bool = False):
    """Load DocLayNet from parquet-based HF repos (no dataset scripts).

    Modern `datasets` versions reject DocLayNet.py loading scripts.
    Use DocLayNet-v1.1 parquet only.
    """
    if pilot:
        return load_small_doclaynet_split(split)

    from datasets import load_dataset

    # HF split naming: some repos use 'val', some 'validation'
    split_aliases = {
        "train": ["train"],
        "val": ["val", "validation"],
        "validation": ["validation", "val"],
        "test": ["test"],
    }
    names_to_try = split_aliases.get(split, [split])

    # Parquet-only repos first (no DocLayNet.py)
    repos = (
        "ds4sd/DocLayNet-v1.1",
        "docling-project/DocLayNet-v1.1",
    )

    last_err: Exception | None = None
    for repo in repos:
        for sp in names_to_try:
            try:
                print(f"  Trying {repo} split={sp} (streaming parquet) ...")
                ds = load_dataset(repo, split=sp, streaming=True)
                first = next(iter(ds))
                print(f"  OK: {repo} keys={list(first.keys())[:12]}")
                # Fresh stream for the collector
                return load_dataset(repo, split=sp, streaming=True), repo
            except Exception as e:
                last_err = e
                print(f"  Failed: {e}")

    # Explicit parquet file glob fallback
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        for repo in repos:
            for sp in names_to_try:
                patterns = [
                    f"datasets/{repo}/data/{sp}*.parquet",
                    f"datasets/{repo}/**/{sp}*.parquet",
                    f"datasets/{repo}/{sp}*.parquet",
                ]
                files: list[str] = []
                for pat in patterns:
                    try:
                        files = list(fs.glob(pat))
                    except Exception:
                        files = []
                    if files:
                        break
                if not files:
                    continue
                hf_paths = [f"hf://{f}" if not f.startswith("hf://") else f for f in files]
                print(f"  Parquet glob: {repo} split={sp} ({len(hf_paths)} files)")
                ds = load_dataset("parquet", data_files={sp: hf_paths}, split=sp, streaming=True)
                return ds, repo
    except Exception as e:
        last_err = e
        print(f"  Parquet fallback failed: {e}")

    raise RuntimeError(
        f"Could not load DocLayNet parquet. Last error: {last_err}\n"
        "Tip: pip install -U datasets huggingface_hub"
    )


def collect_split(
    split: str,
    max_samples: int,
    images_dir: Path,
    seed: int = 42,
    pilot: bool = False,
    include_negative: bool = False,
) -> List[Dict[str, Any]]:
    page_kind = "pages" if include_negative else "Picture pages"
    print(f"\n=== Collecting split={split} (max {page_kind}={max_samples}) ===")
    ds, repo = load_doclaynet_split(split, pilot=pilot)
    print(f"Source: {repo}")

    records: List[Dict[str, Any]] = []
    scanned = 0
    rng = random.Random(seed + hash(split) % 10000)

    for item in ds:
        scanned += 1
        if len(records) >= max_samples:
            break

        if scanned == 1:
            print("  First page received — extracting Picture boxes...", flush=True)

        pil_img = _to_pil_rgb(item.get("image"))
        if pil_img is None:
            continue
        w, h = pil_img.size

        pictures = extract_picture_boxes(item, w, h)
        if not pictures and not include_negative:
            continue

        meta = _item_metadata(item)
        doc_cat = (
            meta.get("doc_category")
            or item.get("doc_category")
            or meta.get("collection")
            or item.get("collection")
            or "unknown"
        )

        idx = len(records) + 1
        fname = f"doclaynet_{split}_{idx:05d}.png"
        fpath = images_dir / fname
        pil_img.save(str(fpath))

        source_id = (
            meta.get("doc_name")
            or item.get("doc_name")
            or meta.get("file_name")
            or item.get("file_name")
            or meta.get("original_filename")
            or item.get("original_filename")
            or item.get("image_id")
            or f"{doc_cat}_{split}_{idx}"
        )
        source_id = str(source_id)

        rec = {
            "id": f"DocLayNet_{split}_{idx:05d}",
            "dataset": "DocLayNet",
            "split": "train" if split == "train" else ("val" if split in ("val", "validation") else "test"),
            "image_filename": fname,
            "image_path": f"data/training_data/images/{fname}",
            "width": w,
            "height": h,
            "source_id": source_id,
            "doc_category": doc_cat,
            "annotations": pictures,
        }
        records.append(rec)

        if len(records) % 50 == 0:
            print(f"  kept {len(records)}/{max_samples} (scanned {scanned})", flush=True)

    print(f"Split {split}: kept {len(records)} Picture pages (scanned {scanned})")
    return records


def _source_id_for_item(item: Dict[str, Any], meta: Dict[str, Any], fallback: str) -> str:
    return str(
        meta.get("doc_name")
        or item.get("doc_name")
        or meta.get("file_name")
        or item.get("file_name")
        or meta.get("original_filename")
        or item.get("original_filename")
        or item.get("image_id")
        or item.get("id")
        or fallback
    )


def _save_record(
    item: Dict[str, Any],
    split: str,
    idx: int,
    images_dir: Path,
    include_negative: bool,
) -> Optional[Dict[str, Any]]:
    pil_img = _to_pil_rgb(item.get("image"))
    if pil_img is None:
        return None
    w, h = pil_img.size
    pictures = extract_picture_boxes(item, w, h)
    if not pictures and not include_negative:
        return None

    meta = _item_metadata(item)
    doc_cat = (
        meta.get("doc_category")
        or item.get("doc_category")
        or meta.get("collection")
        or item.get("collection")
        or "unknown"
    )
    fname = f"doclaynet_{split}_{idx:05d}.png"
    pil_img.save(str(images_dir / fname))
    source_id = _source_id_for_item(item, meta, f"{doc_cat}_{split}_{idx}")
    return {
        "id": f"DocLayNet_{split}_{idx:05d}",
        "dataset": "DocLayNet",
        "split": split,
        "image_filename": fname,
        "image_path": f"data/training_data/images/{fname}",
        "width": w,
        "height": h,
        "source_id": source_id,
        "doc_category": doc_cat,
        "annotations": pictures,
    }


def collect_custom_train_val(
    max_train: int,
    max_val: int,
    images_dir: Path,
    seed: int = 42,
    include_negative: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create document-level train/val quotas from one official train stream."""
    print(
        f"\n=== Collecting custom train/val split (train={max_train}, val={max_val}) ==="
    )
    ds, repo = load_doclaynet_split("train")
    print(f"Source: {repo}")
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    scanned = 0
    val_bucket_count = 30
    source_destinations: Dict[str, str] = {}

    for item in ds:
        scanned += 1
        if len(train_records) >= max_train and len(val_records) >= max_val:
            break
        meta = _item_metadata(item)
        source_id = _source_id_for_item(item, meta, f"train_item_{scanned}")
        destination = source_destinations.get(source_id)
        if destination is None:
            bucket = int(hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
            destination = "val" if bucket < val_bucket_count else "train"
        if destination == "val" and len(val_records) >= max_val:
            if source_id in source_destinations:
                continue
            destination = "train"
        if destination == "train" and len(train_records) >= max_train:
            if source_id in source_destinations:
                continue
            destination = "val"
        if destination == "val" and len(val_records) >= max_val:
            continue

        target = val_records if destination == "val" else train_records
        record = _save_record(item, destination, len(target) + 1, images_dir, include_negative)
        if record is None:
            continue
        source_destinations[source_id] = destination
        target.append(record)
        if len(target) % 1000 == 0:
            print(
                f"  train={len(train_records)}/{max_train}, "
                f"val={len(val_records)}/{max_val} (scanned {scanned})",
                flush=True,
            )

    print(
        f"Custom split: train={len(train_records)}, val={len(val_records)} "
        f"(scanned {scanned})"
    )
    return train_records, val_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DocLayNet Picture training set")
    parser.add_argument("--pilot", action="store_true", help="Small set: 400 train / 100 val")
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--max-val", type=int, default=10000)
    parser.add_argument("--max-test", type=int, default=5000, help="0 = skip test split")
    parser.add_argument(
        "--custom-split",
        action="store_true",
        help="Build train/val quotas from official train by document-level source hash",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Keep pages without Picture annotations (recommended for large training sets)",
    )
    parser.add_argument("--output-dir", default=str(TRAINING_DATA_DIR))
    parser.add_argument("--clear", action="store_true", help="Delete existing training_data images/manifest first")
    args = parser.parse_args()

    if args.pilot:
        args.max_train = 400
        args.max_val = 100
        args.max_test = 0

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    images_dir = out_dir / "images"
    manifest_path = out_dir / "dataset_manifest.jsonl"

    if args.clear and out_dir.exists():
        import shutil
        print(f"Clearing {out_dir} ...")
        shutil.rmtree(out_dir)

    images_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    if args.custom_split and not args.pilot:
        train_records, val_records = collect_custom_train_val(
            args.max_train, args.max_val, images_dir,
            include_negative=args.include_negative,
        )
        if len(train_records) < args.max_train or len(val_records) < args.max_val:
            raise RuntimeError(
                f"Requested quotas not reached: train={len(train_records)}/{args.max_train}, "
                f"val={len(val_records)}/{args.max_val}"
            )
        all_records += train_records + val_records
    else:
        all_records += collect_split(
            "train", args.max_train, images_dir,
            pilot=args.pilot, include_negative=args.include_negative,
        )
        # HF uses 'val' or 'validation'
        try:
            all_records += collect_split(
                "val", args.max_val, images_dir,
                pilot=args.pilot, include_negative=args.include_negative,
            )
        except Exception:
            all_records += collect_split(
                "validation", args.max_val, images_dir,
                pilot=args.pilot, include_negative=args.include_negative,
            )

    if args.max_test > 0:
        all_records += collect_split(
            "test", args.max_test, images_dir,
            pilot=args.pilot, include_negative=args.include_negative,
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print("\n" + "=" * 60)
    print("DOCLAYNET BUILD COMPLETE")
    print("=" * 60)
    print(f"Manifest: {manifest_path}")
    print(f"Images:   {images_dir}")
    print(f"Total:    {len(all_records)}")
    print(f"Splits:   {dict(Counter(r['split'] for r in all_records))}")
    print(f"Dataset:  {dict(Counter(r['dataset'] for r in all_records))}")
    print()
    print("Next:")
    print("  .\\venv\\Scripts\\python.exe scripts\\dataset\\validate_dataset.py")
    print("  .\\venv\\Scripts\\python.exe scripts\\training\\overfit_test.py --fast")
    print("  .\\venv\\Scripts\\python.exe scripts\\training\\train_florence.py --config configs\\florence_layout.yaml")
    print("=" * 60)
    return 0 if all_records else 1


if __name__ == "__main__":
    sys.exit(main())
