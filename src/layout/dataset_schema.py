"""Dataset manifest loading, normalization, and validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from src.layout.bbox_utils import BBoxError, TARGET_CLASS, box_area_ratio, validate_bbox_xyxy
from src.layout.paths import MANIFEST_PATH, resolve_image_path


@dataclass
class ValidationReport:
    status: str = "FAIL"
    total_records: int = 0
    datasets: Dict[str, int] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=dict)
    classification: Dict[str, int] = field(default_factory=dict)
    missing_images: int = 0
    corrupt_images: int = 0
    invalid_boxes: int = 0
    no_picture: int = 0
    annotation_counts: Dict[str, int] = field(default_factory=dict)
    source_overlap: Dict[str, int] = field(default_factory=dict)
    train_sources: int = 0
    val_sources: int = 0
    test_sources: int = 0
    picture_area_stats: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def load_manifest(path: Path | str = MANIFEST_PATH) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def get_source_id(record: Dict[str, Any]) -> str:
    if record.get("source_id"):
        return str(record["source_id"])
    if record.get("source_diagram"):
        return str(record["source_diagram"])
    if record.get("dataset") == "AI2D":
        return Path(record.get("image_filename", "")).stem
    return Path(record.get("image_filename", record.get("image_path", "unknown"))).stem


def classify_record(record: Dict[str, Any]) -> str:
    ds = record.get("dataset", "")
    if ds == "DocLayNet":
        return "REAL"
    if ds == "AI2D":
        return "REAL"
    if ds == "SyntheticLayout":
        return "SOURCE_DERIVED_SYNTHETIC"
    return "OTHER"


def normalize_record(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    rec = dict(record)
    if not rec.get("id"):
        rec["id"] = f"{rec.get('dataset','unknown')}_{idx:06d}"
    rec["source_id"] = get_source_id(rec)
    if not rec.get("split"):
        rec["split"] = "train"
    return rec


def validate_dataset(
    manifest_path: Path | str = MANIFEST_PATH,
    check_images: bool = True,
    max_image_checks: int | None = None,
) -> ValidationReport:
    manifest_path = Path(manifest_path)
    report = ValidationReport()
    records = [normalize_record(r, i) for i, r in enumerate(load_manifest(manifest_path))]
    report.total_records = len(records)

    for r in records:
        report.datasets[r.get("dataset", "unknown")] = report.datasets.get(r.get("dataset", "unknown"), 0) + 1
        report.splits[r.get("split", "missing")] = report.splits.get(r.get("split", "missing"), 0) + 1
        cls = classify_record(r)
        report.classification[cls] = report.classification.get(cls, 0) + 1

    path_counts: Counter = Counter()
    area_ratios: List[float] = []

    for i, rec in enumerate(records):
        path_counts[rec.get("image_path", "")] += 1

        has_picture = False
        for ann in rec.get("annotations", []):
            cat = ann.get("category", "unknown")
            report.annotation_counts[cat] = report.annotation_counts.get(cat, 0) + 1
            if cat == TARGET_CLASS:
                has_picture = True
                try:
                    validate_bbox_xyxy(ann["bbox"], rec["width"], rec["height"])
                    area_ratios.append(box_area_ratio(ann["bbox"], rec["width"], rec["height"]))
                except BBoxError as e:
                    report.invalid_boxes += 1
                    if report.invalid_boxes <= 5:
                        report.errors.append(f"Invalid bbox in {rec.get('id')}: {e}")

        if not has_picture:
            report.no_picture += 1

        if check_images and (max_image_checks is None or i < max_image_checks):
            try:
                img_path = resolve_image_path(rec["image_path"], manifest_path.parent)
                with Image.open(img_path) as img:
                    img.verify()
            except FileNotFoundError:
                report.missing_images += 1
                if report.missing_images <= 5:
                    report.errors.append(f"Missing: {rec.get('image_path')}")
            except Exception as e:
                report.corrupt_images += 1
                if report.corrupt_images <= 5:
                    report.errors.append(f"Corrupt {rec.get('image_path')}: {e}")

    dup_paths = sum(1 for c in path_counts.values() if c > 1)
    if dup_paths:
        report.warnings.append(f"{dup_paths} duplicate image_path entries")

    # Source-level split leakage
    split_sources: Dict[str, Set[str]] = defaultdict(set)
    for rec in records:
        split_sources[rec.get("split", "missing")].add(rec["source_id"])

    report.train_sources = len(split_sources.get("train", set()))
    report.val_sources = len(split_sources.get("val", set()))
    report.test_sources = len(split_sources.get("test", set()))

    train_s = split_sources.get("train", set())
    val_s = split_sources.get("val", set())
    test_s = split_sources.get("test", set())

    report.source_overlap = {
        "train_val": len(train_s & val_s),
        "train_test": len(train_s & test_s),
        "val_test": len(val_s & test_s),
    }

    if area_ratios:
        area_ratios.sort()
        report.picture_area_stats = {
            "min": area_ratios[0],
            "max": area_ratios[-1],
            "median": area_ratios[len(area_ratios) // 2],
        }

    blocking = (
        report.missing_images > 0
        or report.corrupt_images > 0
        or report.source_overlap["train_val"] > 0
        or report.source_overlap["train_test"] > 0
        or report.source_overlap["val_test"] > 0
        or report.splits.get("missing", 0) > 0
    )

    if report.source_overlap["train_val"] > 0:
        report.errors.append(
            f"Source leakage: {report.source_overlap['train_val']} sources in both train and val. "
            f"Run: python scripts/dataset/fix_source_leakage.py"
        )

    report.status = "FAIL" if blocking else "PASS"
    return report


def filter_trainable_records(
    records: List[Dict[str, Any]], allow_empty: bool = False
) -> Tuple[List[Dict], int]:
    """Keep valid records, optionally retaining negative pages without Pictures."""
    kept: List[Dict] = []
    skipped = 0
    for rec in records:
        pics = [a for a in rec.get("annotations", []) if a.get("category") == TARGET_CLASS]
        if not pics:
            if allow_empty:
                kept.append(rec)
            else:
                skipped += 1
            continue
        try:
            for ann in pics:
                validate_bbox_xyxy(ann["bbox"], rec["width"], rec["height"])
            kept.append(rec)
        except BBoxError:
            skipped += 1
    return kept, skipped


def select_overfit_records(records: List[Dict[str, Any]], n: int = 16) -> List[Dict[str, Any]]:
    """Select pages where Picture is NOT the full page (DocLayNet preferred)."""
    preferred_datasets = {"DocLayNet", "SyntheticLayout"}
    candidates = []
    for rec in records:
        if rec.get("dataset") not in preferred_datasets and preferred_datasets:
            # Still allow any dataset with partial Picture
            pass
        pics = [a for a in rec.get("annotations", []) if a.get("category") == TARGET_CLASS]
        if not pics:
            continue
        try:
            ratio = box_area_ratio(pics[0]["bbox"], rec["width"], rec["height"])
        except BBoxError:
            continue
        if 0.02 < ratio < 0.80:
            # Prefer moderate-sized boxes that make memorization measurable.
            weight = 0 if rec.get("dataset") == "DocLayNet" else 1
            candidates.append((weight, abs(ratio - 0.25), ratio, rec))

    candidates.sort(key=lambda x: (x[0], x[1]))
    if len(candidates) < n:
        # Fallback: any Picture page
        fallback = []
        for rec in records:
            pics = [a for a in rec.get("annotations", []) if a.get("category") == TARGET_CLASS]
            if not pics:
                continue
            try:
                ratio = box_area_ratio(pics[0]["bbox"], rec["width"], rec["height"])
            except BBoxError:
                continue
            fallback.append((0, abs(ratio - 0.25), ratio, rec))
        candidates = fallback

    return [r for _, _, _, r in candidates[:n]]
