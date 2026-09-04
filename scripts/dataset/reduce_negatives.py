"""Reduce negative ('No Picture') samples in dataset manifest to a target count.

Preserves 100% of positive ('Picture') records and downsamples negative records
proportionally across splits (train, val, test) while avoiding split leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.bbox_utils import TARGET_CLASS
from src.layout.paths import MANIFEST_PATH, TRAINING_DATA_DIR


def has_picture_annotation(record: Dict[str, Any]) -> bool:
    for ann in record.get("annotations", []):
        if ann.get("category") == TARGET_CLASS:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Reduce No Picture records in dataset")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path to manifest JSONL")
    parser.add_argument("--target-negatives", type=int, default=12000, help="Target count of No Picture records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--backup", default=None, help="Backup file path (default: <manifest>.backup_45k.jsonl)")
    parser.add_argument("--delete-unreferenced-images", action="store_true", help="Delete PNGs that are no longer referenced")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}", file=sys.stderr)
        return 1

    print(f"Loading manifest from {manifest_path} ...")
    with manifest_path.open("r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    print(f"Total records loaded: {len(all_records)}")

    positives: List[Dict[str, Any]] = []
    negatives_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in all_records:
        if has_picture_annotation(rec):
            positives.append(rec)
        else:
            split = rec.get("split", "train")
            negatives_by_split[split].append(rec)

    total_negatives = sum(len(records) for records in negatives_by_split.values())
    print(f"  Positive records (with Picture): {len(positives)}")
    print(f"  Negative records (No Picture):   {total_negatives}")
    for sp, recs in negatives_by_split.items():
        print(f"    - {sp} negatives: {len(recs)}")

    if total_negatives <= args.target_negatives:
        print(f"Negative count ({total_negatives}) is already <= target ({args.target_negatives}). No reduction needed.")
        return 0

    # Calculate proportional targets per split
    splits = list(negatives_by_split.keys())
    split_targets: Dict[str, int] = {}
    allocated = 0

    for i, sp in enumerate(splits):
        if i == len(splits) - 1:
            split_targets[sp] = args.target_negatives - allocated
        else:
            count = round(args.target_negatives * len(negatives_by_split[sp]) / total_negatives)
            split_targets[sp] = count
            allocated += count

    print("\nDownsampling quota per split:")
    for sp, tgt in split_targets.items():
        print(f"  {sp}: {len(negatives_by_split[sp])} -> {tgt}")

    # Deterministic sampling within each split
    rng = random.Random(args.seed)
    sampled_negatives: List[Dict[str, Any]] = []

    for sp, recs in negatives_by_split.items():
        tgt = split_targets[sp]
        if tgt >= len(recs):
            sampled_negatives.extend(recs)
        else:
            # Sample indices and sort to preserve natural order
            indices = sorted(rng.sample(range(len(recs)), tgt))
            sampled_negatives.extend(recs[i] for i in indices)

    print(f"\nTotal sampled negatives: {len(sampled_negatives)}")
    print(f"Total kept positives:    {len(positives)}")
    total_final = len(positives) + len(sampled_negatives)
    print(f"New total records:       {total_final}")

    # Preserve original ordering
    keep_ids = {r.get("id") for r in positives} | {r.get("id") for r in sampled_negatives}
    final_records = [r for r in all_records if r.get("id") in keep_ids]
    assert len(final_records) == total_final, f"Mismatch: {len(final_records)} vs {total_final}"

    final_splits = Counter(r.get("split", "train") for r in final_records)
    print(f"Final records by split: {dict(final_splits)}")

    if args.dry_run:
        print("\n[Dry Run] No files modified.")
        return 0

    # Backup original manifest
    backup_path = Path(args.backup) if args.backup else manifest_path.with_name("dataset_manifest.backup_45k.jsonl")
    if not backup_path.exists():
        print(f"Creating backup at {backup_path} ...")
        shutil.copy2(manifest_path, backup_path)
    else:
        print(f"Backup already exists at {backup_path}")

    # Write new manifest
    print(f"Writing {len(final_records)} records to {manifest_path} ...")
    temp_manifest = manifest_path.with_suffix(".tmp")
    with temp_manifest.open("w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    temp_manifest.replace(manifest_path)
    print("Manifest updated successfully.")

    # Optional deletion of unreferenced images
    if args.delete_unreferenced_images:
        print("\nChecking for unreferenced images...")
        kept_images = {Path(r["image_path"]).name for r in final_records if "image_path" in r}
        images_dir = manifest_path.parent / "images"
        if images_dir.exists():
            deleted = 0
            for img_file in images_dir.glob("*.png"):
                if img_file.name not in kept_images:
                    img_file.unlink()
                    deleted += 1
            print(f"Deleted {deleted} unreferenced image files from {images_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
