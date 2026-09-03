"""Dataset validation CLI — must PASS before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.dataset_schema import ValidationReport, validate_dataset
from src.layout.paths import MANIFEST_PATH


def print_report(report: ValidationReport) -> None:
    print("DATASET VALIDATION")
    print("=" * 50)
    print(f"Records: {report.total_records}")
    print()
    print("By dataset:")
    for k, v in sorted(report.datasets.items()):
        print(f"  {k}: {v}")
    print()
    print("By split:")
    for k, v in sorted(report.splits.items()):
        print(f"  {k}: {v}")
    print()
    print("Classification:")
    for k, v in sorted(report.classification.items()):
        pct = 100 * v / max(1, report.total_records)
        print(f"  {k}: {v} ({pct:.1f}%)")
    print()
    print(f"Missing images: {report.missing_images}")
    print(f"Corrupt images: {report.corrupt_images}")
    print(f"Invalid boxes:  {report.invalid_boxes}")
    print(f"No Picture:     {report.no_picture}")
    print()
    print("Annotations:")
    for k, v in sorted(report.annotation_counts.items()):
        print(f"  {k}: {v}")
    print()
    print("Source counts:")
    print(f"  TRAIN sources: {report.train_sources}")
    print(f"  VAL sources:   {report.val_sources}")
    print(f"  TEST sources:  {report.test_sources}")
    print()
    print("Source overlap:")
    print(f"  train/val:  {report.source_overlap.get('train_val', 0)}")
    print(f"  train/test: {report.source_overlap.get('train_test', 0)}")
    print(f"  val/test:   {report.source_overlap.get('val_test', 0)}")
    if report.picture_area_stats:
        print()
        print("Picture area ratio:")
        for k, v in report.picture_area_stats.items():
            print(f"  {k}: {v:.4f}")
    if report.warnings:
        print()
        print("Warnings:")
        for w in report.warnings[:10]:
            print(f"  - {w}")
    if report.errors:
        print()
        print("Errors:")
        for e in report.errors[:10]:
            print(f"  - {e}")
    print()
    print(f"STATUS: {report.status}")
    print("=" * 50)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate training dataset")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--quick", action="store_true", help="Check first 500 images only")
    args = parser.parse_args()

    max_checks = 500 if args.quick else None
    report = validate_dataset(args.manifest, check_images=True, max_image_checks=max_checks)
    print_report(report)
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
