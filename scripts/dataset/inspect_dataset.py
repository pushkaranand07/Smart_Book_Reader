"""Dataset inspection — detailed statistics without blocking."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.bbox_utils import box_area_ratio, TARGET_CLASS
from src.layout.dataset_schema import classify_record, get_source_id, load_manifest, normalize_record
from src.layout.paths import MANIFEST_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect dataset statistics")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    records = [normalize_record(r, i) for i, r in enumerate(load_manifest(args.manifest))]

    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    areas = []
    for r in records:
        for a in r.get("annotations", []):
            if a.get("category") == TARGET_CLASS:
                try:
                    areas.append(box_area_ratio(a["bbox"], r["width"], r["height"]))
                except Exception:
                    pass

    print("DATASET INSPECTION")
    print("=" * 50)
    print(f"Total records: {len(records)}")
    print(f"Datasets: {dict(Counter(r.get('dataset') for r in records))}")
    print(f"Splits: {dict(Counter(r.get('split') for r in records))}")
    print(f"Classification: {dict(Counter(classify_record(r) for r in records))}")
    print(f"Unique sources: {len(set(get_source_id(r) for r in records))}")
    print(f"Image width  min/med/max: {min(widths)}/{sorted(widths)[len(widths)//2]}/{max(widths)}")
    print(f"Image height min/med/max: {min(heights)}/{sorted(heights)[len(heights)//2]}/{max(heights)}")
    if areas:
        areas.sort()
        print(f"Picture area ratio min/med/max: {areas[0]:.3f}/{areas[len(areas)//2]:.3f}/{areas[-1]:.3f}")
        full_page = sum(1 for a in areas if a > 0.95)
        partial = sum(1 for a in areas if 0.05 < a <= 0.95)
        print(f"Full-page Picture boxes (>95%): {full_page}")
        print(f"Partial Picture boxes (5-95%): {partial}")
    print()
    ds_names = set(r.get("dataset") for r in records)
    if "DocLayNet" in ds_names:
        print("NOTE: DocLayNet = real human-annotated document pages (Picture-only subset).")
        print("  Categories kept: Picture only (Caption/Table/Text ignored for this experiment).")
    else:
        print("NOTE: Expected dataset is DocLayNet. Rebuild with:")
        print("  python scripts/dataset/build_doclaynet.py --pilot")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
