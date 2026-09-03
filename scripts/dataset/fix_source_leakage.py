"""Fix source-level train/val leakage by reassigning conflicting sources to train."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.dataset_schema import get_source_id, load_manifest, normalize_record
from src.layout.paths import MANIFEST_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix source-level split leakage")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    records = [normalize_record(r, i) for i, r in enumerate(load_manifest(manifest))]

    source_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        source_splits[r["source_id"]].add(r.get("split", "train"))

    leaky_sources = {s for s, splits in source_splits.items() if len(splits) > 1}
    fixed = 0
    for r in records:
        if r["source_id"] in leaky_sources and r.get("split") == "val":
            r["split"] = "train"
            fixed += 1

    print(f"Leaky sources: {len(leaky_sources)}")
    print(f"Records reassigned val->train: {fixed}")

    if not args.dry_run and fixed > 0:
        with manifest.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Updated: {manifest}")
    elif args.dry_run:
        print("Dry run — no changes written")

    return 0


if __name__ == "__main__":
    sys.exit(main())
