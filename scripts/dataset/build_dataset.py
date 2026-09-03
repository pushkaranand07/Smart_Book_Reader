"""Dataset build entrypoint — DocLayNet (replaces AI2D + synthetic)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Florence training dataset from DocLayNet")
    parser.add_argument("--pilot", action="store_true", help="Small 400/100 set for CPU smoke tests")
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-val", type=int, default=800)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--clear", action="store_true", help="Wipe existing data/training_data first")
    parser.add_argument("--no-validate", action="store_true")
    args, unknown = parser.parse_known_args()

    py = sys.executable
    cmd = [py, str(PROJECT_ROOT / "scripts" / "dataset" / "build_doclaynet.py")]
    if args.pilot:
        cmd.append("--pilot")
    else:
        cmd.extend([
            "--max-train", str(args.max_train),
            "--max-val", str(args.max_val),
            "--max-test", str(args.max_test),
        ])
    if args.clear:
        cmd.append("--clear")
    cmd.extend(unknown)

    print("Building DocLayNet Picture dataset...")
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    if not args.no_validate:
        print("Validating dataset...")
        return subprocess.call([py, str(PROJECT_ROOT / "scripts" / "dataset" / "validate_dataset.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
