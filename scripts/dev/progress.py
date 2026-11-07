"""Update docs/progress/tasks.json (rendered by progress_board.py).

Examples:
    python scripts/dev/progress.py --done "Write PLAN.md" --log "PLAN.md committed" --current "Phase 1: chunking"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

STATE = Path(__file__).resolve().parents[2] / "docs" / "progress" / "tasks.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--done", action="append", default=[], help="substring of a task name to mark done")
    parser.add_argument("--log", action="append", default=[], help="activity log entry")
    parser.add_argument("--current", help="what is being worked on now")
    args = parser.parse_args()

    state = json.loads(STATE.read_text(encoding="utf-8"))
    for needle in args.done:
        matches = [t for p in state["phases"] for t in p["tasks"] if needle.lower() in t["name"].lower()]
        if not matches:
            raise SystemExit(f"no task matches {needle!r}")
        for task in matches:
            task["done"] = True
    stamp = datetime.now().strftime("%H:%M")
    state.setdefault("log", []).extend(f"{stamp} {entry}" for entry in args.log)
    if args.current:
        state["current"] = args.current

    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(STATE)


if __name__ == "__main__":
    main()
