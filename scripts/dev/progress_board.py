"""Render docs/PA-OH-1.md from docs/progress/tasks.json every N seconds.

Usage:
    python scripts/dev/progress_board.py            # loop, refresh every 10 s
    python scripts/dev/progress_board.py --once     # render once and exit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "docs" / "progress" / "tasks.json"
OUTPUT = ROOT / "docs" / "PA-OH-1.md"
BAR_WIDTH = 30


def bar(done: int, total: int) -> str:
    ratio = done / total if total else 0.0
    filled = round(ratio * BAR_WIDTH)
    return f"`{'█' * filled}{'░' * (BAR_WIDTH - filled)}` {ratio * 100:5.1f}% ({done}/{total})"


def recent_commits(limit: int = 15) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "log", f"-{limit}", "--pretty=format:%h  %ad  %s", "--date=format:%H:%M"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ["(git log unavailable)"]
    return [line for line in out.splitlines() if line.strip()]


def render() -> str:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    phases = state["phases"]
    all_tasks = [t for p in phases for t in p["tasks"]]
    total_done = sum(t["done"] for t in all_tasks)

    lines = [
        f"# {state['title']}",
        "",
        f"**Last refreshed:** {datetime.now():%Y-%m-%d %H:%M:%S} (auto-refreshes every 10 s)  ",
        f"**Branch:** `{state['branch']}`  ",
        f"**Working on now:** {state['current']}",
        "",
        "## Overall",
        "",
        bar(total_done, len(all_tasks)),
        "",
        "## By phase",
        "",
        "| Phase | Progress |",
        "| --- | --- |",
    ]
    for phase in phases:
        done = sum(t["done"] for t in phase["tasks"])
        lines.append(f"| {phase['name']} | {bar(done, len(phase['tasks']))} |")

    lines += ["", "## Task checklist", ""]
    for phase in phases:
        lines.append(f"### {phase['name']}")
        lines.append("")
        for task in phase["tasks"]:
            lines.append(f"- [{'x' if task['done'] else ' '}] {task['name']}")
        lines.append("")

    lines += ["## Activity log (newest first)", ""]
    lines += [f"- {entry}" for entry in reversed(state.get("log", []))]
    lines += ["", "## Recent commits", "", "```text", *recent_commits(), "```", ""]
    return "\n".join(lines)


def write_once() -> None:
    try:
        content = render()
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        # The state file may be mid-write; keep the previous board and retry next tick.
        print(f"progress_board: skipped refresh ({exc})")
        return
    OUTPUT.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    write_once()
    if args.once:
        return
    while True:
        time.sleep(args.interval)
        write_once()


if __name__ == "__main__":
    main()
