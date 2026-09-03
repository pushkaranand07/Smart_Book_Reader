"""Training logging, history, and checkpoint management."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.layout.environment import environment_text


class TrainingLogger:
    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.ema_loss: float | None = None
        self.ema_alpha = 0.1

        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (self.run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")

    def log_step(
        self,
        step: int,
        total_steps: int,
        optimizer_update: int,
        loss: float,
        lr: float,
    ) -> None:
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss

        entry = {
            "step": step,
            "total_steps": total_steps,
            "optimizer_update": optimizer_update,
            "loss": round(loss, 6),
            "ema_loss": round(self.ema_loss, 6),
            "lr": lr,
        }
        self.history.append(entry)

        if step % 10 == 0 or step == total_steps:
            print(
                f"Step {step}/{total_steps} | "
                f"OptUpdate {optimizer_update} | "
                f"Loss {loss:.4f} | "
                f"EMA {self.ema_loss:.4f} | "
                f"LR {lr:.2e}"
            )

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float, elapsed_s: float) -> None:
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
            "elapsed_s": round(elapsed_s, 1),
        }
        self.history.append(entry)
        print(f"\nEpoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {lr:.2e} | {elapsed_s:.0f}s")

    def save(self) -> None:
        hist_path = self.run_dir / "training_history.json"
        hist_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

        csv_path = self.run_dir / "training_history.csv"
        if self.history:
            keys = sorted({k for row in self.history for k in row})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)


def create_run_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
