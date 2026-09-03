"""Canonical Florence-2 Picture detection training script."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.bbox_utils import TARGET_CLASS, validate_bbox_xyxy
from src.layout.dataset_schema import (
    filter_trainable_records,
    load_manifest,
    normalize_record,
    validate_dataset,
)
from src.layout.environment import print_environment
from src.layout.florence_model import (
    attach_lora,
    count_parameters,
    load_base_model,
    load_processor,
    print_preflight,
    run_preflight,
)
from src.layout.florence_processor import PictureDetectionDataset, collate_batch, generate_detections
from src.layout.metrics import compute_detection_metrics
from src.layout.paths import FLORENCE_LAYOUT_DIR, MANIFEST_PATH, PROJECT_ROOT as ROOT, RUNS_DIR, resolve_image_path
from src.layout.training_utils import TrainingLogger, create_run_dir
from PIL import Image


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_records(records: List[Dict]) -> tuple[List[Dict], List[Dict], List[Dict]]:
    train, val, test = [], [], []
    for r in records:
        s = r.get("split", "train")
        if s == "val":
            val.append(r)
        elif s == "test":
            test.append(r)
        else:
            train.append(r)
    return train, val, test


def evaluate_generation(
    model, processor, records, manifest_dir, device, max_samples=50
) -> Dict[str, float]:
    all_gt, all_pred = [], []
    for rec in records[:max_samples]:
        pics = [a for a in rec.get("annotations", []) if a.get("category") == TARGET_CLASS]
        if not pics:
            continue
        gt = [validate_bbox_xyxy(a["bbox"], rec["width"], rec["height"]) for a in pics]
        img_path = resolve_image_path(rec["image_path"], manifest_dir)
        img = Image.open(img_path).convert("RGB")
        det = generate_detections(model, processor, img, device)
        pred = [tuple(b) for b in det["bboxes"]]
        all_gt.append(gt)
        all_pred.append(pred)
    return compute_detection_metrics(all_gt, all_pred)


def train_loop(cfg: Dict[str, Any], records: List[Dict], run_dir: Path) -> int:
    manifest_path = ROOT / cfg["data"]["manifest"]
    manifest_dir = manifest_path.parent

    train_recs, val_recs, _ = split_records(records)
    allow_empty = bool(cfg["data"].get("include_negative_pages", False))
    train_recs, skipped = filter_trainable_records(train_recs, allow_empty=allow_empty)
    val_recs, _ = filter_trainable_records(val_recs, allow_empty=allow_empty)
    print(f"Trainable: {len(train_recs)} train, {len(val_recs)} val ({skipped} skipped — no Picture)")

    tcfg = cfg["training"]
    lcfg = cfg["lora"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" and tcfg.get("mixed_precision") == "fp16" else torch.float32

    processor = load_processor(cfg["model"]["base"])
    model, _ = load_base_model(cfg["model"]["base"], device)

    if lcfg.get("enabled", True):
        model = attach_lora(model, r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"])

    params = count_parameters(model)
    print(f"Trainable: {params['trainable']:,} | Frozen: {params['frozen']:,} ({params['trainable_pct']}%)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0.01))

    batch_size = tcfg["batch_size"]
    grad_accum = tcfg["gradient_accumulation"]
    print(f"Physical batch: {batch_size} | Grad accum: {grad_accum} | Effective batch: {batch_size * grad_accum}")

    train_loader = DataLoader(
        PictureDetectionDataset(train_recs, processor, manifest_dir),
        batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0,
    )
    val_loader = DataLoader(
        PictureDetectionDataset(val_recs, processor, manifest_dir),
        batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * tcfg.get("warmup_ratio", 0.05))),
        num_training_steps=total_steps,
    )

    out_base = ROOT / cfg["model"]["output_dir"]
    best_dir = out_base / "best"
    latest_dir = out_base / "latest"
    logger = TrainingLogger(run_dir, cfg)

    best_val = float("inf")
    optimizer_update = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        step_in_epoch = 0

        for step, batch in enumerate(train_loader, 1):
            global_step = (epoch - 1) * steps_per_epoch + step
            forward_kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "pixel_values": batch["pixel_values"].to(device, dtype=dtype),
                "labels": batch["labels"].to(device),
            }
            if "attention_mask" in batch:
                forward_kwargs["attention_mask"] = batch["attention_mask"].to(device)

            outputs = model(**forward_kwargs)
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()
            step_in_epoch = step

            if step % grad_accum == 0 or step == steps_per_epoch:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("max_grad_norm", 1.0))
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"FATAL: grad_norm={grad_norm}")
                    return 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_update += 1
                lr = scheduler.get_last_lr()[0]
                logger.log_step(global_step, total_steps, optimizer_update, outputs.loss.item(), lr)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                val_kwargs = {
                    "input_ids": batch["input_ids"].to(device),
                    "pixel_values": batch["pixel_values"].to(device, dtype=dtype),
                    "labels": batch["labels"].to(device),
                }
                if "attention_mask" in batch:
                    val_kwargs["attention_mask"] = batch["attention_mask"].to(device)
                outs = model(**val_kwargs)
                val_loss += outs.loss.item()
                val_steps += 1

        avg_train = epoch_loss / max(1, step_in_epoch)
        avg_val = val_loss / max(1, val_steps)
        logger.log_epoch(epoch, avg_train, avg_val, scheduler.get_last_lr()[0], time.time() - epoch_start)

        # Save checkpoints
        epoch_dir = out_base / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        processor.save_pretrained(str(epoch_dir))

        latest_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(latest_dir))
        processor.save_pretrained(str(latest_dir))

        if avg_val < best_val:
            best_val = avg_val
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            print(f"  --> New best val loss: {best_val:.4f}")

        # Generation-based eval
        metrics = evaluate_generation(
            model, processor, val_recs, manifest_dir, device,
            max_samples=cfg["evaluation"].get("max_val_samples", 50),
        )
        print(f"  Gen metrics: mean_iou={metrics.get('mean_iou', 0):.4f} "
              f"recall@50={metrics.get('recall_ap50', 0):.3f}")
        (run_dir / f"metrics_epoch_{epoch:03d}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    logger.save()
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Best model: {best_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Florence-2 Picture detection")
    parser.add_argument("--config", default="configs/florence_layout.yaml")
    parser.add_argument("--preflight", action="store_true", help="Run model preflight only")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    print_environment()

    cfg_path = ROOT / args.config
    cfg = load_config(str(cfg_path)) if cfg_path.exists() else {}

    if args.preflight:
        result = run_preflight()
        print_preflight(result)
        return 0 if result["status"] == "PASS" else 1

    manifest = ROOT / cfg.get("data", {}).get("manifest", str(MANIFEST_PATH))

    if not args.skip_validation and cfg.get("data", {}).get("require_validation_pass", True):
        report = validate_dataset(manifest, check_images=True, max_image_checks=200)
        if report.status != "PASS":
            print(f"\nTRAINING REFUSED: dataset validation {report.status}")
            print("Run: python scripts/dataset/validate_dataset.py")
            return 1
        print("Dataset validation: PASS")

    preflight = run_preflight()
    print_preflight(preflight)
    if preflight["status"] != "PASS":
        print("TRAINING REFUSED: model preflight failed")
        return 1

    records = [normalize_record(r, i) for i, r in enumerate(load_manifest(manifest))]
    global run_dir
    run_dir = create_run_dir(RUNS_DIR)
    print(f"Run directory: {run_dir}")
    return train_loop(cfg, records, run_dir)


if __name__ == "__main__":
    sys.exit(main())
