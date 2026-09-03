"""Tiny-set overfit test — verifies Florence can memorize Picture boxes.

Pass criteria (Phase 23):
  - Spatial localization on memorized pages, NOT a hard 20% loss drop.
  - PASS if >=75% of images reach IoU >= 0.80, OR clear IoU improvement.

Usage:
  # Recommended on CPU (faster):
  .\\venv\\Scripts\\python.exe scripts\\training\\overfit_test.py --fast

  # Full tiny-set (better signal, slower on CPU):
  .\\venv\\Scripts\\python.exe scripts\\training\\overfit_test.py --optimizer-updates 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.bbox_utils import TARGET_CLASS, validate_bbox_xyxy
from src.layout.dataset_schema import load_manifest, normalize_record, select_overfit_records
from src.layout.environment import print_environment
from src.layout.florence_model import attach_lora, load_base_model, load_processor
from src.layout.florence_processor import PictureDetectionDataset, collate_batch, generate_detections
from src.layout.metrics import compute_detection_metrics
from src.layout.paths import MANIFEST_PATH, resolve_image_path


def _evaluate_subset(model, processor, subset, manifest, device):
    """Evaluate all predicted boxes (no label filter) — overfit cares about localization."""
    all_gt, all_pred = [], []
    samples = []
    for rec in subset:
        pics = [a for a in rec["annotations"] if a["category"] == TARGET_CLASS]
        img = Image.open(resolve_image_path(rec["image_path"], manifest.parent)).convert("RGB")
        w, h = img.size
        gt = [validate_bbox_xyxy(a["bbox"], w, h) for a in pics]
        det = generate_detections(model, processor, img, device, num_beams=1, max_new_tokens=64)
        pred = [tuple(b) for b in det["bboxes"]]
        all_gt.append(gt)
        all_pred.append(pred)
        if len(samples) < 3:
            samples.append({
                "file": rec.get("image_filename"),
                "gt": gt,
                "pred": pred,
                "labels": det.get("labels"),
                "raw": (det.get("raw_text") or "")[:100],
            })
    return compute_detection_metrics(all_gt, all_pred), samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Florence overfit test")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--optimizer-updates", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iou-pass-threshold", type=float, default=0.80)
    parser.add_argument("--min-images-pass-pct", type=float, default=75.0)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip pre-training eval (saves time on CPU)")
    parser.add_argument("--fast", action="store_true",
                        help="CPU-friendly: 8 samples, 120 updates, lr=3e-5, skip baseline")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    if args.fast:
        args.n_samples = 8
        args.optimizer_updates = 120
        args.lr = 3e-5
        args.skip_baseline = True
        print("FAST MODE: 8 samples, 120 updates, lr=3e-5, skip baseline", flush=True)

    print_environment()
    print("\nOVERFIT TEST", flush=True)
    print("=" * 50, flush=True)

    manifest = Path(args.manifest)
    records = [normalize_record(r, i) for i, r in enumerate(load_manifest(manifest))]
    subset = select_overfit_records(records, n=args.n_samples)
    print(f"Selected {len(subset)} pages (partial Picture boxes)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device.upper()} | LR: {args.lr} | Updates: {args.optimizer_updates}", flush=True)

    processor = load_processor()
    model, _ = load_base_model(device=device)
    model = attach_lora(model, r=16, lora_alpha=32, lora_dropout=0.0)
    model.print_trainable_parameters()

    loader = DataLoader(
        PictureDetectionDataset(subset, processor, manifest.parent),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        max(1, int(args.optimizer_updates * 0.05)),
        args.optimizer_updates,
    )

    base_mean_iou = 0.0
    if not args.skip_baseline:
        print("Running baseline generation (before training)...", flush=True)
        base_metrics, _ = _evaluate_subset(model, processor, subset, manifest, device)
        base_mean_iou = base_metrics["mean_per_image_iou"]
        print(f"Before training: mean_per_image_iou={base_mean_iou:.3f}", flush=True)

    losses: list[float] = []
    optimizer_update = 0
    model.train()

    print("Training...", flush=True)
    while optimizer_update < args.optimizer_updates:
        for batch in loader:
            if optimizer_update >= args.optimizer_updates:
                break
            optimizer.zero_grad()
            forward_kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "pixel_values": batch["pixel_values"].to(device, dtype=dtype),
                "labels": batch["labels"].to(device),
            }
            if "attention_mask" in batch:
                forward_kwargs["attention_mask"] = batch["attention_mask"].to(device)
            out = model(**forward_kwargs)
            loss_val = out.loss.item()
            if not torch.isfinite(out.loss):
                print(f"FATAL: non-finite loss at update {optimizer_update}", flush=True)
                return 1
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer_update += 1
            losses.append(loss_val)
            if optimizer_update % 20 == 0 or optimizer_update == args.optimizer_updates:
                print(
                    f"  OptUpdate {optimizer_update}/{args.optimizer_updates} | Loss {loss_val:.4f}",
                    flush=True,
                )

    print("Evaluating generation on memorized set...", flush=True)
    metrics, samples = _evaluate_subset(model, processor, subset, manifest, device)

    initial = sum(losses[:5]) / max(1, min(5, len(losses)))
    final = sum(losses[-5:]) / max(1, min(5, len(losses)))
    loss_dropped = final < initial

    print("\nSAMPLE PREDICTIONS (first 3)", flush=True)
    for s in samples:
        print(f"  {s['file']}", flush=True)
        print(f"    GT:   {s['gt']}", flush=True)
        print(f"    Pred: {s['pred']}", flush=True)
        print(f"    Labels: {s['labels']}", flush=True)

    print("\nRESULTS", flush=True)
    print(f"  Initial loss (first 5):     {initial:.4f}", flush=True)
    print(f"  Final loss (last 5):        {final:.4f}", flush=True)
    print(f"  Loss trend:                 {'down' if loss_dropped else 'flat/up'}", flush=True)
    print(f"  Mean per-image IoU:         {metrics['mean_per_image_iou']:.4f}", flush=True)
    print(
        f"  Images with IoU >= {args.iou_pass_threshold}:  {metrics['images_iou80_pct']:.1f}%",
        flush=True,
    )
    print(f"  Recall@50:                  {metrics.get('recall_ap50', 0):.3f}", flush=True)
    print(f"  Precision@50:               {metrics.get('precision_ap50', 0):.3f}", flush=True)
    print(
        f"  Avg pred boxes/image:       {metrics['avg_pred_per_image']:.2f} "
        f"(GT: {metrics['avg_gt_per_image']:.2f})",
        flush=True,
    )

    # Primary: spatial localization (Phase 23) — loss % drop is informational only
    iou_pass = metrics["images_iou80_pct"] >= args.min_images_pass_pct
    iou_improved = (
        (not args.skip_baseline)
        and metrics["mean_per_image_iou"] > base_mean_iou + 0.15
        and metrics["mean_per_image_iou"] >= 0.50
    )
    recall_ok = metrics.get("recall_ap50", 0) >= 0.70
    passed = iou_pass or (iou_improved and recall_ok)

    print(flush=True)
    if not args.skip_baseline:
        print(f"  IoU improved vs baseline:   {'YES' if iou_improved else 'NO'}", flush=True)
    print(
        f"  >= {args.min_images_pass_pct:.0f}% images IoU>={args.iou_pass_threshold}:  "
        f"{'YES' if iou_pass else 'NO'}",
        flush=True,
    )
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        print(
            "  Tip: re-run with --fast, or use Colab T4: "
            "--optimizer-updates 200 --lr 2e-5 --n-samples 16",
            flush=True,
        )
    print("=" * 50, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
