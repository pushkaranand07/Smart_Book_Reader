"""Per-image and aggregate detection metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from src.layout.bbox_utils import compute_iou


def match_boxes(
    gt_boxes: List[Sequence[float]],
    pred_boxes: List[Sequence[float]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, List[float]]:
    """Greedy one-to-one matching. Returns (tp, fp, fn, matched_ious)."""
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, []
    if not gt_boxes:
        return 0, len(pred_boxes), 0, []
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []

    pairs: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = compute_iou(gt, pred)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))

    pairs.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_ious: List[float] = []

    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_ious


def per_image_best_iou(
    gt_boxes: List[Sequence[float]],
    pred_boxes: List[Sequence[float]],
) -> float:
    """Best IoU per ground-truth box (0.0 if no predictions)."""
    if not gt_boxes:
        return 1.0
    if not pred_boxes:
        return 0.0
    best_scores = []
    for gt in gt_boxes:
        best = max((compute_iou(gt, p) for p in pred_boxes), default=0.0)
        best_scores.append(best)
    return sum(best_scores) / len(best_scores)


def compute_detection_metrics(
    all_gt: List[List[Sequence[float]]],
    all_pred: List[List[Sequence[float]]],
    iou_thresholds: List[float] | None = None,
) -> Dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    results: Dict[str, float] = {}
    total_gt = sum(len(g) for g in all_gt)
    total_pred = sum(len(p) for p in all_pred)

    for thr in iou_thresholds:
        tp_total = fp_total = fn_total = 0
        for gt_boxes, pred_boxes in zip(all_gt, all_pred):
            tp, fp, fn, _ = match_boxes(gt_boxes, pred_boxes, thr)
            tp_total += tp
            fp_total += fp
            fn_total += fn

        precision = tp_total / max(1, tp_total + fp_total)
        recall = tp_total / max(1, tp_total + fn_total)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        key = f"ap{int(thr * 100)}"
        results[f"precision_{key}"] = precision
        results[f"recall_{key}"] = recall
        results[f"f1_{key}"] = f1

    # Mean IoU at 0.5 threshold only (avoid double-counting)
    matched_ious_50: List[float] = []
    per_image_ious: List[float] = []
    images_above_80 = 0

    for gt_boxes, pred_boxes in zip(all_gt, all_pred):
        _, _, _, ious = match_boxes(gt_boxes, pred_boxes, 0.5)
        matched_ious_50.extend(ious)
        img_iou = per_image_best_iou(gt_boxes, pred_boxes)
        per_image_ious.append(img_iou)
        if img_iou >= 0.80:
            images_above_80 += 1

    results["mean_iou"] = sum(matched_ious_50) / max(1, len(matched_ious_50))
    results["mean_per_image_iou"] = sum(per_image_ious) / max(1, len(per_image_ious))
    results["images_iou80_pct"] = 100.0 * images_above_80 / max(1, len(all_gt))
    results["total_gt_boxes"] = total_gt
    results["total_pred_boxes"] = total_pred
    results["avg_gt_per_image"] = total_gt / max(1, len(all_gt))
    results["avg_pred_per_image"] = total_pred / max(1, len(all_pred))
    return results
