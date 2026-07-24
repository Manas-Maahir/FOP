"""Detection metrics for the YOLO-style trainer: P / R / F1 curves + COCO mAP.

Division of labour, on purpose:

* **mAP50 and mAP50-95 come from pycocotools** (``evaluate.score_coco``). Those are the numbers the
  paper reports and the numbers in [report.md](report.md), so they must keep being produced by the
  same reference implementation. We do not reimplement them.
* **P / R / F1 come from the sweep in this module.** COCO's ``stats`` has no operating-point
  precision/recall -- it integrates them away. To print an Ultralytics-style row you need P and R at
  a *chosen confidence*, so we do the greedy IoU-0.5 match ourselves.

That split means the two families can disagree slightly (COCO interpolates over 101 recall points and
handles ties differently). That is expected and harmless: mAP is the reported metric, P/R/F1 are for
reading the training curve.

Everything here is numpy-only, so it is unit-testable on any machine with no torch and no GPU.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Confidence grid for the F1/P/R-vs-confidence curves.
CONF_GRID = np.linspace(0.0, 1.0, 1001)


# ------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------
def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO ``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``a`` [N,4] and ``b`` [M,4], both xyxy. Returns [N, M]."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


# ------------------------------------------------------------------------------------------
# matching
# ------------------------------------------------------------------------------------------
def match_image(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray,
                iou_thr: float = 0.5) -> np.ndarray:
    """Greedily match one image's predictions to its ground truth.

    Predictions are considered in descending score order; each takes the highest-IoU ground-truth
    box that is still free and clears ``iou_thr``. This is the standard COCO/VOC assignment: one
    prediction per GT, extras are false positives.

    Returns a 0/1 array aligned with ``pred_scores`` **as given** (not re-sorted).
    """
    n = len(pred_scores)
    tp = np.zeros(n, dtype=np.float64)
    if n == 0 or len(gt_boxes) == 0:
        return tp

    ious = box_iou(pred_boxes, gt_boxes)          # [n_pred, n_gt]
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-np.asarray(pred_scores)):
        row = ious[i].copy()
        row[taken] = -1.0
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            tp[i] = 1.0
            taken[j] = True
    return tp


def gather_tp(results: Sequence[dict], gt_by_image: dict[int, np.ndarray],
              iou_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray, int]:
    """Run :func:`match_image` over a whole COCO-format detection list.

    Args:
        results: COCO detections -- ``{"image_id", "bbox" (xywh), "score", ...}``.
        gt_by_image: image_id -> GT boxes as xyxy [N,4]. **Images with zero GT must still be
            present** (as an empty array): they are the non-TB negatives, and every detection on
            them is a false positive. Dropping them is exactly how an all-images evaluation gets
            silently inflated.
        iou_thr: IoU threshold for a match.

    Returns:
        ``(tp, conf, n_gt)`` -- flat arrays over every prediction in every image, plus the total
        ground-truth count.
    """
    tps, confs = [], []
    by_image: dict[int, list[dict]] = {}
    for r in results:
        by_image.setdefault(int(r["image_id"]), []).append(r)

    for img_id, preds in by_image.items():
        boxes = xywh_to_xyxy([p["bbox"] for p in preds])
        scores = np.array([float(p["score"]) for p in preds], dtype=np.float64)
        gts = gt_by_image.get(img_id, np.zeros((0, 4)))
        tps.append(match_image(boxes, scores, gts, iou_thr))
        confs.append(scores)

    n_gt = int(sum(len(v) for v in gt_by_image.values()))
    if not tps:
        return np.zeros(0), np.zeros(0), n_gt
    return np.concatenate(tps), np.concatenate(confs), n_gt


# ------------------------------------------------------------------------------------------
# curves
# ------------------------------------------------------------------------------------------
def pr_curve(tp: np.ndarray, conf: np.ndarray, n_gt: int):
    """Precision/recall as the confidence threshold sweeps down.

    Returns ``(precision, recall, conf_sorted)``, all aligned and ordered by *descending*
    confidence -- i.e. index i is the operating point "keep the top i+1 predictions".
    """
    if len(tp) == 0 or n_gt == 0:
        z = np.zeros(1)
        return z, z, z

    order = np.argsort(-conf)
    tp, conf = tp[order], conf[order]
    tpc = np.cumsum(tp)
    fpc = np.cumsum(1.0 - tp)
    recall = tpc / max(n_gt, 1)
    precision = tpc / np.maximum(tpc + fpc, np.finfo(np.float64).eps)
    return precision, recall, conf


def f1_from_pr(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(np.float64).eps)
    return np.nan_to_num(f1)


def interp_by_conf(values: np.ndarray, conf_sorted: np.ndarray,
                   grid: np.ndarray = CONF_GRID) -> np.ndarray:
    """Resample a descending-confidence curve onto a fixed ascending confidence grid (for plots)."""
    if len(conf_sorted) == 0:
        return np.zeros_like(grid)
    # conf_sorted is descending -> flip so np.interp sees an increasing x
    return np.interp(grid, conf_sorted[::-1], values[::-1], left=0.0, right=0.0)


def best_operating_point(tp: np.ndarray, conf: np.ndarray, n_gt: int):
    """The maximum-F1 point of the PR curve.

    Ultralytics reports P and R at max-F1 rather than at a fixed 0.25/0.5 confidence, because a
    fixed threshold is meaningless across models whose score distributions differ. We follow that.

    Returns ``dict(precision, recall, f1, conf)``.
    """
    precision, recall, conf_sorted = pr_curve(tp, conf, n_gt)
    f1 = f1_from_pr(precision, recall)
    if len(f1) == 0 or not np.any(f1 > 0):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "conf": 0.0}
    i = int(np.argmax(f1))
    return {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "conf": float(conf_sorted[i]),
    }


def confusion_counts(tp: np.ndarray, conf: np.ndarray, n_gt: int, conf_thr: float):
    """(TP, FP, FN) at a fixed confidence -- the single-class confusion matrix.

    With one category-agnostic class there is no TN cell: background is not an enumerable class,
    so a 2x2 "TB vs background" matrix only has three meaningful entries.
    """
    keep = conf >= conf_thr
    tp_n = int(tp[keep].sum())
    fp_n = int((1.0 - tp[keep]).sum())
    fn_n = int(max(n_gt - tp_n, 0))
    return tp_n, fp_n, fn_n


# ------------------------------------------------------------------------------------------
# fitness (drives best.pt selection)
# ------------------------------------------------------------------------------------------
def fitness(ap: float, ap50: float, mode: str = "ultralytics") -> float:
    """Scalar used to decide whether an epoch produced a new ``best.pt``.

    ``ultralytics``: ``0.1*mAP50 + 0.9*mAP50-95`` -- weights the strict metric, so best.pt is not
    chosen by a model that only localises loosely.
    ``ap50`` / ``ap``: single-metric selection. ``ap50`` reproduces what the Colab PoC did, which
    matters if you want to compare against [report.md](report.md) on equal terms.

    Note this only picks *which checkpoint to keep*. The reported headline stays the final-epoch
    number, because best.pt is selected on val and is therefore optimistically biased.
    """
    if mode == "ap50":
        return float(ap50)
    if mode == "ap":
        return float(ap)
    return 0.1 * float(ap50) + 0.9 * float(ap)


# ------------------------------------------------------------------------------------------
# top-level
# ------------------------------------------------------------------------------------------
def evaluate_detections(results: Sequence[dict], ann_file: str,
                        iou_thr: float = 0.5, quiet: bool = True) -> dict:
    """Full metric bundle for one validation pass.

    mAP comes from pycocotools; P/R/F1 and the curves come from this module. Returns a flat dict
    plus the raw curve arrays under ``curves`` for the plotting module.
    """
    from pycocotools.coco import COCO

    from .evaluate import score_coco

    ap, ap50 = score_coco(list(results), ann_file, quiet=quiet)

    # GT keyed by image, INCLUDING images with no boxes -- see gather_tp's docstring.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(ann_file)
    gt_by_image: dict[int, np.ndarray] = {}
    for img_id in coco.imgs:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        gt_by_image[int(img_id)] = xywh_to_xyxy([a["bbox"] for a in anns]) if anns \
            else np.zeros((0, 4))

    tp, conf, n_gt = gather_tp(results, gt_by_image, iou_thr=iou_thr)
    point = best_operating_point(tp, conf, n_gt)
    precision, recall, conf_sorted = pr_curve(tp, conf, n_gt)
    f1 = f1_from_pr(precision, recall)
    tp_n, fp_n, fn_n = confusion_counts(tp, conf, n_gt, point["conf"])

    return {
        "AP": ap,
        "AP50": ap50,
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
        "conf": point["conf"],
        "n_images": len(gt_by_image),
        "n_instances": n_gt,
        "n_dets": len(results),
        "tp": tp_n,
        "fp": fp_n,
        "fn": fn_n,
        "curves": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf": conf_sorted,
            "p_grid": interp_by_conf(precision, conf_sorted),
            "r_grid": interp_by_conf(recall, conf_sorted),
            "f1_grid": interp_by_conf(f1, conf_sorted),
            "grid": CONF_GRID,
        },
    }


# ------------------------------------------------------------------------------------------
# classification metrics (stage 2)
# ------------------------------------------------------------------------------------------
def roc_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Binary ROC AUC via the rank (Mann-Whitney U) identity, with tie correction.

    Implemented here rather than pulled from sklearn: sklearn is a heavy dependency for one number,
    and this keeps the metrics module numpy-only and unit-testable.
    """
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_score = np.asarray(list(y_score), dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # average rank for ties
        i = j + 1

    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_report(y_true: Sequence[int], y_pred: Sequence[int],
                          tb_score: Sequence[float], tb_index: int = 2) -> dict:
    """Paper Table 3 columns for the 3-class {healthy, sick_non_tb, tb} head.

    Sensitivity/specificity are defined against the **TB vs non-TB** binarisation, which is the
    clinically meaningful one and the one the paper reports.
    """
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_pred = np.asarray(list(y_pred), dtype=np.int64)

    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    is_tb, pred_tb = (y_true == tb_index), (y_pred == tb_index)
    tp = int((is_tb & pred_tb).sum())
    tn = int((~is_tb & ~pred_tb).sum())
    fp = int((~is_tb & pred_tb).sum())
    fn = int((is_tb & ~pred_tb).sum())

    sens = tp / (tp + fn) if (tp + fn) else float("nan")     # recall on TB
    spec = tn / (tn + fp) if (tn + fp) else float("nan")     # the column baselines lose on

    n_cls = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1
    precisions, recalls = [], []
    for c in range(n_cls):
        t, p = (y_true == c), (y_pred == c)
        tp_c = int((t & p).sum())
        precisions.append(tp_c / max(int(p.sum()), 1))
        recalls.append(tp_c / max(int(t.sum()), 1))

    return {
        "accuracy": acc * 100,
        "auc_tb": roc_auc(is_tb.astype(int), list(tb_score)) * 100,
        "sensitivity": sens * 100,
        "specificity": spec * 100,
        "AP": float(np.mean(precisions)) * 100,
        "AR": float(np.mean(recalls)) * 100,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "per_class_precision": [p * 100 for p in precisions],
        "per_class_recall": [r * 100 for r in recalls],
    }
