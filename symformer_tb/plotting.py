"""Run-directory figures: training curves, PR/F1 curves, confusion matrix, batch mosaics.

Every function takes an output path, writes a PNG, and **never raises** -- a plotting failure must
not kill a 20-minute training run. Failures print a one-line warning and return ``False``.

Matplotlib is forced onto the ``Agg`` backend at import: this module is called from scripts running
under ``subprocess`` with no display, and on Windows a GUI backend can otherwise try to open a window
and hang.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour-blind-safe; also legible when a run's figures get pasted into a greyscale report.
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"


def _guard(fn):
    """Wrap a plotter so a failure warns instead of killing training."""
    def wrapper(*a, **kw):
        out = kw.get("out") or (a[-1] if a else "?")
        try:
            fn(*a, **kw)
            return True
        except Exception as e:  # pragma: no cover - defensive
            print(f"[warn] could not write {out}: {type(e).__name__}: {e}")
            plt.close("all")
            return False
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ------------------------------------------------------------------------------------------
# training curves
# ------------------------------------------------------------------------------------------
@_guard
def plot_results(csv_path: str | Path, out: str | Path) -> None:
    """``results.csv`` -> ``results.png``: losses and metrics against epoch.

    Mirrors Ultralytics' results.png so the run is readable at a glance: top row is what training
    is minimising, bottom row is what we actually care about.
    """
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): v for k, v in r.items()})
    if not rows:
        raise ValueError("results.csv is empty")

    def col(name: str) -> np.ndarray:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(name, "nan")))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        return np.array(vals, dtype=np.float64)

    epochs = col("epoch")
    panels = [
        ("train/box_loss", "box loss", BLUE),
        ("train/cls_loss", "cls loss", ORANGE),
        ("train/total_loss", "total loss", GREY),
        ("metrics/precision", "precision", GREEN),
        ("metrics/recall", "recall", GREEN),
        ("metrics/mAP50", "mAP50", RED),
        ("metrics/mAP50-95", "mAP50-95", RED),
        ("lr/0", "learning rate", GREY),
    ]
    present = [p for p in panels if p[0] in rows[0]]
    if not present:
        raise ValueError("no plottable columns in results.csv")

    ncols = 4
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, (key, title, colour) in zip(axes.flat, present):
        y = col(key)
        ax.plot(epochs, y, marker=".", markersize=4, linewidth=1.6, color=colour)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        if np.isfinite(y).any() and "loss" not in key and "lr" not in key:
            ax.set_ylim(bottom=0)
    for ax in axes.flat[len(present):]:
        ax.axis("off")

    fig.suptitle(Path(out).parent.name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# PR / F1 curves
# ------------------------------------------------------------------------------------------
@_guard
def plot_pr_curve(curves: dict, ap50: float, out: str | Path) -> None:
    """Precision against recall, annotated with the mAP50 it integrates to."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curves["recall"], curves["precision"], linewidth=2, color=BLUE,
            label=f"TB  (AP50 = {ap50:.1f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Precision-Recall (IoU 0.5)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


@_guard
def plot_f1_curve(curves: dict, out: str | Path) -> None:
    """P, R and F1 against confidence, with the max-F1 operating point marked.

    That marked point is the confidence at which the trainer's printed P/R row is measured.
    """
    grid = curves["grid"]
    f1 = curves["f1_grid"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(grid, curves["p_grid"], linewidth=1.4, color=GREEN, label="precision")
    ax.plot(grid, curves["r_grid"], linewidth=1.4, color=ORANGE, label="recall")
    ax.plot(grid, f1, linewidth=2.2, color=BLUE, label="F1")
    if np.any(f1 > 0):
        i = int(np.argmax(f1))
        ax.axvline(grid[i], color=GREY, linestyle="--", linewidth=1)
        ax.plot([grid[i]], [f1[i]], "o", color=RED, markersize=6)
        ax.annotate(f"best F1 {f1[i]:.3f} @ {grid[i]:.2f}",
                    xy=(grid[i], f1[i]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("confidence")
    ax.set_ylabel("metric")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower center", fontsize=9)
    ax.set_title("P / R / F1 vs confidence (IoU 0.5)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


@_guard
def plot_confusion(tp: int, fp: int, fn: int, out: str | Path, conf: float = 0.0) -> None:
    """Single-class confusion matrix.

    The background/background cell is left blank rather than filled with a number: with one
    category-agnostic class, "true negative" would mean counting the boxes the model correctly did
    not draw, which is unbounded and meaningless.
    """
    mat = np.array([[tp, fp], [fn, 0]], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(np.array([[tp, fp], [fn, np.nan]]), cmap="Blues")
    ax.set_xticks([0, 1], ["TB", "background"])
    ax.set_yticks([0, 1], ["TB", "background"])
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    labels = [[f"TP\n{tp}", f"FP\n{fp}"], [f"FN\n{fn}", "-"]]
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            colour = "white" if (i, j) != (1, 1) and val > np.nanmax(mat) * 0.6 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=11, color=colour)
    ax.set_title(f"Confusion @ conf {conf:.2f}, IoU 0.5")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# dataset / batch visualisation
# ------------------------------------------------------------------------------------------
@_guard
def plot_labels(boxes_xywh: Sequence[Sequence[float]], img_size: int, out: str | Path) -> None:
    """Ground-truth box statistics: count per image, size distribution, spatial density.

    The spatial-density panel is the one worth reading for this project: SymFormer's whole premise
    is a left-right asymmetry prior, so a visibly lopsided box distribution is context for the
    result -- as is a symmetric one.
    """
    arr = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    if len(arr) == 0:
        raise ValueError("no boxes to plot")

    cx = (arr[:, 0] + arr[:, 2] / 2) / img_size
    cy = (arr[:, 1] + arr[:, 3] / 2) / img_size
    w = arr[:, 2] / img_size
    h = arr[:, 3] / img_size

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].hist(np.sqrt(w * h), bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    axes[0].set_title(f"box scale  (n = {len(arr)})")
    axes[0].set_xlabel("sqrt(w*h), normalised")

    axes[1].scatter(w, h, s=6, alpha=0.35, color=ORANGE)
    axes[1].set_title("width vs height")
    axes[1].set_xlabel("w")
    axes[1].set_ylabel("h")

    axes[2].hexbin(cx, cy, gridsize=28, cmap="Blues", mincnt=1)
    axes[2].axvline(0.5, color=RED, linestyle="--", linewidth=1.2)
    axes[2].set_title("centre density (red = vertical centerline)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].invert_yaxis()

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


@_guard
def plot_batch(images, targets, out: str | Path, preds=None, max_images: int = 8,
               conf_thr: float = 0.25) -> None:
    """Mosaic of a batch with ground truth (green) and, optionally, predictions (red).

    ``images`` are CHW float tensors in [0, 1]; ``targets``/``preds`` are the dicts the trainer
    passes around (``boxes`` xyxy, plus ``scores`` on preds). Accepts torch tensors or numpy.
    """
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    n = min(len(images), max_images)
    if n == 0:
        raise ValueError("empty batch")
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)

    for idx in range(n):
        ax = axes.flat[idx]
        img = to_numpy(images[idx])
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img[..., 0]
        ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None)

        for box in to_numpy(targets[idx]["boxes"]).reshape(-1, 4):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=GREEN, linewidth=1.6))

        if preds is not None:
            pboxes = to_numpy(preds[idx]["boxes"]).reshape(-1, 4)
            pscores = to_numpy(preds[idx].get("scores", np.ones(len(pboxes)))).reshape(-1)
            for box, score in zip(pboxes, pscores):
                if score < conf_thr:
                    continue
                x1, y1, x2, y2 = box
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=RED, linewidth=1.2, linestyle="--"))
                ax.text(x1, max(y1 - 3, 8), f"{score:.2f}", color=RED, fontsize=7)
        ax.axis("off")

    for ax in axes.flat[n:]:
        ax.axis("off")
    title = "green = ground truth" + ("   |   red dashed = prediction" if preds is not None else "")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


@_guard
def plot_cls_confusion(matrix: np.ndarray, class_names: Sequence[str], out: str | Path) -> None:
    """Stage-2 3-class confusion matrix (healthy / sick-non-TB / TB), row-normalised."""
    mat = np.asarray(matrix, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{int(mat[i, j])}\n{norm[i, j] * 100:.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_title("Stage-2 classification")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# Optional-figure convenience used by the trainer: write everything a validation pass can produce.
def write_val_figures(metrics: dict, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    curves = metrics.get("curves")
    if not curves:
        return
    plot_pr_curve(curves, metrics.get("AP50", 0.0), run_dir / "PR_curve.png")
    plot_f1_curve(curves, run_dir / "F1_curve.png")
    plot_confusion(metrics.get("tp", 0), metrics.get("fp", 0), metrics.get("fn", 0),
                   run_dir / "confusion_matrix.png", conf=metrics.get("conf", 0.0))
