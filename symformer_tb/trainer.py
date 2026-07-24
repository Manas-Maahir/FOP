"""Stack-agnostic training loop with an Ultralytics-style console, run directory and checkpoints.

Replaces the loop that lived in ``tools/tv_train.py``, which printed ``ep 3 [20/75] loss 0.8340``
every 20 batches and appended JSONL. What you get instead, per epoch:

      Epoch    GPU_mem   box_loss   cls_loss  Instances       Size
       1/24      4.21G     0.8340     1.2031         14        512: 100%|######| 75/75 [00:38<00:00]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        200        271       61.2       54.9       79.1      33.4

**All metrics are 0-100, including precision and recall.** Ultralytics prints 0-1, but this project's
numbers are quoted as percentages everywhere else -- paper Table 8, [report.md](report.md) §5's
"79.1 AP50" -- and a reader comparing the two should never have to rescale in their head.

Run directory (auto-incremented ``runs/detect/train``, ``train2``, ...):

    args.yaml  results.csv  results.png  labels.jpg
    train_batch{0,1,2}.jpg  val_batch0_pred.jpg
    PR_curve.png  F1_curve.png  confusion_matrix.png
    weights/best.pt  weights/last.pt

Checkpointing is designed for a machine you might close the lid on: ``last.pt`` is written every
epoch, Ctrl-C saves before exiting, and ``--resume`` restores model + EMA + optimizer + scheduler +
scaler + epoch so the LR schedule continues rather than restarting.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import signal
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch

from .metrics import evaluate_detections, fitness as compute_fitness
from . import plotting

# Column widths chosen so the epoch row and the metric row line up under each other.
TRAIN_HEADER = ("%11s" * 6) % ("Epoch", "GPU_mem", "box_loss", "cls_loss", "Instances", "Size")
VAL_HEADER = ("%22s" + "%11s" * 6) % (
    "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")


# ------------------------------------------------------------------------------------------
# small utilities
# ------------------------------------------------------------------------------------------
def colorstr(*args) -> str:
    """ANSI-colour a string: ``colorstr('blue', 'bold', 'text')``. No-ops if colour is disabled."""
    *style, text = args if len(args) > 1 else ("blue", "bold", args[0])
    if os.environ.get("NO_COLOR") or not _color_ok():
        return text
    codes = {"blue": "\033[34m", "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
             "bold": "\033[1m", "underline": "\033[4m"}
    return "".join(codes.get(s, "") for s in style) + text + "\033[0m"


_COLOR_INIT = None


def _color_ok() -> bool:
    """Windows conhost needs colorama; Windows Terminal and every POSIX shell do not."""
    global _COLOR_INIT
    if _COLOR_INIT is None:
        _COLOR_INIT = True
        if platform.system() == "Windows":
            try:
                import colorama

                colorama.just_fix_windows_console()
            except Exception:
                _COLOR_INIT = False
    return _COLOR_INIT


def increment_path(path: str | Path) -> Path:
    """``runs/detect/train`` -> ``runs/detect/train2`` if it already exists.

    Never overwrites a previous run: losing yesterday's numbers to a re-run is the kind of thing you
    only notice once the machine is busy with the next 20-minute job.
    """
    path = Path(path)
    if not path.exists():
        return path
    for n in range(2, 10000):
        candidate = path.parent / f"{path.name}{n}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free run directory next to {path}")


def latest_run(base: str | Path = "runs/detect") -> Optional[Path]:
    """Most recently modified ``*/weights/last.pt`` -- what a bare ``--resume`` picks up."""
    weights = sorted(Path(base).glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    return weights[-1] if weights else None


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent.parent)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "0G"
    return f"{torch.cuda.max_memory_reserved() / 1e9:.3g}G"


def _autocast(enabled: bool):
    """AMP context that works on both torch 2.1 (this project's pin) and newer releases."""
    try:  # torch >= 2.4
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - version dependent
        return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(enabled: bool):
    try:  # torch >= 2.4
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - version dependent
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ------------------------------------------------------------------------------------------
# EMA
# ------------------------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of the weights.

    Standard detection-training trick: the EMA copy is usually a point or two of AP better than the
    live weights and much less noisy epoch to epoch, which matters here because
    [report.md](report.md) §6 found the run-to-run spread (~0.3-0.8 AP50) was the same order as the
    effect being measured. Decay ramps in so the average is not dominated by the random init.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# ------------------------------------------------------------------------------------------
# checkpoints
# ------------------------------------------------------------------------------------------
def strip_optimizer(path: str | Path) -> float:
    """Drop optimizer state from a finished checkpoint. Returns the new size in MB.

    A training checkpoint is ~450 MB; inference needs ~145 MB of it. Worth doing at the end of a
    run, because the ablation sweep produces dozens of them.

    Only ever applied to ``best.pt``. ``last.pt`` is the resume point, and stripping it would let a
    later ``--resume`` (e.g. to extend a finished run to more epochs) silently continue with a fresh
    optimizer and a restarted LR schedule -- which looks like it worked while training a different
    model than you asked for.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("optimizer", "scheduler", "scaler", "ema_updates"):
        ckpt.pop(key, None)
    ckpt["stripped"] = True
    torch.save(ckpt, path)
    return path.stat().st_size / 1e6


# ------------------------------------------------------------------------------------------
# trainer
# ------------------------------------------------------------------------------------------
class Trainer:
    """Owns one training run: the loop, the run directory, the checkpoints and the console."""

    def __init__(self, adapter, train_loader, val_loader, val_ann: Optional[str],
                 args, run_dir: Path, device: torch.device):
        self.adapter = adapter
        self.model = adapter.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_ann = val_ann
        self.args = args
        self.run_dir = Path(run_dir)
        self.device = device

        self.weights_dir = self.run_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "results.csv"
        self.last_pt = self.weights_dir / "last.pt"
        self.best_pt = self.weights_dir / "best.pt"

        self.start_epoch = 0
        self.best_fitness = -1.0
        self.best_epoch = -1
        self.interrupted = False

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.params = params
        self.optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.milestones, gamma=0.1)

        self.amp = bool(args.amp) and device.type == "cuda"
        self.scaler = _grad_scaler(self.amp)
        self.ema = ModelEMA(self.model) if args.ema else None

        self._install_sigint()

    # -- lifecycle -----------------------------------------------------------------------
    def _install_sigint(self):
        """Ctrl-C saves last.pt and stops cleanly, instead of losing the epoch."""
        def handler(signum, frame):
            if self.interrupted:      # second Ctrl-C: give up immediately
                raise KeyboardInterrupt
            self.interrupted = True
            print(colorstr("yellow", "bold",
                           "\nInterrupt received -- finishing this batch, saving last.pt, "
                           "then stopping. Press Ctrl-C again to abort now."))
        try:
            signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            pass

    def resume_from(self, ckpt_path: str | Path) -> None:
        """Restore a run: weights, EMA, optimizer, scheduler, scaler and the epoch counter.

        Restoring the optimizer and scheduler is the point -- resuming with a fresh optimizer
        silently restarts the LR schedule and momentum, which looks like it worked but trains a
        different model than the one you meant to continue.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and self.amp:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and ckpt.get("ema"):
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.updates = ckpt.get("ema_updates", 0)
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.best_fitness = float(ckpt.get("best_fitness", -1.0))
        self.best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"resumed {ckpt_path} -> starting at epoch {self.start_epoch + 1}"
              f" (best fitness {self.best_fitness:.4f} @ epoch {self.best_epoch + 1})")

    def save_ckpt(self, epoch: int, path: Path, extra: Optional[dict] = None) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None,
            "ema_updates": self.ema.updates if self.ema else 0,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.amp else None,
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "best_epoch": self.best_epoch,
            "args": vars(self.args),
            "sas_cfg": getattr(self.adapter, "sas_cfg", None),
            "stack": getattr(self.adapter, "stack", "torchvision"),
            "seed": self.args.seed,
            "torch": torch.__version__,
            "git_sha": git_sha(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    # -- logging -------------------------------------------------------------------------
    def log_row(self, row: dict) -> None:
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def save_args(self) -> None:
        payload = dict(vars(self.args))
        payload.update({
            "run_dir": str(self.run_dir),
            "git_sha": git_sha(),
            "torch": torch.__version__,
            "device": str(self.device),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_images": len(self.train_loader.dataset),
            "val_images": len(self.val_loader.dataset) if self.val_loader else 0,
        })
        try:
            import yaml

            (self.run_dir / "args.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        except Exception:
            (self.run_dir / "args.yaml").write_text(json.dumps(payload, indent=2, default=str))

    # -- the loop ------------------------------------------------------------------------
    def train(self) -> dict:
        from tqdm import tqdm

        args = self.args
        self.save_args()
        nb = len(self.train_loader)
        warmup_iters = min(args.warmup_iters, max(nb * 3, 1))
        global_step = self.start_epoch * nb
        t_start = time.time()
        last_metrics: dict = {}

        print(colorstr(f"\nStarting training for {args.epochs} epochs on {self.device} "
                       f"(AMP {'on' if self.amp else 'off'}, EMA {'on' if self.ema else 'off'})"))
        print(f"Results will be saved to {colorstr('bold', str(self.run_dir))}\n")

        for epoch in range(self.start_epoch, args.epochs):
            self.model.train()
            print(colorstr(TRAIN_HEADER))
            pbar = tqdm(enumerate(self.train_loader), total=nb, ncols=118, leave=True,
                        bar_format="{l_bar}{bar:12}{r_bar}")
            running = {"box_loss": 0.0, "cls_loss": 0.0, "total": 0.0}
            seen = 0

            for i, (images, targets) in pbar:
                if args.limit_batches and i >= args.limit_batches:
                    break

                # linear warmup -- the peak LR is where torchvision RetinaNet used to blow up
                if global_step < warmup_iters:
                    warm = (global_step + 1) / warmup_iters
                    for g in self.optimizer.param_groups:
                        g["lr"] = args.lr * warm

                images = [im.to(self.device, non_blocking=True) for im in images]
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                           for t in targets]

                try:
                    with _autocast(self.amp):
                        losses = self.adapter.loss(images, targets)
                        loss = sum(losses.values())
                except torch.cuda.OutOfMemoryError:
                    self._oom_help()
                    raise

                if not math.isfinite(loss.item()):
                    self._divergence_help(losses)
                    return {"status": "diverged", "epoch": epoch}

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    # unscale first: clipping scaled gradients clips the wrong magnitude
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.params, args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.ema is not None:
                    self.ema.update(self.model)

                seen += 1
                global_step += 1
                running["box_loss"] += float(losses.get("box_loss", 0.0))
                running["cls_loss"] += float(losses.get("cls_loss", 0.0))
                running["total"] += float(loss)
                n_inst = sum(int(t["boxes"].shape[0]) for t in targets)

                pbar.set_description(("%11s" * 2 + "%11.4g" * 4) % (
                    f"{epoch + 1}/{args.epochs}", gpu_mem_str(),
                    running["box_loss"] / seen, running["cls_loss"] / seen,
                    n_inst, args.image_size))

                if epoch == self.start_epoch and i < 3:
                    plotting.plot_batch(images, targets, self.run_dir / f"train_batch{i}.jpg")

                if self.interrupted:
                    break

            self.scheduler.step()
            n = max(seen, 1)
            row = {
                "epoch": epoch + 1,
                "time": round(time.time() - t_start, 1),
                "train/box_loss": round(running["box_loss"] / n, 5),
                "train/cls_loss": round(running["cls_loss"] / n, 5),
                "train/total_loss": round(running["total"] / n, 5),
                "lr/0": self.optimizer.param_groups[0]["lr"],
            }

            # -- validation ---------------------------------------------------------------
            metrics = {}
            if self.val_loader is not None and args.eval_every > 0 \
                    and (epoch + 1) % args.eval_every == 0:
                metrics = self.validate(verbose=True)
                last_metrics = metrics
                row.update({
                    "metrics/precision": round(metrics["precision"], 4),
                    "metrics/recall": round(metrics["recall"], 4),
                    "metrics/mAP50": round(metrics["AP50"], 4),
                    "metrics/mAP50-95": round(metrics["AP"], 4),
                })
                self.model.train()

            self.log_row(row)

            # -- checkpoints ---------------------------------------------------------------
            fit = compute_fitness(metrics.get("AP", 0.0), metrics.get("AP50", 0.0),
                                  args.fitness) if metrics else None
            if fit is not None and fit > self.best_fitness:
                self.best_fitness, self.best_epoch = fit, epoch
            self.save_ckpt(epoch, self.last_pt)
            if fit is not None and self.best_epoch == epoch:
                self.save_ckpt(epoch, self.best_pt)
                print(colorstr("green", "bold",
                               f"  new best: fitness {fit:.4f} "
                               f"(mAP50 {metrics['AP50']:.1f}) -> weights/best.pt"))
            if args.save_period > 0 and (epoch + 1) % args.save_period == 0:
                self.save_ckpt(epoch, self.weights_dir / f"epoch{epoch + 1}.pt")

            plotting.plot_results(self.csv_path, self.run_dir / "results.png")

            if self.interrupted:
                print(colorstr("yellow", "bold",
                               f"\nStopped at epoch {epoch + 1}. Resume with:\n"
                               f"    python tools/train.py --resume {self.last_pt}"))
                return {"status": "interrupted", "epoch": epoch, **last_metrics}

        # -- finish ------------------------------------------------------------------------
        hours = (time.time() - t_start) / 3600
        print(f"\n{args.epochs - self.start_epoch} epochs completed in {hours:.3f} hours.")

        final = last_metrics
        if self.best_pt.exists() and self.val_loader is not None:
            print(f"\nValidating {self.best_pt} ...")
            best_metrics = self.validate(weights=self.best_pt, verbose=True)
            plotting.write_val_figures(best_metrics, self.run_dir)
            final = {"final_epoch": last_metrics, "best": best_metrics}

        if self.best_pt.exists():                       # last.pt keeps its optimizer for --resume
            mb = strip_optimizer(self.best_pt)
            print(f"Optimizer stripped from {self.best_pt}, {mb:.1f}MB")

        if last_metrics:
            # report.md's convention: the headline is the final-epoch number, because best.pt was
            # selected on val and its AP is therefore optimistically biased.
            print(colorstr("bold", "\nHeadline (final epoch, unbiased): ")
                  + f"AP50 = {last_metrics['AP50']:.1f}   AP = {last_metrics['AP']:.1f}")
            if self.best_epoch >= 0:
                print(f"Best-on-val checkpoint was epoch {self.best_epoch + 1} "
                      f"(reported for reference only).")

        print(f"Results saved to {colorstr('bold', str(self.run_dir))}")
        return {"status": "complete", **(last_metrics or {}), "run_dir": str(self.run_dir),
                "best_epoch": self.best_epoch}

    # -- validation ----------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, weights: Optional[Path] = None, verbose: bool = False,
                 save_preds: bool = True) -> dict:
        """One validation pass -> the metric bundle. Uses the EMA weights when available."""
        from .evaluate import predict_coco

        model_for_eval = self.model
        restore = None
        if weights is not None:
            ckpt = torch.load(weights, map_location=self.device, weights_only=False)
            restore = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(ckpt["ema"] or ckpt["model"])
        elif self.ema is not None:
            restore = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema.state_dict())

        model_for_eval.eval()
        results = predict_coco(self.adapter, self.val_loader, self.device)
        metrics = evaluate_detections(results, self.val_ann, quiet=True)

        if verbose:
            print(colorstr(VAL_HEADER))
            print(("%22s" + "%11s" * 2 + "%11.4g" * 4) % (
                "all", metrics["n_images"], metrics["n_instances"],
                metrics["precision"] * 100, metrics["recall"] * 100,
                metrics["AP50"], metrics["AP"]))
            if metrics["n_dets"] == 0:
                print(colorstr("yellow", "  no detections above threshold -- "
                                         "expected for a 1-epoch smoke run, not for a real run"))

        if save_preds and self.val_loader is not None:
            self._save_pred_figure()

        if restore is not None:
            self.model.load_state_dict(restore)
        return metrics

    def _save_pred_figure(self) -> None:
        try:
            images, targets = next(iter(self.val_loader))
            imgs = [im.to(self.device) for im in images]
            preds = self.adapter.predict(imgs)
            plotting.plot_batch(images, targets, self.run_dir / "val_batch0_pred.jpg", preds=preds)
        except Exception as e:  # pragma: no cover - cosmetic only
            print(f"[warn] could not write val_batch0_pred.jpg: {e}")

    # -- diagnostics ---------------------------------------------------------------------
    def _oom_help(self) -> None:
        print(colorstr("red", "bold", "\nCUDA out of memory."))
        print("  This build targets an 8 GB laptop GPU; the paper's batch 8 at 512px is close to")
        print("  the limit. Retry with a smaller batch and a linearly scaled LR:")
        print(f"      --batch-size 4 --lr {self.args.lr / 2:.4g}")
        print("      --batch-size 2 --lr %.4g" % (self.args.lr / 4))
        print("  Or keep the batch and add --accumulate 2 for the same effective batch size.")

    def _divergence_help(self, losses: dict) -> None:
        print(colorstr("red", "bold", "\nNon-finite loss -- training diverged."))
        print("  losses:", {k: float(v) for k, v in losses.items()})
        print(f"  lr={self.optimizer.param_groups[0]['lr']:.5g}, grad_clip={self.args.grad_clip}")
        print("  torchvision's RetinaNet can spike bbox_regression as the warmup LR peaks. Lower")
        print("  the peak LR (--lr 0.0025) or tighten the clip (--grad-clip 5), and start from a")
        print("  FRESH run dir so it does not resume the diverged checkpoint.")
