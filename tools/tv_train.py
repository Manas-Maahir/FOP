#!/usr/bin/env python
"""Train RetinaNet (baseline) or SymFormer (RetinaNet + SAS) on the TB-only 512 COCO set.

Implements the paper's stage-1 detection recipe: SGD, batch 8, 24 epochs, 512x512, random
horizontal flip, fixed seed. Checkpoints every epoch to --work-dir and auto-resumes.

Storage: keep --work-dir on /content (ephemeral VM disk, ~100GB) and let --drive-sync copy the
tiny logs to Drive. Do NOT point --work-dir at Drive: checkpoints are ~300MB/epoch and Colab's
Drive mount turns our pruning of the previous one into a move to Trash, which keeps counting
against the 15GB quota (~7GB trashed per 24-epoch run). A run takes ~15-20min, so retraining is
cheaper than storing, and the logs already hold every AP/AP50.

Baseline (Table 8 "No / No"):
    python tools/tv_train.py --work-dir /content/work/baseline --data-root DATA/ --no-sas \
        --drive-sync /content/drive/MyDrive/tb_runs/baseline

Full SymFormer (SymAttention + SPE + STN, right->left):
    python tools/tv_train.py --work-dir /content/work/symformer --data-root DATA/ \
        --attention symattention --pe spe --stn --direction r2l \
        --drive-sync /content/drive/MyDrive/tb_runs/symformer

Ablation cells vary --attention / --pe / --stn / --direction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True,
                    help="checkpoints + logs. MUST be on /content (ephemeral), never Drive: "
                         "checkpoints are ~300MB/epoch and Colab's Drive mount sends deletions to "
                         "Trash, which keeps counting against the 15GB quota. A full run here is "
                         "only ~15-20min, so retraining is cheaper than storing. Use --drive-sync "
                         "to copy the tiny logs to Drive.")
    ap.add_argument("--data-root", required=True, help="compact dataset root (tbx11k_tb512)")
    ap.add_argument("--train-ann", default="annotations/tb_train_agnostic.json")
    ap.add_argument("--train-img", default="images/train")
    ap.add_argument("--val-ann", default="annotations/tb_val_agnostic.json")
    ap.add_argument("--val-img", default="images/val")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run val AP every N epochs (0 disables). Gives the convergence curve and "
                         "drives best.pth selection.")
    ap.add_argument("--drive-sync", default=None,
                    help="optional Drive dir to copy the logs into after each epoch. Logs are a few "
                         "KB and hold every AP/AP50 — they ARE the results. Weights stay on "
                         "--work-dir unless --sync-weights.")
    ap.add_argument("--sync-weights", action="store_true",
                    help="also copy best.pth (~300MB) to --drive-sync. Off by default: the headline "
                         "number is the final-epoch AP, which is already in the logs, so weights are "
                         "disposable. Turn on only to keep one model (e.g. for a figure).")
    # SAS options (omit --no-sas to enable the SAS block)
    ap.add_argument("--no-sas", action="store_true", help="plain RetinaNet baseline")
    ap.add_argument("--attention", default="symattention", choices=["vanilla", "symattention"])
    ap.add_argument("--pe", default="spe", choices=["none", "ape", "rpe", "spe"])
    ap.add_argument("--stn", action="store_true", help="enable the STN inside SPE")
    ap.add_argument("--no-stn", dest="stn", action="store_false")
    ap.set_defaults(stn=True)
    ap.add_argument("--direction", default="r2l", choices=["r2l", "l2r"])
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-points", type=int, default=4)
    # schedule (paper stage-1)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.005, help="0.01 is tuned for batch 16; linear-scale")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--milestones", type=int, nargs="+", default=[16, 22])
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--max-keep-ckpts", type=int, default=1,
                    help="checkpoints are ~300MB; keep few to bound --work-dir disk usage")
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke test: stop after N batches")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    from symformer_tb.tv_model import build_model, count_parameters
    from symformer_tb.tv_dataset import build_loader
    from symformer_tb.evaluate import evaluate_model

    torch.manual_seed(args.seed)
    _warn_if_work_dir_on_drive(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed}")

    sas_cfg = None
    if not args.no_sas:
        sas_cfg = dict(attention=args.attention, pe=args.pe, use_stn=args.stn,
                       direction=args.direction, num_heads=args.num_heads,
                       num_points=args.num_points)
    print("SAS config:", sas_cfg if sas_cfg else "(none — plain RetinaNet baseline)")

    model = build_model(sas=sas_cfg, image_size=args.image_size).to(device)
    print("params:", count_parameters(model))

    root = args.data_root.rstrip("/")
    loader = build_loader(os.path.join(root, args.train_ann), os.path.join(root, args.train_img),
                          batch_size=args.batch_size, train=True,
                          num_workers=args.num_workers, seed=args.seed)
    print(f"train images: {len(loader.dataset)}  batches/epoch: {len(loader)}")

    val_loader, val_ann = None, None
    if args.eval_every > 0:
        val_ann = os.path.join(root, args.val_ann)
        if os.path.isfile(val_ann):
            val_loader = build_loader(val_ann, os.path.join(root, args.val_img),
                                      batch_size=4, train=False, num_workers=args.num_workers)
            print(f"val images  : {len(val_loader.dataset)} (AP every {args.eval_every} epoch/s)")
        else:
            print(f"[warn] no val annotations at {val_ann}; per-epoch eval disabled")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                     gamma=0.1)

    # ---- resume -----------------------------------------------------------------------
    start_epoch, best_ap50, best_epoch = 0, -1.0, -1
    last_path = os.path.join(args.work_dir, "last.pth")
    if os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ap50 = ckpt.get("best_ap50", -1.0)
        best_epoch = ckpt.get("best_epoch", -1)
        print(f"resumed from {last_path} at epoch {start_epoch} (best AP50 so far {best_ap50:.1f})")
    if start_epoch >= args.epochs:
        print(f"already trained {start_epoch}/{args.epochs} epochs — nothing to do.")
        return 0

    log_path = os.path.join(args.work_dir, "train_log.jsonl")
    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for i, (images, targets) in enumerate(loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not math.isfinite(loss.item()):
                print("non-finite loss, stopping:", {k: v.item() for k, v in loss_dict.items()})
                return 1

            # linear warmup
            if global_step < args.warmup_iters:
                warm = (global_step + 1) / args.warmup_iters
                for g in optimizer.param_groups:
                    g["lr"] = args.lr * warm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            global_step += 1
            if i % 20 == 0:
                print(f"ep {epoch} [{i}/{len(loader)}] loss {loss.item():.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.5f}")
        scheduler.step()
        n = (args.limit_batches or len(loader))
        rec = {"epoch": epoch, "loss": running / max(1, n), "secs": round(time.time() - t0, 1),
               "lr": optimizer.param_groups[0]["lr"]}

        # ---- per-epoch validation: the convergence curve + best.pth selection ----
        is_best = False
        if val_loader is not None and (epoch + 1) % args.eval_every == 0:
            ap, ap50 = evaluate_model(model, val_loader, val_ann, device, quiet=True)
            rec["AP"], rec["AP50"] = round(ap, 2), round(ap50, 2)
            if ap50 > best_ap50:
                best_ap50, best_epoch, is_best = ap50, epoch, True
            rec["best_AP50"] = round(best_ap50, 2)
            model.train()
        print("EPOCH DONE:", rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "sas_cfg": sas_cfg, "args": vars(args),
                "best_ap50": best_ap50, "best_epoch": best_epoch}
        torch.save(ckpt, last_path)                                   # for resume
        torch.save(ckpt, os.path.join(args.work_dir, f"epoch_{epoch+1}.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(args.work_dir, "best.pth"))
            print(f"  ** new best AP50 {best_ap50:.1f} @ epoch {epoch+1} -> best.pth")
        _prune_ckpts(args.work_dir, args.max_keep_ckpts)
        _drive_sync(args.work_dir, args.drive_sync, args.sync_weights)

    print(f"training complete -> {args.work_dir}")
    if best_epoch >= 0:
        print(f"best val AP50 = {best_ap50:.1f} at epoch {best_epoch+1} (best.pth)")
        print("NOTE: report the FINAL-epoch AP as the headline (paper-faithful, unbiased); "
              "best.pth is selected on val, so its AP is optimistically biased.")
    return 0


def _warn_if_work_dir_on_drive(work_dir: str):
    """A --work-dir on Drive quietly burns the 15GB quota; say so loudly.

    Colab's Drive mount turns every deletion into a move to Drive's Trash, and trashed files keep
    counting against quota for 30 days. Since we write ~300MB per epoch and prune the previous one,
    a 24-epoch run silently trashes ~7GB. Warn rather than fail: someone with a big Drive may
    legitimately want this.

    Substring rather than prefix match: abspath() on a non-Colab box prepends a drive letter
    (C:/content/drive/...), which would slip past startswith() and make this untestable off-Colab.
    """
    if "/content/drive/" in os.path.abspath(work_dir).replace("\\", "/"):
        # ASCII only: this runs before training on whatever console the user has, and a
        # UnicodeEncodeError here would kill the run the warning exists to protect.
        print("=" * 78)
        print("!! --work-dir is on Google Drive. Checkpoints are ~300MB/epoch and Colab's Drive")
        print("   mount sends deletions to TRASH, which still counts against your quota: a")
        print("   24-epoch run will trash ~7GB. Prefer:")
        print("       --work-dir /content/work/<run>  --drive-sync <drive_dir>")
        print("   which keeps weights on the VM's ephemeral disk and copies only the tiny logs.")
        print("=" * 78)


def _drive_sync(work_dir: str, drive_dir: Optional[str], sync_weights: bool = False):
    """Copy the logs (and optionally best.pth) to Drive; tolerate a full/unavailable Drive.

    Logs only by default. They are a few KB, they carry every AP/AP50, and the headline number is
    the final-epoch AP — so the logs are the results and the ~300MB weights are disposable.

    Only train_log.jsonl is ours. eval_log.jsonl belongs to tv_eval.py, which appends its own
    entries to Drive; copying it from here would overwrite that accumulated history.
    """
    if not drive_dir:
        return
    import shutil
    names = ["train_log.jsonl"]
    if sync_weights:
        names.append("best.pth")
    try:
        os.makedirs(drive_dir, exist_ok=True)
        for name in names:
            src = os.path.join(work_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(drive_dir, name))
    except OSError as e:
        # e.g. "Google Drive storage quota has been exceeded" — never kill a run over this
        print(f"[warn] drive sync failed ({e}); continuing. Logs remain in {work_dir}")


def _prune_ckpts(work_dir: str, keep: int):
    """Keep only the newest `keep` epoch_*.pth (each is ~300MB)."""
    import re
    files = [f for f in os.listdir(work_dir) if re.fullmatch(r"epoch_\d+\.pth", f)]
    files.sort(key=lambda f: int(re.findall(r"\d+", f)[0]))
    for f in files[:-keep] if keep > 0 else []:
        os.remove(os.path.join(work_dir, f))


if __name__ == "__main__":
    sys.exit(main())
