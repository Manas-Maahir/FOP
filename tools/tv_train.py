#!/usr/bin/env python
"""Train RetinaNet (baseline) or SymFormer (RetinaNet + SAS) on the TB-only 512 COCO set.

Implements the paper's stage-1 detection recipe: SGD, batch 8, 24 epochs, 512x512, random
horizontal flip, fixed seed. Checkpoints every epoch to --work-dir and auto-resumes, so a Colab
time-out costs at most the current epoch (point --work-dir at Google Drive).

Baseline (Table 8 "No / No"):
    python tools/tv_train.py --work-dir RUNS/baseline --data-root DATA/ --no-sas

Full SymFormer (SymAttention + SPE + STN, right->left):
    python tools/tv_train.py --work-dir RUNS/symformer --data-root DATA/ \
        --attention symattention --pe spe --stn --direction r2l

Ablation cells vary --attention / --pe / --stn / --direction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True,
                    help="checkpoints + logs. Prefer /content/... (ephemeral): checkpoints are "
                         "~300MB and a full run here is only ~15-20min, so Drive is not worth the "
                         "quota. Use --drive-sync to copy the tiny logs (and best.pth) to Drive.")
    ap.add_argument("--data-root", required=True, help="compact dataset root (tbx11k_tb512)")
    ap.add_argument("--train-ann", default="annotations/tb_train_agnostic.json")
    ap.add_argument("--train-img", default="images/train")
    ap.add_argument("--val-ann", default="annotations/tb_val_agnostic.json")
    ap.add_argument("--val-img", default="images/val")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run val AP every N epochs (0 disables). Gives the convergence curve and "
                         "drives best.pth selection.")
    ap.add_argument("--drive-sync", default=None,
                    help="optional Drive dir to copy logs + best.pth into after each eval")
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
                    help="checkpoints are ~300MB; keep few to bound Drive usage")
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke test: stop after N batches")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    from symformer_tb.tv_model import build_model, count_parameters
    from symformer_tb.tv_dataset import build_loader
    from symformer_tb.evaluate import evaluate_model

    torch.manual_seed(args.seed)
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
        _drive_sync(args.work_dir, args.drive_sync)

    print(f"training complete -> {args.work_dir}")
    if best_epoch >= 0:
        print(f"best val AP50 = {best_ap50:.1f} at epoch {best_epoch+1} (best.pth)")
        print("NOTE: report the FINAL-epoch AP as the headline (paper-faithful, unbiased); "
              "best.pth is selected on val, so its AP is optimistically biased.")
    return 0


def _drive_sync(work_dir: str, drive_dir: Optional[str]):
    """Copy the small stuff (logs + best.pth) to Drive; tolerate a full/unavailable Drive."""
    if not drive_dir:
        return
    import shutil
    try:
        os.makedirs(drive_dir, exist_ok=True)
        for name in ("train_log.jsonl", "eval_log.jsonl", "best.pth"):
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
