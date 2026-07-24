#!/usr/bin/env python
"""Train RetinaNet (baseline) or SymFormer (RetinaNet + SAS) on the TB 512 COCO set.

Replaces ``tools/tv_train.py``. Same paper stage-1 recipe -- SGD, batch 8, 24 epochs, 512x512,
random horizontal flip, fixed seed -- now with the run directory, progress bar, metric table, plots
and resume behaviour in ``symformer_tb/trainer.py``, and with either detection stack behind
``--stack``.

    # baseline (paper Table 8 "No attention / No PE")
    python tools/train.py --data-root data/tbx11k_512 --no-sas

    # full SymFormer (SymAttention + SPE + STN, right->left)
    python tools/train.py --data-root data/tbx11k_512 \
        --attention symattention --pe spe --stn --direction r2l

    # the same thing on the paper's own framework
    python tools/train.py --data-root data/tbx11k_512 --stack mmdet ...

    # continue the most recent run after a Ctrl-C or a closed lid
    python tools/train.py --resume

Ablation cells vary --attention / --pe / --stn / --direction; see tools/ablate.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -- data ---------------------------------------------------------------------------
    g = ap.add_argument_group("data")
    g.add_argument("--data-root", required=False, help="compact dataset root")
    g.add_argument("--train-ann", default="annotations/tb_train_agnostic.json")
    g.add_argument("--train-img", default="images/train")
    g.add_argument("--val-ann", default="annotations/tb_val_agnostic.json")
    g.add_argument("--val-img", default="images/val")
    g.add_argument("--image-size", type=int, default=512)
    g.add_argument("--num-workers", type=int, default=4)

    # -- model --------------------------------------------------------------------------
    g = ap.add_argument_group("model")
    g.add_argument("--stack", default="torchvision", choices=["torchvision", "mmdet"],
                   help="detection framework. mmdet is the paper's own; torchvision is the "
                        "maintained one the Colab PoC used")
    g.add_argument("--config", default=None,
                   help="mmdet config path (only used with --stack mmdet)")
    g.add_argument("--no-sas", action="store_true", help="plain RetinaNet baseline")
    g.add_argument("--attention", default="symattention", choices=["vanilla", "symattention"])
    g.add_argument("--pe", default="spe", choices=["none", "ape", "rpe", "spe"])
    g.add_argument("--stn", action="store_true", help="enable the STN inside SPE")
    g.add_argument("--no-stn", dest="stn", action="store_false")
    ap.set_defaults(stn=True)
    g.add_argument("--direction", default="r2l", choices=["r2l", "l2r"])
    g.add_argument("--num-heads", type=int, default=8)
    g.add_argument("--num-points", type=int, default=4)

    # -- schedule (paper stage-1) ---------------------------------------------------------
    g = ap.add_argument_group("schedule")
    g.add_argument("--epochs", type=int, default=24)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--lr", type=float, default=0.005,
                   help="0.01 is tuned for batch 16; linear-scale with the batch")
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--milestones", type=int, nargs="+", default=[16, 22])
    g.add_argument("--warmup-iters", type=int, default=500)
    g.add_argument("--grad-clip", type=float, default=10.0,
                   help="clip gradient L2 norm before each step (0 disables). torchvision's "
                        "RetinaNet can spike bbox_regression to ~1e34 as the warmup LR peaks and "
                        "then NaN out; clipping is the standard guard")
    g.add_argument("--seed", type=int, default=0)

    # -- run ----------------------------------------------------------------------------
    g = ap.add_argument_group("run")
    g.add_argument("--project", default="runs/detect", help="parent directory for runs")
    g.add_argument("--name", default="train", help="run name; auto-incremented if it exists")
    g.add_argument("--exist-ok", action="store_true", help="reuse --name instead of incrementing")
    g.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="resume: bare for the most recent run, or a path to a last.pt")
    g.add_argument("--eval-every", type=int, default=1, help="validate every N epochs (0 disables)")
    g.add_argument("--save-period", type=int, default=0, help="also snapshot every N epochs")
    g.add_argument("--fitness", default="ultralytics", choices=["ultralytics", "ap50", "ap"],
                   help="what best.pt is selected on. 'ap50' matches the Colab PoC")
    g.add_argument("--amp", action="store_true", default=True, help="mixed precision (default on)")
    g.add_argument("--no-amp", dest="amp", action="store_false")
    g.add_argument("--ema", action="store_true", default=True, help="weight EMA (default on)")
    g.add_argument("--no-ema", dest="ema", action="store_false")
    g.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True,
                   help="skip the ImageNet-pretrained backbone (offline / smoke tests)")
    g.add_argument("--device", default=None, help="cuda, cuda:0 or cpu (default: auto)")
    g.add_argument("--limit-batches", type=int, default=0, help="smoke test: stop after N batches")

    args = ap.parse_args(argv)
    if not args.resume and not args.data_root:
        ap.error("--data-root is required (unless --resume)")
    return args


def main(argv=None):
    args = parse_args(argv)

    import torch

    from symformer_tb.adapters import build_adapter, mmdet_available
    from symformer_tb.trainer import Trainer, colorstr, increment_path, latest_run
    from symformer_tb.tv_dataset import build_loader
    from symformer_tb import plotting

    # -- resume: recover the original arguments so the run continues, not restarts ---------
    resume_ckpt = None
    if args.resume:
        resume_ckpt = latest_run(args.project) if args.resume == "latest" else Path(args.resume)
        if resume_ckpt is None or not Path(resume_ckpt).is_file():
            print(f"ERROR: nothing to resume (looked for {resume_ckpt or args.project + '/*/weights/last.pt'})")
            return 2
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        saved = ckpt.get("args", {})
        keep = {"resume", "device", "num_workers", "epochs"}   # allow these to be overridden
        for k, v in saved.items():
            if k not in keep and hasattr(args, k):
                setattr(args, k, v)
        run_dir = Path(resume_ckpt).parent.parent
        print(colorstr("bold", f"Resuming {resume_ckpt}"))
    else:
        run_dir = Path(args.project) / args.name
        if not args.exist_ok:
            run_dir = increment_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stack == "mmdet" and not mmdet_available():
        print(colorstr("yellow", "bold",
                       "--stack mmdet requested but mmdet is not installed in this environment."))
        print("Run `python scripts/setup_env.py` (mmdet is stage 3), or use --stack torchvision.")
        return 3

    sas_cfg = None
    if not args.no_sas:
        sas_cfg = dict(attention=args.attention, pe=args.pe, use_stn=args.stn,
                       direction=args.direction, num_heads=args.num_heads,
                       num_points=args.num_points)

    print(colorstr("bold", "\nSymFormer / TBX11K -- stage-1 detection"))
    print(f"  stack   : {args.stack}")
    print(f"  device  : {device}"
          f"{'  (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"  SAS     : {sas_cfg if sas_cfg else '(none -- plain RetinaNet baseline)'}")
    print(f"  seed    : {args.seed}")

    adapter = build_adapter(stack=args.stack, sas=sas_cfg, image_size=args.image_size,
                            config=args.config, pretrained_backbone=args.pretrained)
    adapter.model.to(device)

    n_total = sum(p.numel() for p in adapter.model.parameters())
    n_sas = 0
    sas_mod = getattr(getattr(adapter.model, "backbone", None), "sas", None)
    if sas_mod is not None:
        n_sas = sum(p.numel() for p in sas_mod.parameters())
    print(f"  params  : {n_total:,}" + (f"  (SAS {n_sas:,} = {100 * n_sas / n_total:.2f}%)"
                                        if n_sas else ""))

    root = Path(args.data_root)
    train_loader = build_loader(str(root / args.train_ann), str(root / args.train_img),
                                batch_size=args.batch_size, train=True,
                                num_workers=args.num_workers, seed=args.seed)
    print(f"  train   : {len(train_loader.dataset)} images, {len(train_loader)} batches/epoch")

    val_loader, val_ann = None, None
    if args.eval_every > 0:
        candidate = root / args.val_ann
        if candidate.is_file():
            val_ann = str(candidate)
            val_loader = build_loader(val_ann, str(root / args.val_img),
                                      batch_size=max(2, args.batch_size // 2), train=False,
                                      num_workers=args.num_workers,
                                      keep_empty="all_" in Path(args.val_ann).name)
            print(f"  val     : {len(val_loader.dataset)} images (AP every {args.eval_every} ep)")
        else:
            print(f"  [warn] no val annotations at {candidate}; per-epoch eval disabled")

    plotting.plot_labels(train_loader.dataset.all_boxes_xywh(), args.image_size,
                         run_dir / "labels.jpg")

    trainer = Trainer(adapter, train_loader, val_loader, val_ann, args, run_dir, device)
    if resume_ckpt is not None:
        trainer.resume_from(resume_ckpt)
        if trainer.start_epoch >= args.epochs:
            print(f"already trained {trainer.start_epoch}/{args.epochs} epochs -- nothing to do.")
            return 0

    result = trainer.train()
    return 0 if result.get("status") in ("complete", "interrupted") else 1


if __name__ == "__main__":
    sys.exit(main())
