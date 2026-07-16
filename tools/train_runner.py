#!/usr/bin/env python
"""Minimal mmengine training runner (version-robust; no need to clone the mmdet repo).

Registers the SASFPN neck, loads an mmdet config, and trains. Checkpointing/resume come from the
config's CheckpointHook; pass --resume to continue from the latest checkpoint in --work-dir
(this is what makes runs survive Colab time-outs — point --work-dir at Google Drive).

Usage:
    python tools/train_runner.py configs/symformer_retinanet_r50_fpn_tbx11k_512.py \
        --work-dir /content/drive/MyDrive/tb_runs/symformer \
        --data-root /content/drive/MyDrive/tbx11k_tb512/ --resume
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--data-root", default=None, help="override data_root in the config")
    ap.add_argument("--resume", action="store_true", help="resume from latest ckpt in work-dir")
    ap.add_argument("--max-epochs", type=int, default=None, help="override epochs (e.g. smoke test)")
    ap.add_argument("--batch-size", type=int, default=None, help="override train batch size")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # register the custom neck BEFORE building the runner
    import symformer_tb.mmdet_plugin  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    if args.data_root:
        dr = args.data_root if args.data_root.endswith("/") else args.data_root + "/"
        cfg.data_root = dr
        for loader in ("train_dataloader", "val_dataloader", "test_dataloader"):
            if loader in cfg and "dataset" in cfg[loader]:
                cfg[loader]["dataset"]["data_root"] = dr
        for ev in ("val_evaluator", "test_evaluator"):
            if ev in cfg and "ann_file" in cfg[ev]:
                # keep the same relative annotation path under the new root
                rel = cfg[ev]["ann_file"].split("annotations/")[-1]
                cfg[ev]["ann_file"] = dr + "annotations/" + rel
    if args.max_epochs is not None:
        cfg.train_cfg["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg.train_dataloader["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed)
    if args.resume:
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
