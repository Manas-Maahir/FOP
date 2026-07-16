#!/usr/bin/env python
"""Minimal mmengine evaluation runner (COCO AP/AP50 on TB-val).

Loads a config + a checkpoint and runs evaluation only. Use it for the baseline vs SymFormer
comparison, for each ablation cell, and for the optional authors'-checkpoint sanity check.

Usage:
    python tools/test_runner.py configs/symformer_retinanet_r50_fpn_tbx11k_512.py \
        /content/drive/MyDrive/tb_runs/symformer/epoch_24.pth \
        --data-root /content/drive/MyDrive/tbx11k_tb512/
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--work-dir", default="./work_dirs/eval")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    import symformer_tb.mmdet_plugin  # noqa: F401  (register SASFPN)
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    if args.data_root:
        dr = args.data_root if args.data_root.endswith("/") else args.data_root + "/"
        cfg.data_root = dr
        for loader in ("val_dataloader", "test_dataloader"):
            if loader in cfg and "dataset" in cfg[loader]:
                cfg[loader]["dataset"]["data_root"] = dr
        for ev in ("val_evaluator", "test_evaluator"):
            if ev in cfg and "ann_file" in cfg[ev]:
                rel = cfg[ev]["ann_file"].split("annotations/")[-1]
                cfg[ev]["ann_file"] = dr + "annotations/" + rel

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print("EVAL METRICS:", metrics)


if __name__ == "__main__":
    main()
