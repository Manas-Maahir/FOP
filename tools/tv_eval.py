#!/usr/bin/env python
"""Evaluate a checkpoint on TB-val: COCO AP and AP50 (category-agnostic TB detection).

This is the metric the whole PoC turns on — the primary claim is that SymFormer's AP50 exceeds the
baseline's. The SAS config is read from the checkpoint, so you don't have to repeat the flags.

    python tools/tv_eval.py --ckpt /content/work/symformer/epoch_24.pth --data-root DATA/ \
        --drive-sync /content/drive/MyDrive/tb_runs/symformer

Appends a one-line JSON record to <ckpt-dir>/eval_log.jsonl and prints AP/AP50. Since checkpoints
live on ephemeral /content, pass --drive-sync to copy that log to Drive — otherwise the result dies
with the session.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--val-ann", default="annotations/tb_val_agnostic.json")
    ap.add_argument("--val-img", default="images/val")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--tag", default=None, help="name for the results log (defaults to ckpt dir)")
    ap.add_argument("--drive-sync", default=None,
                    help="Drive dir to copy eval_log.jsonl into. Checkpoints live on ephemeral "
                         "/content, so without this the result is lost when the session ends.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    from symformer_tb.tv_model import build_model
    from symformer_tb.tv_dataset import build_loader
    from symformer_tb.evaluate import predict_coco, score_coco

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sas_cfg = ckpt.get("sas_cfg")
    print("SAS config from checkpoint:", sas_cfg if sas_cfg else "(none — baseline)")

    model = build_model(sas=sas_cfg, image_size=args.image_size,
                        pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = args.data_root.rstrip("/")
    ann_file = os.path.join(root, args.val_ann)
    loader = build_loader(ann_file, os.path.join(root, args.val_img),
                          batch_size=args.batch_size, train=False,
                          num_workers=args.num_workers)
    print(f"val images: {len(loader.dataset)}")

    results = predict_coco(model, loader, device)          # infer once
    if not results:
        print("NO DETECTIONS — AP is 0. (Expected for an untrained/1-epoch smoke model.)")
    ap, ap50 = score_coco(results, ann_file)               # ...then score
    n_dets = len(results)

    print(f"\n==== RESULT ====\nAP   (IoU .50:.95) = {ap:.1f}\nAP50 (IoU .50)     = {ap50:.1f}")

    work_dir = os.path.dirname(os.path.abspath(args.ckpt))
    tag = args.tag or os.path.basename(work_dir)
    # score_coco already returns percentages (0-100); do NOT scale again.
    rec = {"tag": tag, "ckpt": os.path.basename(args.ckpt), "sas_cfg": sas_cfg,
           "AP": round(ap, 2), "AP50": round(ap50, 2),
           "n_val": len(loader.dataset), "n_dets": len(results)}
    line = json.dumps(rec) + "\n"
    with open(os.path.join(work_dir, "eval_log.jsonl"), "a") as f:
        f.write(line)
    print("logged ->", os.path.join(work_dir, "eval_log.jsonl"))

    # The checkpoint dir is on ephemeral /content, so persist the result itself to Drive. Append
    # rather than copy: a re-run in a later session has a fresh (empty) /content log, and copying
    # would overwrite the earlier results with just this one.
    if args.drive_sync:
        try:
            os.makedirs(args.drive_sync, exist_ok=True)
            with open(os.path.join(args.drive_sync, "eval_log.jsonl"), "a") as f:
                f.write(line)
            print("synced ->", os.path.join(args.drive_sync, "eval_log.jsonl"))
        except OSError as e:
            # e.g. "Google Drive storage quota has been exceeded" — the number is already printed
            print(f"[warn] drive sync failed ({e}); result remains in {work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
