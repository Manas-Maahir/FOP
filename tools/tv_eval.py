#!/usr/bin/env python
"""Evaluate a checkpoint on TB-val: COCO AP and AP50 (category-agnostic TB detection).

This is the metric the whole PoC turns on — the primary claim is that SymFormer's AP50 exceeds the
baseline's. The SAS config is read from the checkpoint, so you don't have to repeat the flags.

    python tools/tv_eval.py --ckpt RUNS/symformer/epoch_24.pth --data-root DATA/

Appends a one-line JSON record to <work-dir>/eval_log.jsonl and prints AP/AP50.
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
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from symformer_tb.tv_model import build_model
    from symformer_tb.tv_dataset import build_loader

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

    results = []
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            outputs = model(images)
            for out, t in zip(outputs, targets):
                img_id = int(t["image_id"])
                boxes = out["boxes"].cpu()
                scores = out["scores"].cpu()
                labels = out["labels"].cpu()
                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = [float(v) for v in b]
                    results.append({
                        "image_id": img_id,
                        "category_id": int(l),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],   # xyxy -> COCO xywh
                        "score": float(s),
                    })

    if not results:
        print("NO DETECTIONS — AP is 0. (Expected for an untrained/1-epoch smoke model.)")
        ap = ap50 = 0.0
    else:
        coco_gt = COCO(ann_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(results)
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
        ap, ap50 = float(E.stats[0]), float(E.stats[1])

    print(f"\n==== RESULT ====\nAP   (IoU .50:.95) = {ap*100:.1f}\nAP50 (IoU .50)     = {ap50*100:.1f}")

    work_dir = os.path.dirname(os.path.abspath(args.ckpt))
    tag = args.tag or os.path.basename(work_dir)
    rec = {"tag": tag, "ckpt": os.path.basename(args.ckpt), "sas_cfg": sas_cfg,
           "AP": round(ap * 100, 2), "AP50": round(ap50 * 100, 2),
           "n_val": len(loader.dataset), "n_dets": len(results)}
    with open(os.path.join(work_dir, "eval_log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("logged ->", os.path.join(work_dir, "eval_log.jsonl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
