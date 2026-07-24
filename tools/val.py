#!/usr/bin/env python
"""Evaluate a checkpoint: COCO AP / AP50 for category-agnostic TB detection.

Replaces ``tools/tv_eval.py``. The SAS config and stack are read from the checkpoint, so the flags
never have to be repeated.

Two modes, matching the paper's §4 benchmark:

    --mode tb-only   the ~200 val images that contain TB. What the Colab PoC reported.
    --mode all       all 1,800 val images. The 1,600 non-TB images have zero ground truth, so every
                     box drawn on them is a false positive -- strictly harder, and the mode the
                     stage-2 classifier exists to fix.

    python tools/val.py --weights runs/detect/train/weights/best.pt --data-root data/tbx11k_512
    python tools/val.py --weights ... --mode all --cls-ckpt runs/classify/train/weights/best.pt

Appends a JSON record to ``<run>/eval_log.jsonl`` and prints the result table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODES = {
    # mode      -> (annotation file, keep images with no boxes)
    "tb-only": ("annotations/tb_val_agnostic.json", False),
    "all": ("annotations/all_val_agnostic.json", True),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, help="path to best.pt / last.pt")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--mode", default="tb-only", choices=list(MODES),
                    help="which val population to score over")
    ap.add_argument("--val-ann", default=None, help="override the annotation file for --mode")
    ap.add_argument("--val-img", default="images/val")
    ap.add_argument("--cls-ckpt", default=None,
                    help="stage-2 classifier; images it calls non-TB get their detections "
                         "suppressed (the paper's false-positive filter)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None, help="name for the results log (defaults to the run dir)")
    ap.add_argument("--save-json", default=None, help="also write raw detections here")
    ap.add_argument("--plots", action="store_true", default=True)
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    return ap.parse_args(argv)


def load_cls_filter(ckpt_path: str, data_root: Path, val_img: str, device,
                    batch_size: int, num_workers: int, image_size: int) -> dict:
    """Run the stage-2 head over the val split -> ``{image_id: predicted_class_index}``."""
    import torch

    from symformer_tb.adapters import build_adapter
    from symformer_tb.cls_head import ClassifierModel
    from symformer_tb.tv_dataset import build_cls_loader

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    adapter = build_adapter(stack=ckpt.get("stack", "torchvision"), sas=ckpt.get("sas_cfg"),
                            image_size=image_size, pretrained_backbone=False)
    model = ClassifierModel(adapter, tap=args.get("tap", "2")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ann = data_root / "annotations" / "cls_val.json"
    loader = build_cls_loader(str(ann), str(data_root / val_img), batch_size=batch_size,
                              train=False, num_workers=num_workers)

    out = {}
    with torch.no_grad():
        for images, _labels, ids in loader:
            logits = model(images.to(device))
            pred = logits.argmax(dim=1).cpu()
            for i, p in zip(ids.tolist(), pred.tolist()):
                out[int(i)] = int(p)
    n_tb = sum(1 for v in out.values() if v == 2)
    print(f"  classifier: {n_tb}/{len(out)} images predicted TB "
          f"({len(out) - n_tb} will have their detections suppressed)")
    return out


def main(argv=None):
    args = parse_args(argv)

    import torch

    from symformer_tb.adapters import build_adapter
    from symformer_tb.evaluate import predict_coco
    from symformer_tb.metrics import evaluate_detections
    from symformer_tb.trainer import VAL_HEADER, colorstr
    from symformer_tb.tv_dataset import build_loader
    from symformer_tb import plotting

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    sas_cfg = ckpt.get("sas_cfg")
    stack = ckpt.get("stack", "torchvision")

    print(colorstr("bold", "\nSymFormer / TBX11K -- evaluation"))
    print(f"  weights : {args.weights}")
    print(f"  stack   : {stack}")
    print(f"  SAS     : {sas_cfg if sas_cfg else '(none -- baseline)'}")
    print(f"  mode    : {args.mode}")

    adapter = build_adapter(stack=stack, sas=sas_cfg, image_size=args.image_size,
                            pretrained_backbone=False)
    # prefer the EMA weights: they are what best.pt was scored on during training
    adapter.model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    adapter.model.to(device).eval()

    root = Path(args.data_root)
    default_ann, keep_empty = MODES[args.mode]
    ann_file = str(root / (args.val_ann or default_ann))
    if not Path(ann_file).is_file():
        print(f"\nERROR: annotation file not found: {ann_file}")
        if args.mode == "all":
            print("  --mode all needs the full dataset. Rebuild it with:")
            print("      python tools/prepare_tbx11k.py --scope all --src <raw> --dst <out>")
        return 2

    loader = build_loader(ann_file, str(root / args.val_img), batch_size=args.batch_size,
                          train=False, num_workers=args.num_workers, keep_empty=keep_empty)
    print(f"  images  : {len(loader.dataset)}")

    cls_filter = None
    if args.cls_ckpt:
        cls_filter = load_cls_filter(args.cls_ckpt, root, args.val_img, device,
                                     args.batch_size, args.num_workers, args.image_size)

    results = predict_coco(adapter, loader, device, cls_filter=cls_filter)
    metrics = evaluate_detections(results, ann_file, quiet=True)

    print()
    print(colorstr(VAL_HEADER))
    print(("%22s" + "%11s" * 2 + "%11.4g" * 4) % (
        "all", metrics["n_images"], metrics["n_instances"],
        metrics["precision"] * 100, metrics["recall"] * 100, metrics["AP50"], metrics["AP"]))
    print(f"\n  AP   (IoU .50:.95) = {metrics['AP']:.1f}")
    print(f"  AP50 (IoU .50)     = {metrics['AP50']:.1f}")
    print(f"  TP/FP/FN @ conf {metrics['conf']:.2f}: "
          f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    if not results:
        print(colorstr("yellow", "  NO DETECTIONS -- AP is 0. Expected for an untrained/1-epoch "
                                 "smoke model, not for a real run."))

    run_dir = Path(args.weights).resolve().parent.parent
    tag = args.tag or run_dir.name
    if args.plots:
        plotting.write_val_figures(metrics, run_dir)

    record = {
        "tag": tag, "weights": str(args.weights), "stack": stack, "sas_cfg": sas_cfg,
        "mode": args.mode, "cls_filter": bool(cls_filter),
        "AP": round(metrics["AP"], 2), "AP50": round(metrics["AP50"], 2),
        "precision": round(metrics["precision"] * 100, 2),
        "recall": round(metrics["recall"] * 100, 2),
        "f1": round(metrics["f1"] * 100, 2),
        "n_images": metrics["n_images"], "n_instances": metrics["n_instances"],
        "n_dets": metrics["n_dets"],
    }
    log_path = run_dir / "eval_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nlogged -> {log_path}")

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(results))
        print(f"detections -> {args.save_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
