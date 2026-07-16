"""Shared COCO evaluation (AP / AP50) for category-agnostic TB detection.

Used by both ``tools/tv_eval.py`` (standalone eval) and ``tools/tv_train.py`` (the per-epoch
validation curve + best-checkpoint tracking), so training and final reporting can never drift
apart in how they measure.
"""

from __future__ import annotations

import contextlib
import io


def predict_coco(model, loader, device):
    """Run inference and return COCO-format detection dicts."""
    import torch

    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            outputs = model(images)
            for out, t in zip(outputs, targets):
                img_id = int(t["image_id"])
                for b, s, l in zip(out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()):
                    x1, y1, x2, y2 = [float(v) for v in b]
                    results.append({
                        "image_id": img_id,
                        "category_id": int(l),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],   # xyxy -> COCO xywh
                        "score": float(s),
                    })
    return results


def score_coco(results, ann_file, quiet: bool = False):
    """Score already-computed detections. Returns (AP, AP50) as percentages."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not results:
        return 0.0, 0.0

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO(ann_file)
        coco_dt = coco_gt.loadRes(results)
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    if not quiet:
        print(buf.getvalue())
    return float(E.stats[0]) * 100.0, float(E.stats[1]) * 100.0


def evaluate_model(model, loader, ann_file, device, quiet: bool = False):
    """Convenience: predict then score. Returns (AP, AP50) as percentages."""
    return score_coco(predict_coco(model, loader, device), ann_file, quiet=quiet)
