"""Shared COCO evaluation (AP / AP50) for category-agnostic TB detection.

Used by the trainer's per-epoch validation, by ``tools/val.py``, and by ``symformer_tb/metrics.py``,
so training and final reporting can never drift apart in how they measure.

Two evaluation modes, matching the paper's §4 benchmark:

* **TB-only** -- score over the ~200 val images that contain TB. This is what the Colab PoC did and
  what [report.md](report.md) §5 reports.
* **All-images** -- score over all 1,800 val images, where the 1,600 non-TB images contribute zero
  ground truth. Every box drawn on them is a false positive, so this mode is strictly harder and is
  the one the stage-2 classifier exists to fix. Choosing the mode is a matter of which annotation
  file you pass; the scoring code is identical.
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional


def predict_coco(model_or_adapter, loader, device, cls_filter: Optional[dict] = None):
    """Run inference over a loader and return COCO-format detection dicts.

    Accepts either a :class:`~symformer_tb.adapters.DetAdapter` or a bare torchvision detector, so
    the same function serves the trainer (which holds an adapter) and any standalone script holding
    a plain model.

    Args:
        cls_filter: optional ``{image_id: predicted_class}`` from the stage-2 head. Images the
            classifier calls non-TB have **all** their detections dropped -- the paper's
            false-positive filter, and the whole point of stage 2 in the all-images mode.
    """
    import torch

    predict = getattr(model_or_adapter, "predict", None)
    model = getattr(model_or_adapter, "model", model_or_adapter)
    model.eval()

    results = []
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            outputs = predict(images) if predict is not None else model(images)
            for out, t in zip(outputs, targets):
                img_id = int(t["image_id"])
                if cls_filter is not None and cls_filter.get(img_id, 2) != 2:
                    continue  # classifier says non-TB -> suppress every detection on this image
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
    """Score already-computed detections. Returns (AP, AP50) as **percentages** (0-100).

    Callers must not multiply by 100 again -- doing exactly that produced the ``AP50 = 7905`` rows
    fixed in commit ``6a68bfc``.
    """
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


def evaluate_model(model_or_adapter, loader, ann_file, device, quiet: bool = False,
                   cls_filter: Optional[dict] = None):
    """Convenience: predict then score. Returns (AP, AP50) as percentages."""
    results = predict_coco(model_or_adapter, loader, device, cls_filter=cls_filter)
    return score_coco(results, ann_file, quiet=quiet)
