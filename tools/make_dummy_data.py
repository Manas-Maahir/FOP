#!/usr/bin/env python
"""Generate a tiny synthetic dataset shaped exactly like the real one, for the smoke test.

Why this exists: the old runbook made you download tens of gigabytes before you could discover that
your environment was broken. This produces a dataset with the same directory layout, the same COCO
schema and the same file names as ``tools/prepare_tbx11k.py --scope all``, in about two seconds, so
the entire chain -- model build, training loop, checkpointing, resume, COCO scoring, plots -- can be
proven on a fresh PC *before* anything large is fetched.

The images are not chest X-rays and are not meant to be. They are coarse synthetic lungs: a dark
background with two bright elliptical fields, plus bright blobs standing in for lesions. Boxes are
generated from the blob positions, so AP is a real (if easy) measurement rather than noise, and a
pipeline that silently mismatches boxes to images will still show up as AP ~ 0.

    python tools/make_dummy_data.py --dst data/dummy512
    python tools/make_dummy_data.py --dst data/dummy512 --n-train 16 --n-val 8 --size 256
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Same three classes and the same index order as the real prep tool.
CLASSES = ("healthy", "sick_non_tb", "tb")


def draw_chest(size: int, rng: random.Random, n_lesions: int | None = None
               ) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """One synthetic CXR-ish image plus its lesion boxes in xywh.

    Roughly bilaterally symmetric on purpose: SymFormer's premise is that a lesion breaks left/right
    symmetry, so a smoke fixture with symmetric anatomy and asymmetric lesions exercises the SAS
    block's mirror path rather than feeding it noise.

    ``n_lesions=0`` draws the anatomy with no lesions -- the negatives, which must look like
    plausible chests rather than blank frames or the detector learns "bright blob = anything".
    """
    img = Image.new("RGB", (size, size), (12, 12, 16))
    d = ImageDraw.Draw(img)

    # thorax
    d.ellipse([size * 0.08, size * 0.10, size * 0.92, size * 0.95], fill=(52, 52, 58))
    # two lung fields, mirrored about the vertical centerline
    for x0, x1 in ((0.14, 0.46), (0.54, 0.86)):
        d.ellipse([size * x0, size * 0.18, size * x1, size * 0.80], fill=(120, 120, 128))
    # spine
    d.rectangle([size * 0.47, size * 0.15, size * 0.53, size * 0.90], fill=(70, 70, 76))

    boxes: list[tuple[float, float, float, float]] = []
    count = rng.randint(1, 3) if n_lesions is None else n_lesions
    for _ in range(count):
        r = rng.uniform(size * 0.035, size * 0.085)
        # keep lesions inside one lung field so the box always sits on plausible anatomy
        left = rng.random() < 0.5
        cx = rng.uniform(size * 0.17, size * 0.43) if left else rng.uniform(size * 0.57, size * 0.83)
        cy = rng.uniform(size * 0.24, size * 0.74)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 235, 240))
        boxes.append((max(cx - r, 0.0), max(cy - r, 0.0), 2 * r, 2 * r))

    # mild noise so the backbone does not see a perfectly flat texture
    px = img.load()
    for _ in range(size * 8):
        x, y = rng.randrange(size), rng.randrange(size)
        v = px[x, y]
        j = rng.randint(-14, 14)
        px[x, y] = tuple(max(0, min(255, c + j)) for c in v)
    return img, boxes


def build_split(dst: Path, split: str, n_tb: int, n_neg: int, size: int, rng: random.Random):
    """Write one split's images and return (detection records, classification records)."""
    img_dir = dst / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    det_images, det_anns, cls_records = [], [], []
    ann_id = 1
    image_id = 1

    for i in range(n_tb):
        stem = f"{split}_tb{i:04d}"
        img, boxes = draw_chest(size, rng)
        img.save(img_dir / f"{stem}.png")
        det_images.append({"id": image_id, "file_name": f"{stem}.png",
                           "width": size, "height": size})
        for (x, y, w, h) in boxes:
            det_anns.append({"id": ann_id, "image_id": image_id, "category_id": 1,
                             "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                             "area": round(w * h, 2), "iscrowd": 0})
            ann_id += 1
        cls_records.append({"image_id": image_id, "file_name": f"{stem}.png", "class": "tb"})
        image_id += 1

    # negatives: same anatomy, no lesions. These are what make the all-images mode meaningful.
    for i in range(n_neg):
        cls = "healthy" if i % 2 == 0 else "sick_non_tb"
        stem = f"{split}_{cls}{i:04d}"
        img, _ = draw_chest(size, rng, n_lesions=0)
        img.save(img_dir / f"{stem}.png")
        det_images.append({"id": image_id, "file_name": f"{stem}.png",
                           "width": size, "height": size})
        cls_records.append({"image_id": image_id, "file_name": f"{stem}.png", "class": cls})
        image_id += 1

    return det_images, det_anns, cls_records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dst", default="data/dummy512", help="output dataset root")
    ap.add_argument("--n-train", type=int, default=24, help="TB images in train")
    ap.add_argument("--n-val", type=int, default=12, help="TB images in val")
    ap.add_argument("--n-neg-train", type=int, default=12, help="non-TB images in train")
    ap.add_argument("--n-neg-val", type=int, default=12, help="non-TB images in val")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild even if it already exists")
    args = ap.parse_args(argv)

    dst = Path(args.dst)
    ann_dir = dst / "annotations"
    marker = ann_dir / "tb_val_agnostic.json"
    if marker.is_file() and not args.force:
        print(f"dummy dataset already present at {dst} (use --force to rebuild)")
        return 0

    rng = random.Random(args.seed)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split, n_tb, n_neg in (("train", args.n_train, args.n_neg_train),
                               ("val", args.n_val, args.n_neg_val)):
        images, anns, cls_records = build_split(dst, split, n_tb, n_neg, args.size, rng)
        tb_ids = {a["image_id"] for a in anns}
        tb_images = [im for im in images if im["id"] in tb_ids]

        # TB-only detection JSON (stage-1 training/eval) -- mirrors tb_{split}_agnostic.json
        (ann_dir / f"tb_{split}_agnostic.json").write_text(json.dumps({
            "images": tb_images, "annotations": anns,
            "categories": [{"id": 1, "name": "TB"}],
        }))
        # all-images detection JSON -- negatives carry zero annotations
        (ann_dir / f"all_{split}_agnostic.json").write_text(json.dumps({
            "images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "TB"}],
        }))
        # stage-2 classification labels
        (ann_dir / f"cls_{split}.json").write_text(json.dumps({
            "images": cls_records, "classes": list(CLASSES),
        }))
        print(f"[{split}] {len(tb_images)} TB + {len(images) - len(tb_images)} non-TB "
              f"= {len(images)} images, {len(anns)} boxes")

    print(f"\nSynthetic dataset written to {dst}")
    print("It has the same layout and schema as the real one, so every downstream tool works "
          "against it unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
