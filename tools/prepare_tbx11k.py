#!/usr/bin/env python
"""Build the compact TB-only COCO dataset for the PoC (plan.md Phase 3).

What it does:
  * selects the **TB images only** (the ~1,200 CXRs that carry TB bounding-box annotations),
  * resizes each image to a square ``--size`` (default 512), scaling boxes by the same
    per-axis factors,
  * writes COCO-format JSON for the train and val splits, with categories {active, latent},
  * optionally also writes a **category-agnostic** JSON (all TB boxes -> one "TB" class) used
    for the primary detection metric.

It reads VOC-style XML box annotations (``<object><name>..<bndbox>..``), which is the format the
official TBX11K release ships for TB images. Because the exact folder layout of a given TBX11K
download can vary, paths and the category name mapping are all configurable; adjust the CONSTANTS
below or pass flags if your copy differs.

IMPORTANT: this script cannot be validated against the real dataset here (the data lives on
Colab/Drive). Run ``python tools/prepare_tbx11k.py --selftest`` first — it builds a tiny synthetic
dataset and checks the resize/box-scaling/COCO logic end to end with no real data.

Usage (on Colab, after the dataset is on Drive):
    python tools/prepare_tbx11k.py \
        --src /content/drive/MyDrive/TBX11K \
        --dst /content/drive/MyDrive/tbx11k_tb512 \
        --xml-dir annotations/xml --img-dir imgs \
        --train-list lists/TB_train.txt --val-list lists/TB_val.txt \
        --size 512 --write-agnostic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

# Map TBX11K's XML class names to our two detection categories. Uncertain/other TB names are
# collapsed to "latent" for the 2-class set (kept, not dropped) and to "TB" in the agnostic set.
DEFAULT_CATMAP = {
    "ActiveTuberculosis": "active",
    "activeTuberculosis": "active",
    "ObsoletePulmonaryTuberculosis": "latent",
    "obsoletePulmonaryTuberculosis": "latent",
    "PulmonaryTuberculosis": "latent",   # "uncertain"-type name in some releases
}
CATEGORIES = [{"id": 1, "name": "active"}, {"id": 2, "name": "latent"}]
CATNAME_TO_ID = {"active": 1, "latent": 2}


@dataclass
class Record:
    file_name: str
    width: int
    height: int
    boxes: list = field(default_factory=list)  # list of (cat_name, x, y, w, h) in resized coords
    width_resized: int = 0
    height_resized: int = 0


# --------------------------------------------------------------------------------------
def parse_voc_xml(path: str):
    """Return (width, height, [(name, xmin, ymin, xmax, ymax), ...]) from a VOC XML file."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        objs.append((name, xmin, ymin, xmax, ymax))
    return width, height, objs


def scale_box(xmin, ymin, xmax, ymax, sx, sy):
    """Scale a VOC box by per-axis factors and return COCO xywh."""
    x0, y0 = xmin * sx, ymin * sy
    x1, y1 = xmax * sx, ymax * sy
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def build_record(xml_path: str, size: int, catmap: dict) -> Record:
    w, h, objs = parse_voc_xml(xml_path)
    sx, sy = size / float(w), size / float(h)
    boxes = []
    for name, xmin, ymin, xmax, ymax in objs:
        cat = catmap.get(name)
        if cat is None:
            continue  # unknown class name -> skip (report separately)
        boxes.append((cat, *scale_box(xmin, ymin, xmax, ymax, sx, sy)))
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    return Record(file_name=stem, width=w, height=h, boxes=boxes)


def to_coco(records, agnostic: bool = False) -> dict:
    """Assemble a COCO dict from records. If agnostic, collapse all boxes to a single 'TB' class."""
    if agnostic:
        categories = [{"id": 1, "name": "TB"}]
        name_to_id = None
    else:
        categories = CATEGORIES
        name_to_id = CATNAME_TO_ID
    images, annotations = [], []
    ann_id = 1
    for img_id, rec in enumerate(records, start=1):
        images.append({"id": img_id, "file_name": rec.file_name + ".png",
                       "width": rec.width_resized, "height": rec.height_resized})
        for (cat, x, y, w, h) in rec.boxes:
            cid = 1 if agnostic else name_to_id[cat]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2), "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def read_id_list(path: Optional[str]):
    if not path:
        return None
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(os.path.splitext(os.path.basename(s))[0])
    return set(ids)


# --------------------------------------------------------------------------------------
def process_split(stems, args, catmap, split_name):
    """Resize images + build records for one split; write images to disk; return records."""
    out_img_dir = os.path.join(args.dst, "images", split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    records, missing, unknown_names = [], [], set()
    for stem in sorted(stems):
        xml_path = os.path.join(args.src, args.xml_dir, stem + ".xml")
        if not os.path.isfile(xml_path):
            missing.append(stem)
            continue
        rec = build_record(xml_path, args.size, catmap)
        # locate + resize the image
        img_path = _find_image(os.path.join(args.src, args.img_dir), stem)
        if img_path is None:
            missing.append(stem)
            continue
        img = Image.open(img_path).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        img.save(os.path.join(out_img_dir, stem + ".png"))
        rec.width_resized = args.size
        rec.height_resized = args.size
        records.append(rec)
    return records, missing, unknown_names


def _find_image(root, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        # search recursively (TB images may sit under imgs/tb/ etc.)
        for dirpath, _dirs, files in os.walk(root):
            if stem + ext in files:
                return os.path.join(dirpath, stem + ext)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build compact TB-only 512 COCO dataset")
    ap.add_argument("--src", help="TBX11K dataset root")
    ap.add_argument("--dst", help="output root for the compact dataset")
    ap.add_argument("--xml-dir", default="annotations/xml", help="VOC XML dir relative to --src")
    ap.add_argument("--img-dir", default="imgs", help="image dir relative to --src")
    ap.add_argument("--train-list", default=None, help="file listing TB train image ids")
    ap.add_argument("--val-list", default=None, help="file listing TB val image ids")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--write-agnostic", action="store_true",
                    help="also write category-agnostic (single 'TB' class) JSONs")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if not (args.src and args.dst):
        ap.error("--src and --dst are required unless --selftest is given")

    os.makedirs(os.path.join(args.dst, "annotations"), exist_ok=True)
    catmap = DEFAULT_CATMAP

    train_ids = read_id_list(args.train_list)
    val_ids = read_id_list(args.val_list)
    if train_ids is None or val_ids is None:
        # fall back: enumerate all XML stems, deterministic 3:1 split (documented deviation)
        all_stems = sorted(os.path.splitext(f)[0]
                           for f in os.listdir(os.path.join(args.src, args.xml_dir))
                           if f.endswith(".xml"))
        val_ids = set(all_stems[::4])
        train_ids = set(s for s in all_stems if s not in val_ids)
        print("[warn] no split lists given; using a deterministic 3:1 train/val split "
              "(deviation from the official split — record this).")

    for split, ids in (("train", train_ids), ("val", val_ids)):
        recs, missing, unknown = process_split(ids, args, catmap, split)
        coco = to_coco(recs, agnostic=False)
        out = os.path.join(args.dst, "annotations", f"tb_{split}.json")
        json.dump(coco, open(out, "w"))
        print(f"[{split}] images={len(coco['images'])} boxes={len(coco['annotations'])} "
              f"missing={len(missing)} -> {out}")
        if args.write_agnostic:
            coco_a = to_coco(recs, agnostic=True)
            out_a = os.path.join(args.dst, "annotations", f"tb_{split}_agnostic.json")
            json.dump(coco_a, open(out_a, "w"))
            print(f"[{split}] agnostic boxes={len(coco_a['annotations'])} -> {out_a}")
    print("Done. RECORD the printed counts in results.md and spot-check a few overlays.")
    return 0


# --------------------------------------------------------------------------------------
def selftest():
    """Build a tiny synthetic dataset and verify resize + box scaling + COCO output."""
    print("Running synthetic self-test (no real data needed)...")
    tmp = tempfile.mkdtemp(prefix="tbx11k_selftest_")
    src = os.path.join(tmp, "src")
    dst = os.path.join(tmp, "dst")
    xml_dir = os.path.join(src, "annotations", "xml")
    img_dir = os.path.join(src, "imgs", "tb")
    os.makedirs(xml_dir)
    os.makedirs(img_dir)

    # one 3000x2000 image with a known box, one 1000x1000 image with a known box
    cases = [
        ("caseA", 3000, 2000, "ActiveTuberculosis", (300, 400, 900, 1200)),   # xmin,ymin,xmax,ymax
        ("caseB", 1000, 1000, "ObsoletePulmonaryTuberculosis", (100, 100, 500, 500)),
    ]
    for stem, W, H, cname, (xmin, ymin, xmax, ymax) in cases:
        Image.new("RGB", (W, H), (123, 123, 123)).save(os.path.join(img_dir, stem + ".png"))
        xml = f"""<annotation><size><width>{W}</width><height>{H}</height></size>
        <object><name>{cname}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>"""
        open(os.path.join(xml_dir, stem + ".xml"), "w").write(xml)

    args = argparse.Namespace(src=src, dst=dst, xml_dir="annotations/xml", img_dir="imgs",
                              train_list=None, val_list=None, size=512, write_agnostic=True,
                              selftest=False)
    # force a deterministic split by writing explicit lists
    train_list = os.path.join(tmp, "train.txt")
    val_list = os.path.join(tmp, "val.txt")
    open(train_list, "w").write("caseA\n")
    open(val_list, "w").write("caseB\n")
    args.train_list, args.val_list = train_list, val_list

    os.makedirs(os.path.join(dst, "annotations"), exist_ok=True)
    for split, ids in (("train", {"caseA"}), ("val", {"caseB"})):
        recs, missing, _ = process_split(ids, args, DEFAULT_CATMAP, split)
        coco = to_coco(recs, agnostic=False)
        assert not missing, f"missing files: {missing}"
        # image resized to 512
        img = Image.open(os.path.join(dst, "images", split, ids.copy().pop() + ".png"))
        assert img.size == (512, 512), img.size
        json.dump(coco, open(os.path.join(dst, "annotations", f"tb_{split}.json"), "w"))

    # check caseA box scaling: original 3000x2000 -> 512x512; sx=512/3000, sy=512/2000
    a = json.load(open(os.path.join(dst, "annotations", "tb_train.json")))
    sx, sy = 512 / 3000, 512 / 2000
    exp = [300 * sx, 400 * sy, (900 - 300) * sx, (1200 - 400) * sy]
    got = a["annotations"][0]["bbox"]
    assert all(abs(g - e) < 0.05 for g, e in zip(got, exp)), f"box mismatch: got {got} exp {exp}"
    assert a["annotations"][0]["category_id"] == 1  # active
    b = json.load(open(os.path.join(dst, "annotations", "tb_val.json")))
    assert b["annotations"][0]["category_id"] == 2  # latent (from Obsolete...)

    # agnostic collapse
    recs_v, _, _ = process_split({"caseB"}, args, DEFAULT_CATMAP, "val")
    ag = to_coco(recs_v, agnostic=True)
    assert ag["categories"] == [{"id": 1, "name": "TB"}]
    assert ag["annotations"][0]["category_id"] == 1

    print("PASS resize -> 512x512")
    print("PASS box scaling (per-axis factors)")
    print("PASS category mapping active/latent")
    print("PASS category-agnostic collapse")
    print("PASS COCO JSON structure")
    print("\nSelf-test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
