#!/usr/bin/env python
"""Build the compact TB-only COCO dataset for the PoC (plan.md Phases 2-3).

What it does:
  * selects the **TB images only** (the ~1,200 CXRs that carry TB bounding-box annotations),
  * resizes each image to a square ``--size`` (default 512), scaling boxes by the same per-axis
    factors,
  * writes COCO-format JSON for the train and val splits, categories {active, latent},
  * optionally also writes a **category-agnostic** JSON (all TB boxes -> one "TB" class), which is
    what the PoC's primary AP/AP50 metric uses.

The official TBX11K README documents download links but NOT the archive's folder layout, so this
script **auto-discovers** it: run ``--inspect`` first to see what's actually in your copy.

    python tools/prepare_tbx11k.py --inspect --src /content/TBX11K
    python tools/prepare_tbx11k.py --src /content/TBX11K --dst /content/drive/MyDrive/tbx11k_tb512 \
        --size 512 --write-agnostic

``--selftest`` builds a tiny synthetic dataset and verifies the resize/box-scaling/COCO logic with
no real data — run it before touching the real thing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# Map TBX11K's XML class names to our two detection categories. Run --inspect to see the class
# names your copy actually uses, and extend this if they differ.
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
    boxes: list = field(default_factory=list)  # (cat_name, x, y, w, h) in resized coords
    width_resized: int = 0
    height_resized: int = 0


# --------------------------------------------------------------------------------------
# Layout discovery
# --------------------------------------------------------------------------------------
def discover_layout(src: str) -> dict:
    """Walk `src` once and report where the XMLs, images and list files actually live."""
    xml_by_dir, img_by_dir = Counter(), Counter()
    list_files, xml_paths = [], []
    img_index = {}  # stem -> path (built in the same single walk)

    for dirpath, _dirs, files in os.walk(src):
        for fn in files:
            p = os.path.join(dirpath, fn)
            low = fn.lower()
            if low.endswith(".xml"):
                xml_by_dir[dirpath] += 1
                xml_paths.append(p)
            elif low.endswith(IMG_EXTS):
                img_by_dir[dirpath] += 1
                img_index.setdefault(os.path.splitext(fn)[0], p)
            elif low.endswith(".txt"):
                list_files.append(p)

    # sample some XMLs to learn the class names in use
    class_names = Counter()
    for p in xml_paths[:300]:
        try:
            root = ET.parse(p).getroot()
            for obj in root.findall("object"):
                name = obj.find("name")
                if name is not None and name.text:
                    class_names[name.text.strip()] += 1
        except Exception:
            continue

    return {
        "src": src,
        "exists": os.path.isdir(src),
        "xml_dirs": xml_by_dir.most_common(),
        "img_dirs": img_by_dir.most_common(),
        "list_files": list_files,
        "n_xml": len(xml_paths),
        "n_img": len(img_index),
        "class_names": class_names.most_common(),
        "img_index": img_index,
    }


def print_report(rep: dict) -> None:
    print("=" * 72)
    print("TBX11K layout report for:", rep["src"])
    print("=" * 72)
    if not rep["exists"]:
        print("!! DIRECTORY DOES NOT EXIST — download the dataset first.")
        return
    print(f"\nXML annotation files: {rep['n_xml']}")
    for d, n in rep["xml_dirs"][:8]:
        print(f"   {n:6d}  {d}")
    print(f"\nImage files: {rep['n_img']}")
    for d, n in rep["img_dirs"][:8]:
        print(f"   {n:6d}  {d}")
    print(f"\nCandidate list (.txt) files: {len(rep['list_files'])}")
    for p in rep["list_files"][:12]:
        print("   ", p)
    print("\nXML class names found (sampled):")
    if rep["class_names"]:
        for c, n in rep["class_names"]:
            mapped = DEFAULT_CATMAP.get(c, "!! UNMAPPED — add it to DEFAULT_CATMAP")
            print(f"   {n:6d}  {c!r:42s} -> {mapped}")
    else:
        print("    (none found — is this the right archive?)")
    print("\nExpected per paper Table 2: ~1,200 TB images total; TB train ~600, val ~200.")
    print("=" * 72)


def pick_xml_dir(rep: dict) -> Optional[str]:
    return rep["xml_dirs"][0][0] if rep["xml_dirs"] else None


def guess_list(rep: dict, *keywords) -> Optional[str]:
    """Find a .txt list file whose name contains all keywords (case-insensitive)."""
    for p in rep["list_files"]:
        low = os.path.basename(p).lower()
        if all(k in low for k in keywords):
            return p
    return None


# --------------------------------------------------------------------------------------
# Parsing / conversion
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
        objs.append((name,
                     float(bb.find("xmin").text), float(bb.find("ymin").text),
                     float(bb.find("xmax").text), float(bb.find("ymax").text)))
    return width, height, objs


def scale_box(xmin, ymin, xmax, ymax, sx, sy):
    x0, y0, x1, y1 = xmin * sx, ymin * sy, xmax * sx, ymax * sy
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def build_record(xml_path: str, size: int, catmap: dict):
    w, h, objs = parse_voc_xml(xml_path)
    sx, sy = size / float(w), size / float(h)
    boxes, unknown = [], set()
    for name, xmin, ymin, xmax, ymax in objs:
        cat = catmap.get(name)
        if cat is None:
            unknown.add(name)
            continue
        boxes.append((cat, *scale_box(xmin, ymin, xmax, ymax, sx, sy)))
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    return Record(file_name=stem, width=w, height=h, boxes=boxes), unknown


def to_coco(records, agnostic: bool = False) -> dict:
    categories = [{"id": 1, "name": "TB"}] if agnostic else CATEGORIES
    images, annotations = [], []
    ann_id = 1
    for img_id, rec in enumerate(records, start=1):
        images.append({"id": img_id, "file_name": rec.file_name + ".png",
                       "width": rec.width_resized, "height": rec.height_resized})
        for (cat, x, y, w, h) in rec.boxes:
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": 1 if agnostic else CATNAME_TO_ID[cat],
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2), "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def read_id_list(path: Optional[str]):
    if not path or not os.path.isfile(path):
        return None
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(os.path.splitext(os.path.basename(s.split()[0]))[0])
    return set(ids)


def process_split(stems, xml_dir, img_index, dst, size, catmap, split_name):
    """Resize images + build records for one split. img_index maps stem -> source image path."""
    out_img_dir = os.path.join(dst, "images", split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    records, missing_xml, missing_img, unknown_names = [], [], [], set()
    for stem in sorted(stems):
        xml_path = os.path.join(xml_dir, stem + ".xml")
        if not os.path.isfile(xml_path):
            missing_xml.append(stem)
            continue
        rec, unknown = build_record(xml_path, size, catmap)
        unknown_names |= unknown
        img_path = img_index.get(stem)
        if img_path is None:
            missing_img.append(stem)
            continue
        img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
        img.save(os.path.join(out_img_dir, stem + ".png"))
        rec.width_resized = rec.height_resized = size
        records.append(rec)
    return records, missing_xml, missing_img, unknown_names


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build compact TB-only 512 COCO dataset")
    ap.add_argument("--src", help="TBX11K dataset root")
    ap.add_argument("--dst", help="output root for the compact dataset")
    ap.add_argument("--xml-dir", default=None, help="override the auto-discovered XML dir")
    ap.add_argument("--train-list", default=None, help="file listing TB train image ids")
    ap.add_argument("--val-list", default=None, help="file listing TB val image ids")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--write-agnostic", action="store_true",
                    help="also write category-agnostic (single 'TB' class) JSONs")
    ap.add_argument("--inspect", action="store_true", help="report the layout and exit")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if not args.src:
        ap.error("--src is required (or use --selftest)")
    if not os.path.isdir(args.src):
        print(f"ERROR: --src does not exist: {args.src}\n"
              f"Download TBX11K first (see the notebook's download cell).", file=sys.stderr)
        return 2

    rep = discover_layout(args.src)
    if args.inspect:
        print_report(rep)
        return 0

    if rep["n_xml"] == 0:
        print_report(rep)
        print("\nERROR: no XML annotations found under --src. Wrong folder, or the archive did not "
              "extract as expected.", file=sys.stderr)
        return 2

    xml_dir = args.xml_dir or pick_xml_dir(rep)
    img_index = rep["img_index"]
    print(f"using xml-dir : {xml_dir}  ({rep['n_xml']} xml files)")
    print(f"indexed images: {rep['n_img']}")

    # splits: explicit flags > auto-discovered list files > deterministic 3:1 fallback
    train_list = args.train_list or guess_list(rep, "train")
    val_list = args.val_list or guess_list(rep, "val")
    train_ids, val_ids = read_id_list(train_list), read_id_list(val_list)
    all_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.endswith(".xml"))

    if train_ids and val_ids:
        # keep only ids that actually have TB annotations
        train_ids &= set(all_stems)
        val_ids &= set(all_stems)
        print(f"split lists   : train={train_list}\n                val={val_list}")
    else:
        val_ids = set(all_stems[::4])
        train_ids = set(s for s in all_stems if s not in val_ids)
        print("[warn] no usable train/val list files found; using a deterministic 3:1 split "
              "(DEVIATION from the official split — record this in results.md).")
    print(f"TB images     : train={len(train_ids)} val={len(val_ids)} "
          f"(paper Table 2: ~600 / ~200)")

    os.makedirs(os.path.join(args.dst, "annotations"), exist_ok=True)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        recs, miss_xml, miss_img, unknown = process_split(
            ids, xml_dir, img_index, args.dst, args.size, DEFAULT_CATMAP, split)
        coco = to_coco(recs, agnostic=False)
        out = os.path.join(args.dst, "annotations", f"tb_{split}.json")
        json.dump(coco, open(out, "w"))
        print(f"[{split}] images={len(coco['images'])} boxes={len(coco['annotations'])} "
              f"missing_xml={len(miss_xml)} missing_img={len(miss_img)} -> {out}")
        if unknown:
            print(f"[{split}] !! UNMAPPED class names (boxes dropped): {sorted(unknown)}\n"
                  f"        add them to DEFAULT_CATMAP in {__file__}")
        if args.write_agnostic:
            coco_a = to_coco(recs, agnostic=True)
            out_a = os.path.join(args.dst, "annotations", f"tb_{split}_agnostic.json")
            json.dump(coco_a, open(out_a, "w"))
            print(f"[{split}] agnostic boxes={len(coco_a['annotations'])} -> {out_a}")
        if not recs:
            print(f"ERROR: split '{split}' produced 0 images — check the layout with --inspect.",
                  file=sys.stderr)
            return 2
    print("\nDone. RECORD the counts above in results.md and spot-check a few overlays.")
    return 0


# --------------------------------------------------------------------------------------
def selftest():
    """Build a tiny synthetic dataset and verify resize + box scaling + COCO output."""
    print("Running synthetic self-test (no real data needed)...")
    tmp = tempfile.mkdtemp(prefix="tbx11k_selftest_")
    src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
    xml_dir = os.path.join(src, "annotations", "xml")
    img_dir = os.path.join(src, "imgs", "tb")
    os.makedirs(xml_dir); os.makedirs(img_dir)

    cases = [
        ("caseA", 3000, 2000, "ActiveTuberculosis", (300, 400, 900, 1200)),
        ("caseB", 1000, 1000, "ObsoletePulmonaryTuberculosis", (100, 100, 500, 500)),
    ]
    for stem, W, H, cname, (xmin, ymin, xmax, ymax) in cases:
        Image.new("RGB", (W, H), (123, 123, 123)).save(os.path.join(img_dir, stem + ".png"))
        open(os.path.join(xml_dir, stem + ".xml"), "w").write(
            f"<annotation><size><width>{W}</width><height>{H}</height></size>"
            f"<object><name>{cname}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
            f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>")

    rep = discover_layout(src)
    assert rep["n_xml"] == 2 and rep["n_img"] == 2, rep
    assert pick_xml_dir(rep) == xml_dir, pick_xml_dir(rep)
    assert dict(rep["class_names"]).keys() == {"ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"}
    print("PASS layout discovery (xml dir, image index, class names)")

    os.makedirs(os.path.join(dst, "annotations"), exist_ok=True)
    for split, ids in (("train", {"caseA"}), ("val", {"caseB"})):
        recs, mx, mi, unk = process_split(ids, xml_dir, rep["img_index"], dst, 512,
                                          DEFAULT_CATMAP, split)
        assert not mx and not mi and not unk, (mx, mi, unk)
        assert Image.open(os.path.join(dst, "images", split, ids.copy().pop() + ".png")).size == (512, 512)
        json.dump(to_coco(recs), open(os.path.join(dst, "annotations", f"tb_{split}.json"), "w"))
    print("PASS resize -> 512x512")

    a = json.load(open(os.path.join(dst, "annotations", "tb_train.json")))
    sx, sy = 512 / 3000, 512 / 2000
    exp = [300 * sx, 400 * sy, (900 - 300) * sx, (1200 - 400) * sy]
    got = a["annotations"][0]["bbox"]
    assert all(abs(g - e) < 0.05 for g, e in zip(got, exp)), f"box mismatch: {got} vs {exp}"
    assert a["annotations"][0]["category_id"] == 1
    b = json.load(open(os.path.join(dst, "annotations", "tb_val.json")))
    assert b["annotations"][0]["category_id"] == 2
    print("PASS box scaling (per-axis factors)")
    print("PASS category mapping active/latent")

    recs_v, _, _, _ = process_split({"caseB"}, xml_dir, rep["img_index"], dst, 512,
                                    DEFAULT_CATMAP, "val")
    ag = to_coco(recs_v, agnostic=True)
    assert ag["categories"] == [{"id": 1, "name": "TB"}]
    assert ag["annotations"][0]["category_id"] == 1
    print("PASS category-agnostic collapse")
    print("PASS COCO JSON structure")
    print("\nSelf-test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
