#!/usr/bin/env python
"""Build the compact TBX11K dataset for local training.

What it does:
  * resizes every selected image to a square ``--size`` (default 512), scaling boxes by the same
    per-axis factors,
  * writes COCO-format detection JSON for train and val, categories {active, latent}, plus a
    **category-agnostic** variant (all TB boxes -> one "TB" class) which is what the primary
    AP/AP50 metric uses,
  * with ``--scope all`` (the default) also emits the **non-TB** images and the image-level class
    labels needed for stage 2.

Two scopes
----------
``--scope tb``   only the ~1,200 CXRs carrying TB boxes. This is what the Colab PoC built.
``--scope all``  all 11,200 images. Additionally writes:
                   ``all_{split}[_agnostic].json`` -- every image, non-TB ones carrying **zero
                       annotations**, for the all-images evaluation mode where false positives on
                       healthy chests finally count;
                   ``cls_{split}.json`` -- ``file_name -> healthy | sick_non_tb | tb`` for the
                       stage-2 classification head.

Note that stage-1 *detection training* still uses the TB-only file either way: the paper (§3.4)
trains the detector on TB images alone, "to avoid drowning the detector in pure-background non-TB
images". ``--scope all`` widens what you can *evaluate* and adds stage 2; it does not change the
stage-1 training set.

The official TBX11K README documents download links but NOT the archive's folder layout, so this
script **auto-discovers** it: run ``--inspect`` first to see what is actually in your copy.

    python tools/prepare_tbx11k.py --inspect --src data/raw/TBX11K
    python tools/prepare_tbx11k.py --src data/raw/TBX11K --dst data/tbx11k_512 \
        --scope all --size 512 --write-agnostic --jobs 8

``--selftest`` builds a tiny synthetic dataset and verifies the resize / box-scaling / COCO / class
label logic with no real data -- run it before touching the real thing.
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

# Image-level classes. Order fixed so index 2 is always TB -- symformer_tb/metrics.py binarises on
# that index for sensitivity/specificity.
CLASSES = ("healthy", "sick_non_tb", "tb")

# Directory-name fragments that identify an image's class inside the archive. TBX11K lays images out
# as imgs/health/*, imgs/sick/*, imgs/tb/*, and the split .txt files carry those same prefixes.
CLASS_DIR_HINTS = {
    "health": "healthy",
    "healthy": "healthy",
    "sick": "sick_non_tb",
    "tb": "tb",
}

# Paper Table 2, used only for a sanity warning -- never to filter.
EXPECTED = {
    "train": {"healthy": 3000, "sick_non_tb": 3000, "tb": 600},
    "val": {"healthy": 800, "sick_non_tb": 800, "tb": 200},
}


@dataclass
class Record:
    file_name: str
    width: int
    height: int
    boxes: list = field(default_factory=list)  # (cat_name, x, y, w, h) in resized coords
    width_resized: int = 0
    height_resized: int = 0
    cls: str = "tb"


# --------------------------------------------------------------------------------------
# Layout discovery
# --------------------------------------------------------------------------------------
def classify_path(path: str) -> Optional[str]:
    """Infer an image's class from where it sits in the archive.

    Matching is done on whole path *segments*, not substrings: a naive ``"tb" in path`` also fires
    on any directory whose name merely contains those letters, and would silently relabel non-TB
    images as TB.
    """
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    for part in reversed(parts[:-1]):        # skip the filename itself
        if part in CLASS_DIR_HINTS:
            return CLASS_DIR_HINTS[part]
    return None


def discover_layout(src: str) -> dict:
    """Walk `src` once and report where the XMLs, images and list files actually live."""
    xml_by_dir, img_by_dir = Counter(), Counter()
    list_files, xml_paths = [], []
    img_index = {}            # stem -> path (built in the same single walk)
    class_of_stem = {}        # stem -> healthy | sick_non_tb | tb
    class_counts = Counter()

    for dirpath, _dirs, files in os.walk(src):
        for fn in files:
            p = os.path.join(dirpath, fn)
            low = fn.lower()
            if low.endswith(".xml"):
                xml_by_dir[dirpath] += 1
                xml_paths.append(p)
            elif low.endswith(IMG_EXTS):
                img_by_dir[dirpath] += 1
                stem = os.path.splitext(fn)[0]
                if stem not in img_index:
                    img_index[stem] = p
                    cls = classify_path(p)
                    if cls:
                        class_of_stem[stem] = cls
                        class_counts[cls] += 1
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
        "class_of_stem": class_of_stem,
        "class_counts": class_counts,
    }


def print_report(rep: dict) -> None:
    print("=" * 72)
    print("TBX11K layout report for:", rep["src"])
    print("=" * 72)
    if not rep["exists"]:
        print("!! DIRECTORY DOES NOT EXIST - download the dataset first.")
        return
    print(f"\nXML annotation files: {rep['n_xml']}")
    for d, n in rep["xml_dirs"][:8]:
        print(f"   {n:6d}  {d}")
    print(f"\nImage files: {rep['n_img']}")
    for d, n in rep["img_dirs"][:10]:
        print(f"   {n:6d}  {d}")

    print(f"\nImage-level classes inferred from directory names:")
    if rep["class_counts"]:
        for c in CLASSES:
            print(f"   {rep['class_counts'].get(c, 0):6d}  {c}")
        unknown = rep["n_img"] - sum(rep["class_counts"].values())
        if unknown:
            print(f"   {unknown:6d}  !! UNCLASSIFIED - stage 2 and --scope all need these mapped.")
            print(f"           Extend CLASS_DIR_HINTS in {os.path.basename(__file__)}")
    else:
        print("   (none - directory names did not match health/sick/tb; --scope all will fail)")
    print("   Expected per paper Table 2: 5,000 healthy / 5,000 sick-non-TB / 1,200 TB")

    print(f"\nCandidate list (.txt) files: {len(rep['list_files'])}")
    for p in rep["list_files"][:12]:
        print("   ", p)
    tr, va = pick_split_lists(rep)
    print("\nSplit lists that WOULD be used:")
    print("   train ->", tr or "(none found - would fall back to a deterministic 3:1 split)")
    print("   val   ->", va or "(none found - would fall back to a deterministic 3:1 split)")

    print("\nXML class names found (sampled):")
    if rep["class_names"]:
        for c, n in rep["class_names"]:
            mapped = DEFAULT_CATMAP.get(c, "!! UNMAPPED - add it to DEFAULT_CATMAP")
            print(f"   {n:6d}  {c!r:42s} -> {mapped}")
    else:
        print("    (none found - is this the right archive?)")
    print("\nExpected per paper Table 2: ~1,200 TB images total; TB train ~600, val ~200.")
    print("=" * 72)


def pick_xml_dir(rep: dict) -> Optional[str]:
    return rep["xml_dirs"][0][0] if rep["xml_dirs"] else None


def pick_split_lists(rep: dict):
    """Choose the (train_list, val_list) pair from the discovered .txt files.

    Careful: naive substring matching is wrong here. TBX11K ships
    ``{TBX11K,all}_{train,val,trainval,test}.txt`` and **"trainval" contains both "train" and
    "val"** - so a plain `"train" in name` test can select `all_trainval.txt` as the *training*
    split, leaking val into train and silently inflating AP. We therefore exclude "trainval"
    explicitly, and prefer the TBX11K_* pair over all_* (all_* additionally covers the
    mc+shenzhen / da+db extras, which carry no TB boxes) so the pair is consistent.
    """
    def name(p):
        return os.path.basename(p).lower()

    def find(kind: str) -> Optional[str]:
        opts = [p for p in rep["list_files"]
                if kind in name(p) and "trainval" not in name(p) and "test" not in name(p)]
        # prefer TBX11K_* over all_*, then alphabetical for determinism
        opts.sort(key=lambda p: (0 if name(p).startswith("tbx11k") else 1, name(p)))
        return opts[0] if opts else None

    return find("train"), find("val")


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
    """COCO JSON. Records with no boxes still get an image entry -- that is what makes the
    all-images evaluation mode count false positives on healthy chests."""
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


def to_cls_json(records) -> dict:
    """Stage-2 labels, image_id aligned with the ``all_*`` detection JSON built from the same list."""
    return {
        "images": [{"image_id": i, "file_name": r.file_name + ".png", "class": r.cls}
                   for i, r in enumerate(records, start=1)],
        "classes": list(CLASSES),
    }


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


def read_id_list_with_class(path: Optional[str]):
    """Split list -> ``{stem: class}``.

    The list files carry the class in the path (``imgs/health/h0001.png``), which is more reliable
    than re-deriving it from the image index, so we use it when available and fall back otherwise.
    """
    if not path or not os.path.isfile(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            token = s.split()[0]
            stem = os.path.splitext(os.path.basename(token))[0]
            out[stem] = classify_path(token)
    return out


# --------------------------------------------------------------------------------------
# Resizing (parallel)
# --------------------------------------------------------------------------------------
def _resize_one(job):
    """Worker: resize one image. Returns (stem, ok, skipped)."""
    stem, src_path, out_path, size = job
    try:
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return stem, True, True          # already done -- makes prep restartable
        img = Image.open(src_path).convert("RGB").resize((size, size), Image.BILINEAR)
        img.save(out_path)
        return stem, True, False
    except Exception:
        return stem, False, False


def resize_images(jobs, workers: int):
    """Resize in parallel. 11,200 3000x3000 PNG decodes is the wall-clock bottleneck of the whole
    prep step -- roughly an hour single-threaded, minutes across cores."""
    done, skipped, failed = 0, 0, []
    if workers <= 1 or len(jobs) < 8:
        for job in jobs:
            stem, ok, was_skipped = _resize_one(job)
            done += ok
            skipped += was_skipped
            if not ok:
                failed.append(stem)
        return done, skipped, failed

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (stem, ok, was_skipped) in enumerate(ex.map(_resize_one, jobs, chunksize=16), 1):
            done += ok
            skipped += was_skipped
            if not ok:
                failed.append(stem)
            if i % 500 == 0:
                print(f"      {i}/{len(jobs)} images ...", flush=True)
    return done, skipped, failed


def process_split(stems, xml_dir, img_index, class_of_stem, dst, size, catmap, split_name,
                  workers: int = 1):
    """Resize images + build records for one split.

    ``stems`` may include non-TB images (no XML). Those become records with an empty box list and
    their image-level class attached; they are dropped from the TB-only JSONs and kept in the
    all-images ones.
    """
    out_img_dir = os.path.join(dst, "images", split_name)
    os.makedirs(out_img_dir, exist_ok=True)

    jobs, pending = [], []
    missing_img, unknown_names = [], set()

    for stem in sorted(stems):
        img_path = img_index.get(stem)
        if img_path is None:
            missing_img.append(stem)
            continue
        xml_path = os.path.join(xml_dir, stem + ".xml") if xml_dir else None
        if xml_path and os.path.isfile(xml_path):
            rec, unknown = build_record(xml_path, size, catmap)
            unknown_names |= unknown
            rec.cls = "tb"
        else:
            # non-TB image: no annotation file, no boxes
            with Image.open(img_path) as im:
                w, h = im.size
            rec = Record(file_name=stem, width=w, height=h, boxes=[])
            rec.cls = class_of_stem.get(stem) or "healthy"
        rec.width_resized = rec.height_resized = size
        pending.append(rec)
        jobs.append((stem, img_path, os.path.join(out_img_dir, stem + ".png"), size))

    print(f"   [{split_name}] resizing {len(jobs)} images with {workers} worker(s) ...", flush=True)
    _, skipped, failed = resize_images(jobs, workers)
    if skipped:
        print(f"   [{split_name}] {skipped} already present, skipped")
    if failed:
        print(f"   [{split_name}] !! {len(failed)} images failed to convert: {failed[:5]}")

    bad = set(failed)
    records = [r for r in pending if r.file_name not in bad]
    return records, [], missing_img + failed, unknown_names


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the compact TBX11K COCO dataset")
    ap.add_argument("--src", help="TBX11K dataset root")
    ap.add_argument("--dst", help="output root for the compact dataset")
    ap.add_argument("--scope", default="all", choices=["tb", "all"],
                    help="'tb' = TB images only (the Colab PoC scope); 'all' = every image plus "
                         "stage-2 class labels and the all-images eval JSONs")
    ap.add_argument("--xml-dir", default=None, help="override the auto-discovered XML dir")
    ap.add_argument("--train-list", default=None, help="file listing train image ids")
    ap.add_argument("--val-list", default=None, help="file listing val image ids")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel resize workers (0 = cpu_count-1)")
    ap.add_argument("--write-agnostic", action="store_true", default=True,
                    help="also write category-agnostic (single 'TB' class) JSONs")
    ap.add_argument("--no-agnostic", dest="write_agnostic", action="store_false")
    ap.add_argument("--inspect", action="store_true", help="report the layout and exit")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if not args.src:
        ap.error("--src is required (or use --selftest)")
    if not os.path.isdir(args.src):
        print(f"ERROR: --src does not exist: {args.src}\n"
              f"Download TBX11K first (scripts/download_tbx11k.py).", file=sys.stderr)
        return 2

    workers = args.jobs or max(1, (os.cpu_count() or 2) - 1)

    print("scanning the archive (one pass) ...", flush=True)
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
    print(f"indexed images: {rep['n_img']}   scope: {args.scope}")

    # splits: explicit flags > auto-discovered list files > deterministic 3:1 fallback
    auto_train, auto_val = pick_split_lists(rep)
    train_list = args.train_list or auto_train
    val_list = args.val_list or auto_val
    train_map = read_id_list_with_class(train_list)
    val_map = read_id_list_with_class(val_list)
    tb_stems = set(os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.endswith(".xml"))

    class_of_stem = dict(rep["class_of_stem"])
    if train_map:
        class_of_stem.update({k: v for k, v in train_map.items() if v})
    if val_map:
        class_of_stem.update({k: v for k, v in val_map.items() if v})

    if train_map and val_map:
        train_ids = set(train_map) & set(img_index)
        val_ids = set(val_map) & set(img_index)
        print(f"split lists   : train={train_list}\n                val={val_list}")
        if args.scope == "tb":
            train_ids &= tb_stems
            val_ids &= tb_stems
    else:
        all_stems = sorted(tb_stems if args.scope == "tb" else img_index)
        val_ids = set(all_stems[::4])
        train_ids = set(s for s in all_stems if s not in val_ids)
        print("[warn] no usable train/val list files found; using a deterministic 3:1 split "
              "(DEVIATION from the official split - record this in results.md).")

    overlap = train_ids & val_ids
    if overlap:
        print(f"ERROR: train and val overlap by {len(overlap)} images - the split lists are wrong "
              f"(did one of them resolve to a *_trainval.txt?). Refusing to build a leaky dataset.",
              file=sys.stderr)
        return 2

    def summarise(ids, split):
        counts = Counter(class_of_stem.get(s, "unknown") for s in ids)
        counts["tb"] = len(ids & tb_stems) or counts.get("tb", 0)
        print(f"{split:5s} : {len(ids):5d} images  " +
              "  ".join(f"{c}={counts.get(c, 0)}" for c in CLASSES))
        exp = EXPECTED.get(split, {})
        for c, want in exp.items():
            got = counts.get(c, 0)
            if args.scope == "tb" and c != "tb":
                continue
            if abs(got - want) > max(20, want * 0.1):
                print(f"        [warn] {c}: got {got}, paper Table 2 says {want}")
        return counts

    print()
    summarise(train_ids, "train")
    summarise(val_ids, "val")
    print()

    os.makedirs(os.path.join(args.dst, "annotations"), exist_ok=True)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        recs, _miss_xml, miss_img, unknown = process_split(
            ids, xml_dir, img_index, class_of_stem, args.dst, args.size,
            DEFAULT_CATMAP, split, workers=workers)
        if not recs:
            print(f"ERROR: split '{split}' produced 0 images - check the layout with --inspect.",
                  file=sys.stderr)
            return 2

        tb_recs = [r for r in recs if r.boxes]
        ann_dir = os.path.join(args.dst, "annotations")

        # TB-only detection JSONs (stage-1 training + the TB-only eval mode)
        for agnostic, suffix in ((False, ""), (True, "_agnostic")):
            if agnostic and not args.write_agnostic:
                continue
            out = os.path.join(ann_dir, f"tb_{split}{suffix}.json")
            coco = to_coco(tb_recs, agnostic=agnostic)
            with open(out, "w") as f:
                json.dump(coco, f)
            print(f"   [{split}] tb{suffix}: {len(coco['images'])} images, "
                  f"{len(coco['annotations'])} boxes -> {os.path.basename(out)}")

        if args.scope == "all":
            # All-images detection JSONs: non-TB images present with zero annotations
            for agnostic, suffix in ((False, ""), (True, "_agnostic")):
                if agnostic and not args.write_agnostic:
                    continue
                out = os.path.join(ann_dir, f"all_{split}{suffix}.json")
                coco = to_coco(recs, agnostic=agnostic)
                with open(out, "w") as f:
                    json.dump(coco, f)
                print(f"   [{split}] all{suffix}: {len(coco['images'])} images, "
                      f"{len(coco['annotations'])} boxes -> {os.path.basename(out)}")

            # Stage-2 labels, image_id aligned with all_{split}.json (same record order)
            out = os.path.join(ann_dir, f"cls_{split}.json")
            payload = to_cls_json(recs)
            with open(out, "w") as f:
                json.dump(payload, f)
            counts = Counter(r["class"] for r in payload["images"])
            print(f"   [{split}] cls: {dict(counts)} -> {os.path.basename(out)}")

        if miss_img:
            print(f"   [{split}] !! {len(miss_img)} images missing/unreadable")
        if unknown:
            print(f"   [{split}] !! UNMAPPED XML class names (boxes dropped): {sorted(unknown)}\n"
                  f"           add them to DEFAULT_CATMAP in {__file__}")

    print("\nDone. RECORD the counts above in results.md and spot-check a few overlays.")
    return 0


# --------------------------------------------------------------------------------------
def selftest():
    """Build a tiny synthetic dataset and verify resize + box scaling + COCO + class labels."""
    print("Running synthetic self-test (no real data needed)...")
    tmp = tempfile.mkdtemp(prefix="tbx11k_selftest_")
    src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
    xml_dir = os.path.join(src, "annotations", "xml")
    tb_dir = os.path.join(src, "imgs", "tb")
    health_dir = os.path.join(src, "imgs", "health")
    sick_dir = os.path.join(src, "imgs", "sick")
    for d in (xml_dir, tb_dir, health_dir, sick_dir):
        os.makedirs(d)

    cases = [
        ("caseA", 3000, 2000, "ActiveTuberculosis", (300, 400, 900, 1200)),
        ("caseB", 1000, 1000, "ObsoletePulmonaryTuberculosis", (100, 100, 500, 500)),
    ]
    for stem, W, H, cname, (xmin, ymin, xmax, ymax) in cases:
        Image.new("RGB", (W, H), (123, 123, 123)).save(os.path.join(tb_dir, stem + ".png"))
        open(os.path.join(xml_dir, stem + ".xml"), "w").write(
            f"<annotation><size><width>{W}</width><height>{H}</height></size>"
            f"<object><name>{cname}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
            f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>")
    # non-TB images: no XML at all
    Image.new("RGB", (900, 900), (60, 60, 60)).save(os.path.join(health_dir, "h0001.png"))
    Image.new("RGB", (900, 900), (80, 80, 80)).save(os.path.join(sick_dir, "s0001.png"))

    rep = discover_layout(src)
    assert rep["n_xml"] == 2 and rep["n_img"] == 4, rep
    assert pick_xml_dir(rep) == xml_dir, pick_xml_dir(rep)
    assert dict(rep["class_names"]).keys() == {"ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"}
    print("PASS layout discovery (xml dir, image index, class names)")

    assert rep["class_of_stem"]["h0001"] == "healthy"
    assert rep["class_of_stem"]["s0001"] == "sick_non_tb"
    assert rep["class_of_stem"]["caseA"] == "tb"
    assert classify_path("imgs/health/h0001.png") == "healthy"
    assert classify_path("imgs/sick/s0001.png") == "sick_non_tb"
    # segment matching, not substring: a dir merely containing the letters must not match
    assert classify_path("imgs/nottbstuff/x.png") is None
    print("PASS image-level class inference (whole path segments, not substrings)")

    # Regression: the real TBX11K ships {TBX11K,all}_{train,val,trainval,test}.txt. "trainval"
    # contains both "train" and "val", so naive substring matching can pick a trainval file as the
    # training split (leaking val into train). Reproduce that exact filename set here.
    lists_dir = os.path.join(src, "lists")
    os.makedirs(lists_dir)
    for fn in ("TBX11K_val.txt", "all_test.txt", "all_val.txt", "all_train.txt",
               "TBX11K_trainval.txt", "TBX11K_train.txt", "all_trainval.txt"):
        open(os.path.join(lists_dir, fn), "w").write("imgs/tb/caseA.png\n")
    rep2 = discover_layout(src)
    tr, va = pick_split_lists(rep2)
    assert os.path.basename(tr) == "TBX11K_train.txt", tr
    assert os.path.basename(va) == "TBX11K_val.txt", va
    print("PASS split-list selection (rejects *_trainval.txt, prefers the matching TBX11K_ pair)")

    m = read_id_list_with_class(os.path.join(lists_dir, "TBX11K_train.txt"))
    assert m == {"caseA": "tb"}, m
    print("PASS split-list class parsing")

    os.makedirs(os.path.join(dst, "annotations"), exist_ok=True)
    class_of_stem = rep["class_of_stem"]
    for split, ids in (("train", {"caseA", "h0001"}), ("val", {"caseB", "s0001"})):
        recs, _mx, mi, unk = process_split(ids, xml_dir, rep["img_index"], class_of_stem,
                                           dst, 512, DEFAULT_CATMAP, split)
        assert not mi and not unk, (mi, unk)
        assert len(recs) == 2, recs
        for r in recs:
            p = os.path.join(dst, "images", split, r.file_name + ".png")
            assert Image.open(p).size == (512, 512)
        tb_recs = [r for r in recs if r.boxes]
        assert len(tb_recs) == 1
        json.dump(to_coco(tb_recs), open(os.path.join(dst, "annotations", f"tb_{split}.json"), "w"))
        json.dump(to_coco(recs), open(os.path.join(dst, "annotations", f"all_{split}.json"), "w"))
        json.dump(to_cls_json(recs), open(os.path.join(dst, "annotations", f"cls_{split}.json"), "w"))
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

    # The all-images JSON must carry the negative WITH an image entry and no annotations -- this is
    # what makes false positives on healthy chests count during all-images evaluation.
    allj = json.load(open(os.path.join(dst, "annotations", "all_train.json")))
    assert len(allj["images"]) == 2, allj["images"]
    assert len(allj["annotations"]) == 1
    ann_img_ids = {x["image_id"] for x in allj["annotations"]}
    empty = [im for im in allj["images"] if im["id"] not in ann_img_ids]
    assert len(empty) == 1 and empty[0]["file_name"] == "h0001.png", empty
    print("PASS all-images JSON keeps negatives with zero annotations")

    clsj = json.load(open(os.path.join(dst, "annotations", "cls_train.json")))
    assert clsj["classes"] == list(CLASSES)
    by_name = {r["file_name"]: r["class"] for r in clsj["images"]}
    assert by_name == {"caseA.png": "tb", "h0001.png": "healthy"}, by_name
    # image_id must agree between cls_*.json and all_*.json or the classifier filter mislabels
    all_ids = {im["file_name"]: im["id"] for im in allj["images"]}
    cls_ids = {r["file_name"]: r["image_id"] for r in clsj["images"]}
    assert all_ids == cls_ids, (all_ids, cls_ids)
    print("PASS stage-2 class labels + image_id alignment with all_*.json")

    recs_v, _, _, _ = process_split({"caseB"}, xml_dir, rep["img_index"], class_of_stem,
                                    dst, 512, DEFAULT_CATMAP, "val")
    ag = to_coco(recs_v, agnostic=True)
    assert ag["categories"] == [{"id": 1, "name": "TB"}]
    assert ag["annotations"][0]["category_id"] == 1
    print("PASS category-agnostic collapse")
    print("PASS COCO JSON structure")
    print("\nSelf-test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
