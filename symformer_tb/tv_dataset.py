"""Datasets for the torchvision detection API and for the stage-2 classifier.

Detection yields the (image, target) pairs torchvision's detectors expect:

    image  : FloatTensor [3, H, W] in [0, 1]   (the model's own transform normalises)
    target : {"boxes": FloatTensor[N, 4] xyxy, "labels": Int64Tensor[N],
              "image_id": Int64Tensor[]}       (image_id is needed for COCO eval)

Augmentation is the paper's only one: random horizontal flip.

**Empty images matter.** ``keep_empty`` controls whether images with no boxes survive. It must be
False for stage-1 training -- the paper (§3.4) trains detection on TB images only, "to avoid drowning
the detector in pure-background non-TB images" -- and True for all-images evaluation, where the 1,600
non-TB val images are precisely what makes false positives count. Silently dropping them is how an
all-images score gets inflated into meaninglessness.
"""

from __future__ import annotations

import os
import platform
from typing import Optional

import torch
from PIL import Image


class CocoDetectionDataset(torch.utils.data.Dataset):
    """COCO-format detection dataset with optional horizontal-flip augmentation.

    Args:
        keep_empty: keep images that have no annotations. See the module docstring -- this flag is
            the difference between the paper's TB-only stage-1 training set and the all-images
            evaluation set.
    """

    def __init__(self, ann_file: str, img_dir: str, train: bool = False,
                 hflip_prob: float = 0.5, keep_empty: bool = False):
        from pycocotools.coco import COCO

        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.train = train
        self.hflip_prob = hflip_prob if train else 0.0
        self.keep_empty = keep_empty

        ids = sorted(self.coco.imgs.keys())
        self.ids = ids if keep_empty else [
            i for i in ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0
        ]

    def __len__(self):
        return len(self.ids)

    def all_boxes_xywh(self):
        """Every GT box in the split, for the labels.jpg distribution figure."""
        out = []
        for i in self.ids:
            for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=i, iscrowd=False)):
                out.append(a["bbox"])
        return out

    def __getitem__(self, idx):
        import torchvision.transforms.functional as TF

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        img = Image.open(path).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])  # COCO xywh -> xyxy
            labels.append(a["category_id"])     # 1 = TB (0 is background in torchvision)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)

        tensor = TF.to_tensor(img)  # [3,H,W] in [0,1]

        if self.hflip_prob > 0 and torch.rand(1).item() < self.hflip_prob:
            tensor = torch.flip(tensor, dims=[-1])
            w_img = tensor.shape[-1]
            if boxes.numel():
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = w_img - x2
                boxes[:, 2] = w_img - x1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(img_id, dtype=torch.int64),
        }
        return tensor, target


class ClassificationDataset(torch.utils.data.Dataset):
    """Stage-2 dataset: all 11,200 images with their image-level class.

    Reads the ``cls_{split}.json`` written by ``tools/prepare_tbx11k.py``:
    ``{"images": [{"file_name": ..., "class": "healthy"|"sick_non_tb"|"tb", "image_id": int}]}``.

    Class order is fixed at ``healthy, sick_non_tb, tb`` so index **2 is TB** everywhere -- the
    metrics module binarises on that index for sensitivity/specificity.
    """

    CLASSES = ("healthy", "sick_non_tb", "tb")
    TB_INDEX = 2

    def __init__(self, ann_file: str, img_dir: str, train: bool = False, hflip_prob: float = 0.5):
        import json

        with open(ann_file) as f:
            payload = json.load(f)
        self.records = payload["images"]
        self.img_dir = img_dir
        self.hflip_prob = hflip_prob if train else 0.0
        self.class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

    def __len__(self):
        return len(self.records)

    def class_counts(self):
        from collections import Counter

        return Counter(r["class"] for r in self.records)

    def __getitem__(self, idx):
        import torchvision.transforms.functional as TF

        rec = self.records[idx]
        img = Image.open(os.path.join(self.img_dir, rec["file_name"])).convert("RGB")
        tensor = TF.to_tensor(img)
        if self.hflip_prob > 0 and torch.rand(1).item() < self.hflip_prob:
            tensor = torch.flip(tensor, dims=[-1])
        label = self.class_to_idx[rec["class"]]
        return tensor, torch.tensor(label, dtype=torch.int64), \
            torch.tensor(int(rec.get("image_id", idx)), dtype=torch.int64)


def collate_fn(batch):
    """Detection batches are ragged: keep images/targets as lists."""
    return tuple(zip(*batch))


def cls_collate_fn(batch):
    images, labels, ids = zip(*batch)
    return torch.stack(images), torch.stack(labels), torch.stack(ids)


def _loader_kwargs(num_workers: int) -> dict:
    """Worker settings that behave on Windows.

    Windows has no fork, so every worker re-imports the module and re-spawns the interpreter --
    expensive enough that persistent_workers is worth it, and a reason to keep the count modest.
    prefetch_factor is only a valid argument when workers > 0.
    """
    kw: dict = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    if platform.system() == "Windows" and num_workers > 8:
        kw["num_workers"] = 8
    return kw


def build_loader(ann_file: str, img_dir: str, batch_size: int, train: bool,
                 num_workers: int = 4, seed: Optional[int] = None,
                 keep_empty: bool = False):
    ds = CocoDetectionDataset(ann_file, img_dir, train=train, keep_empty=keep_empty)
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, collate_fn=collate_fn,
        generator=g, drop_last=False, **_loader_kwargs(num_workers),
    )


def build_cls_loader(ann_file: str, img_dir: str, batch_size: int, train: bool,
                     num_workers: int = 4, seed: Optional[int] = None):
    ds = ClassificationDataset(ann_file, img_dir, train=train)
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, collate_fn=cls_collate_fn,
        generator=g, drop_last=False, **_loader_kwargs(num_workers),
    )
