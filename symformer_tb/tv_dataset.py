"""COCO dataset for the torchvision detection API (TB-only, 512x512).

Reads the COCO JSON produced by ``tools/prepare_tbx11k.py`` and yields the
(image_tensor, target_dict) pairs torchvision's detectors expect:

    image  : FloatTensor [3, H, W] in [0, 1]   (the model's own transform normalises)
    target : {"boxes": FloatTensor[N, 4] in xyxy,
              "labels": Int64Tensor[N],
              "image_id": Int64Tensor[]}       (image_id is needed for COCO eval)

Augmentation is the paper's only one: random horizontal flip.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image


class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file: str, img_dir: str, train: bool = False,
                 hflip_prob: float = 0.5):
        from pycocotools.coco import COCO

        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.train = train
        self.hflip_prob = hflip_prob if train else 0.0
        # keep only images that actually have boxes (mirrors filter_empty_gt)
        ids = sorted(self.coco.imgs.keys())
        self.ids = [i for i in ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0]

    def __len__(self):
        return len(self.ids)

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
        labels = torch.as_tensor(labels, dtype=torch.int64)

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


def collate_fn(batch):
    """Detection batches are ragged: keep images/targets as lists."""
    return tuple(zip(*batch))


def build_loader(ann_file: str, img_dir: str, batch_size: int, train: bool,
                 num_workers: int = 2, seed: Optional[int] = None):
    ds = CocoDetectionDataset(ann_file, img_dir, train=train)
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        collate_fn=collate_fn, generator=g, drop_last=False,
    )
