"""torchvision RetinaNet + the shared SAS block (replaces the mmdetection path).

Why torchvision: OpenMMLab's mmcv/mmdet publish prebuilt wheels only up to ~torch 2.1 / Python
3.11, while Colab now runs Python 3.12 with a much newer torch — `mim install mmcv` falls back to
a source build that generally fails. torchvision ships with Colab, is maintained, and provides the
*same* architecture the paper uses: ResNet-50 + FPN + the RetinaNet head.

The science is unchanged: `SASBlock` (symformer_tb/sas.py) is pure torch and is reused verbatim.
Only the surrounding detector framework differs.

Model:
    backbone (ResNet-50, ImageNet-pretrained) -> FPN (C=256) -> {P3..P7}
      each level -> SASBlock (SHARED weights)          [omitted for the baseline]
      -> RetinaNet head (classification + box regression)

num_classes=2 follows torchvision's convention (index 0 is background), so our single
category-agnostic "TB" class uses label 1 — which is exactly the category_id the prep tool writes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from .sas import SASBlock


class SASBackbone(nn.Module):
    """Wrap a torchvision BackboneWithFPN so every FPN level passes through one shared SAS block."""

    def __init__(self, backbone: nn.Module, sas: SASBlock):
        super().__init__()
        self.backbone = backbone
        self.sas = sas
        # torchvision's detector reads this attribute off the backbone
        self.out_channels = backbone.out_channels

    def forward(self, x):
        feats = self.backbone(x)  # OrderedDict[str, Tensor], one entry per pyramid level
        # the SAME self.sas instance is applied to every level -> weights are shared
        return OrderedDict((k, self.sas(v)) for k, v in feats.items())


def build_model(sas: Optional[dict] = None,
                num_classes: int = 2,
                image_size: int = 512,
                pretrained_backbone: bool = True) -> nn.Module:
    """Build RetinaNet-R50-FPN, optionally with the SAS block after the FPN.

    Args:
        sas: SAS options, e.g. dict(attention='symattention', pe='spe', use_stn=True,
             direction='r2l', num_heads=8, num_points=4). Pass None for the plain RetinaNet
             baseline (paper Table 8 row "No attention / No PE").
        num_classes: torchvision convention — includes background. 2 = background + TB.
        image_size: images are already 512x512 from the prep step, so min_size=max_size=512
                    keeps torchvision's transform from resizing them again.
    """
    from torchvision.models.detection import retinanet_resnet50_fpn

    weights_backbone = None
    if pretrained_backbone:
        from torchvision.models import ResNet50_Weights
        weights_backbone = ResNet50_Weights.IMAGENET1K_V1

    model = retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        min_size=image_size,
        max_size=image_size,
    )

    if sas is not None:
        channels = model.backbone.out_channels  # 256, matches the paper's C
        model.backbone = SASBackbone(model.backbone, SASBlock(channels=channels, **sas))
    return model


def count_parameters(model: nn.Module) -> dict:
    """Handy for reporting how little the SAS block adds (paper: ~negligible vs the detector)."""
    total = sum(p.numel() for p in model.parameters())
    sas = 0
    if hasattr(model, "backbone") and hasattr(model.backbone, "sas"):
        sas = sum(p.numel() for p in model.backbone.sas.parameters())
    return {"total": total, "sas": sas, "sas_fraction": (sas / total) if total else 0.0}
