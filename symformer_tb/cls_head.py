"""Stage-2 image classification head: healthy / sick-non-TB / TB.

Paper §3.3 and §3.4. The detector alone cannot say "this chest is fine" -- it can only draw boxes,
so on a non-TB image its only options are to draw a false positive or stay silent. SymFormer adds a
classification head on top of the *frozen* stage-1 features and trains it on all 11,200 images; at
inference the classifier vetoes detections on images it calls non-TB. That veto is what produces the
specificity column of Table 3 and what makes the all-images detection mode survivable.

Architecture (paper §3.3): top pyramid level -> 5x (Conv3x3-512 + ReLU) -> GAP -> FC(3).

Which level is "top"
--------------------
The paper's FPN emits {F1..F4} and taps F̂4, the 1/32-stride level. torchvision's RetinaNet FPN emits
five levels keyed ``'0','1','2','3','pool'`` = P3..P7, where the 1/32-stride level is ``'2'`` (P5);
P6/P7 are extra downsampling blocks RetinaNet adds and have no counterpart in the paper's diagram.
So the default tap is ``'2'``. **This mapping is a decision, not a given** -- it is exposed as
``--tap`` so it can be checked rather than assumed.

Training protocol (paper §3.4): freeze the backbone, FPN, SAS block and detection head; train only
this head, 12 epochs, on all images. Freezing is the point -- it keeps the detector's fine-grained
features intact instead of letting a global classification objective wash them out.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """5x (Conv3x3 + ReLU) -> global average pool -> FC.

    Deliberately plain: the paper specifies exactly this, and any extra regularisation here would
    confound a comparison against its Table 3.
    """

    def __init__(self, in_channels: int = 256, hidden: int = 512, num_classes: int = 3,
                 num_convs: int = 5):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(num_convs):
            layers += [nn.Conv2d(c, hidden, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            c = hidden
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.convs(feat)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ClassifierModel(nn.Module):
    """Frozen stage-1 detector + trainable classification head.

    Only the head's parameters are returned by :meth:`trainable_parameters`, and the detector is
    forced into ``eval()`` on every forward so its BatchNorm running statistics cannot drift while
    the head trains -- a frozen backbone that still updates its BN stats is not actually frozen, and
    it would quietly change the stage-1 detector you already measured.
    """

    def __init__(self, adapter, tap: str = "2", hidden: int = 512, num_classes: int = 3,
                 num_convs: int = 5):
        super().__init__()
        self.adapter = adapter
        self.detector = adapter.model
        self.tap = tap

        for p in self.detector.parameters():
            p.requires_grad_(False)

        in_channels = getattr(getattr(self.detector, "backbone", None), "out_channels", 256)
        self.head = ClassificationHead(in_channels=in_channels, hidden=hidden,
                                       num_classes=num_classes, num_convs=num_convs)

    def trainable_parameters(self):
        return [p for p in self.head.parameters() if p.requires_grad]

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        self.detector.eval()                      # keep BN statistics frozen, not just the weights
        with torch.no_grad():
            feats = self.adapter.backbone_features(images)
        if self.tap in feats:
            return feats[self.tap]
        # Fall back to the deepest available level rather than crashing: mmdet necks key their
        # outputs by position and a config with fewer levels is legitimate.
        key = list(feats.keys())[-1]
        return feats[key]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(images))

    def state_dict(self, *a, **kw):
        """Only the head is ours to save; the detector came from the stage-1 checkpoint."""
        return self.head.state_dict(*a, **kw)

    def load_state_dict(self, sd, strict: bool = True):
        return self.head.load_state_dict(sd, strict=strict)


def normalise_batch(images: torch.Tensor, adapter) -> torch.Tensor:
    """Apply the detector's own input normalisation to a plain [B,3,H,W] tensor in [0,1].

    The FPN features the head consumes are only meaningful under the normalisation the detector was
    trained with, so this must not be skipped or re-invented.
    """
    transform = getattr(adapter.model, "transform", None)
    if transform is not None and hasattr(transform, "normalize"):
        return transform.normalize(images)
    preproc = getattr(adapter.model, "data_preprocessor", None)
    if preproc is not None and hasattr(preproc, "mean"):
        return (images * 255.0 - preproc.mean.to(images)) / preproc.std.to(images)
    return images
