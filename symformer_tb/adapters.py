"""One interface, two detection stacks.

The trainer must not know whether it is driving torchvision or mmdetection, so both hide behind
:class:`DetAdapter`:

    loss(images, targets)  -> {"box_loss": Tensor, "cls_loss": Tensor, ...}
    predict(images)        -> [{"boxes": [N,4] xyxy, "scores": [N], "labels": [N]}, ...]

Why two stacks at all
---------------------
The project pivoted to torchvision because Colab's Python 3.12 made ``mim install mmcv``
uninstallable (see README). Locally that constraint is gone, and [report.md](report.md) §7 names
returning to mmdetection as improvement #3: mmdet is what the paper used, so its anchor / loss / NMS
defaults remove a confound from the comparison. Rather than replace one with the other, we keep both
and pin them to the *same* torch, so a torchvision-vs-mmdet difference is attributable to the
detector framework rather than to the tensor library underneath.

``SASBlock`` (symformer_tb/sas.py) is pure torch operating on ``[B, C, H, W]``, so **both stacks
reuse it verbatim** -- ``SASBackbone`` for torchvision, ``SASFPN`` for mmdet. The novel code has
exactly one implementation.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class DetAdapter(Protocol):
    """What the trainer needs from a detector. Deliberately tiny."""

    model: nn.Module

    def loss(self, images: list[torch.Tensor],
             targets: list[dict]) -> dict[str, torch.Tensor]: ...

    def predict(self, images: list[torch.Tensor]) -> list[dict]: ...


# ------------------------------------------------------------------------------------------
# torchvision
# ------------------------------------------------------------------------------------------
class TorchvisionAdapter:
    """``torchvision.models.detection.retinanet_resnet50_fpn`` (+ the shared SAS block).

    torchvision's detectors are mode-switched rather than method-switched: ``model(images, targets)``
    in train mode returns a loss dict, ``model(images)`` in eval mode returns detections. We hide
    that so the trainer never has to touch ``.train()`` / ``.eval()`` for correctness reasons.
    """

    stack = "torchvision"

    def __init__(self, sas: Optional[dict] = None, num_classes: int = 2, image_size: int = 512,
                 pretrained_backbone: bool = True):
        from .tv_model import build_model

        self.model = build_model(sas=sas, num_classes=num_classes, image_size=image_size,
                                 pretrained_backbone=pretrained_backbone)
        self.sas_cfg = sas

    def loss(self, images, targets):
        was_training = self.model.training
        self.model.train()
        losses = self.model(images, targets)
        if not was_training:
            self.model.eval()
        # torchvision names them classification / bbox_regression; normalise for the progress bar
        return {
            "cls_loss": losses.get("classification", torch.zeros((), device=_dev(images))),
            "box_loss": losses.get("bbox_regression", torch.zeros((), device=_dev(images))),
        }

    @torch.no_grad()
    def predict(self, images):
        was_training = self.model.training
        self.model.eval()
        outputs = self.model(images)
        if was_training:
            self.model.train()
        return [{"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"]}
                for o in outputs]

    def backbone_features(self, images: torch.Tensor):
        """FPN feature dict, for the stage-2 classification head to tap."""
        return self.model.backbone(images)


# ------------------------------------------------------------------------------------------
# mmdetection
# ------------------------------------------------------------------------------------------
class MMDetAdapter:
    """mmdetection 3.x RetinaNet built from ``configs/*.py`` (+ ``SASFPN``).

    mmdet models speak ``DetDataSample``, not plain dicts, so this class does the packing. It also
    runs mmdet's own ``data_preprocessor`` (mean/std normalisation and padding) rather than
    duplicating it, which is what keeps the mmdet path faithful to the paper's pipeline instead of
    "mmdet weights driven by torchvision preprocessing".

    Import is lazy and the whole module is optional: ``scripts/setup_env.py`` installs mmdet last
    and tolerates failure, so a machine without it still trains on torchvision.
    """

    stack = "mmdet"

    def __init__(self, config: str, sas: Optional[dict] = None, num_classes: int = 1,
                 image_size: int = 512):
        try:
            from mmengine.config import Config
            from mmengine.registry import init_default_scope
            from mmdet.registry import MODELS
        except ImportError as e:  # pragma: no cover - depends on install
            raise ImportError(
                "The mmdet stack is not installed in this environment.\n"
                "Run `python scripts/setup_env.py` (mmdet is stage 3), or use --stack torchvision."
            ) from e

        from . import mmdet_plugin  # noqa: F401  -- registers SASFPN with mmdet's MODELS registry

        init_default_scope("mmdet")
        cfg = Config.fromfile(config)

        # The config carries the SAS settings for its ablation cell; a CLI override wins so
        # tools/train.py can drive every Table 8 cell from flags without 13 config edits.
        if sas is not None:
            cfg.model.neck.type = "SASFPN"
            cfg.model.neck.sas = dict(sas)
        elif sas is None and cfg.model.neck.get("type") == "SASFPN":
            cfg.model.neck.sas = None
        cfg.model.bbox_head.num_classes = num_classes

        self.cfg = cfg
        self.image_size = image_size
        self.model = MODELS.build(cfg.model)
        self.model.init_weights()
        self.sas_cfg = sas

    # -- packing -------------------------------------------------------------------------
    def _samples(self, images: list[torch.Tensor], targets: Optional[list[dict]] = None):
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        samples = []
        for i, img in enumerate(images):
            h, w = int(img.shape[-2]), int(img.shape[-1])
            s = DetDataSample()
            s.set_metainfo({
                "img_shape": (h, w),
                "ori_shape": (h, w),
                "pad_shape": (h, w),
                "scale_factor": (1.0, 1.0),
                "batch_input_shape": (h, w),
                "img_id": int(targets[i]["image_id"]) if targets else i,
            })
            if targets is not None:
                inst = InstanceData()
                inst.bboxes = targets[i]["boxes"].float()
                # mmdet labels are 0-based over foreground classes; ours are torchvision-style
                # (1 = TB, 0 = background), so shift down by one.
                inst.labels = (targets[i]["labels"].long() - 1).clamp(min=0)
                s.gt_instances = inst
            samples.append(s)
        return samples

    def _preprocess(self, images, targets, training: bool):
        """Run mmdet's own data_preprocessor so normalisation matches the config."""
        data = {"inputs": [(img * 255.0).to(torch.uint8) for img in images],
                "data_samples": self._samples(images, targets)}
        return self.model.data_preprocessor(data, training)

    # -- interface -----------------------------------------------------------------------
    def loss(self, images, targets):
        was_training = self.model.training
        self.model.train()
        data = self._preprocess(images, targets, training=True)
        raw = self.model.loss(data["inputs"], data["data_samples"])
        if not was_training:
            self.model.eval()

        # mmdet returns loss_cls / loss_bbox, each often a *list* (one entry per FPN level).
        def reduce(v):
            if isinstance(v, (list, tuple)):
                return sum(x.sum() for x in v)
            return v.sum()

        out: dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            if "cls" in k:
                out["cls_loss"] = out.get("cls_loss", 0) + reduce(v)
            elif "bbox" in k or "box" in k:
                out["box_loss"] = out.get("box_loss", 0) + reduce(v)
            else:
                out[k] = reduce(v)
        return out

    @torch.no_grad()
    def predict(self, images):
        was_training = self.model.training
        self.model.eval()
        data = self._preprocess(images, None, training=False)
        results = self.model.predict(data["inputs"], data["data_samples"])
        if was_training:
            self.model.train()

        out = []
        for r in results:
            inst = r.pred_instances
            out.append({
                "boxes": inst.bboxes,
                "scores": inst.scores,
                "labels": inst.labels + 1,   # back to our 1 = TB convention
            })
        return out

    def backbone_features(self, images: torch.Tensor):
        feats = self.model.extract_feat(images)
        return {str(i): f for i, f in enumerate(feats)}


# ------------------------------------------------------------------------------------------
# factory
# ------------------------------------------------------------------------------------------
def build_adapter(stack: str = "torchvision", sas: Optional[dict] = None,
                  image_size: int = 512, config: Optional[str] = None,
                  pretrained_backbone: bool = True) -> DetAdapter:
    """Build the requested stack.

    ``num_classes`` differs by convention and is not a modelling choice: torchvision counts
    background as class 0 (so 2 = background + TB), mmdet counts foreground only (so 1 = TB).
    """
    if stack == "torchvision":
        return TorchvisionAdapter(sas=sas, num_classes=2, image_size=image_size,
                                  pretrained_backbone=pretrained_backbone)
    if stack == "mmdet":
        cfg = config or "configs/retinanet_r50_fpn_tbx11k_512.py"
        return MMDetAdapter(config=cfg, sas=sas, num_classes=1, image_size=image_size)
    raise ValueError(f"unknown stack {stack!r} (expected 'torchvision' or 'mmdet')")


def mmdet_available() -> bool:
    """Whether `--stack mmdet` can run here. Used to skip, not to fail."""
    try:
        import mmcv  # noqa: F401
        import mmdet  # noqa: F401
        return True
    except Exception:
        return False


def _dev(images):
    return images[0].device if images else torch.device("cpu")
