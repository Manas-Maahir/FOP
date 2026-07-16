"""Torch/torchvision tests for the RetinaNet + SAS integration (run on Colab).

Validates the piece the mmdet->torchvision pivot introduced: that the shared SAS block sits
correctly between the FPN and the RetinaNet head, that the detector still trains and infers, and
that weight sharing across pyramid levels actually holds.

Run:  python tests/test_tv_model.py
Skips cleanly if torch/torchvision are unavailable (e.g. the local Windows box).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torchvision  # noqa: F401
    _OK = hasattr(torch, "nn") and hasattr(torch, "randn")
except Exception:
    _OK = False

if _OK:
    from symformer_tb.tv_model import build_model, count_parameters, SASBackbone

SAS = dict(attention="symattention", pe="spe", use_stn=True, direction="r2l",
           num_heads=8, num_points=4)


def _skip():
    if not _OK:
        print("SKIP: torch/torchvision not available")
        return True
    return False


def _tiny_batch(n=2, size=128):
    imgs = [torch.rand(3, size, size) for _ in range(n)]
    targets = []
    for _ in range(n):
        targets.append({
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),  # 1 = TB (0 = background)
            "image_id": torch.tensor(0),
        })
    return imgs, targets


def test_baseline_builds_without_sas():
    if _skip():
        return
    m = build_model(sas=None, image_size=128, pretrained_backbone=False)
    assert not isinstance(m.backbone, SASBackbone)
    assert count_parameters(m)["sas"] == 0


def test_symformer_wraps_backbone_and_adds_few_params():
    if _skip():
        return
    m = build_model(sas=SAS, image_size=128, pretrained_backbone=False)
    assert isinstance(m.backbone, SASBackbone)
    p = count_parameters(m)
    assert p["sas"] > 0
    # the paper's point: the novel block is cheap relative to the detector
    assert p["sas_fraction"] < 0.15, p


def test_sas_weights_shared_across_levels():
    if _skip():
        return
    m = build_model(sas=SAS, image_size=128, pretrained_backbone=False)
    # one SASBlock instance is reused for every pyramid level
    assert isinstance(m.backbone.sas, torch.nn.Module)
    n_sas_modules = sum(1 for mod in m.backbone.modules() if type(mod).__name__ == "SASBlock")
    assert n_sas_modules == 1, f"expected exactly one shared SASBlock, found {n_sas_modules}"


def test_backbone_preserves_level_shapes():
    if _skip():
        return
    base = build_model(sas=None, image_size=128, pretrained_backbone=False)
    sasm = build_model(sas=SAS, image_size=128, pretrained_backbone=False)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        f_base = base.backbone(x)
        f_sas = sasm.backbone(x)
    assert list(f_base.keys()) == list(f_sas.keys())
    for k in f_base:
        assert f_base[k].shape == f_sas[k].shape, (k, f_base[k].shape, f_sas[k].shape)


def test_training_step_produces_finite_losses():
    if _skip():
        return
    torch.manual_seed(0)
    m = build_model(sas=SAS, image_size=128, pretrained_backbone=False)
    m.train()
    imgs, targets = _tiny_batch()
    losses = m(imgs, targets)
    total = sum(losses.values())
    assert torch.isfinite(total), losses
    total.backward()
    grads = [p.grad for p in m.backbone.sas.parameters() if p.grad is not None]
    assert grads, "no gradients reached the SAS block"
    assert all(torch.isfinite(g).all() for g in grads)


def test_inference_returns_detection_dicts():
    if _skip():
        return
    m = build_model(sas=SAS, image_size=128, pretrained_backbone=False)
    m.eval()
    with torch.no_grad():
        out = m([torch.rand(3, 128, 128)])
    assert isinstance(out, list) and len(out) == 1
    for k in ("boxes", "scores", "labels"):
        assert k in out[0], out[0].keys()


def _run():
    if _skip():
        print("0 tests run — needs torch+torchvision (run this on Colab)")
        return
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} torchvision-model tests passed")


if __name__ == "__main__":
    _run()
