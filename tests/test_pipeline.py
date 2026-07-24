"""Integration tests for the local pipeline: adapters, dataset negatives, resume, classifier head.

These need torch, and are skipped cleanly without it so the numpy-only tests still run anywhere.
They use tiny tensors and a 2-image synthetic dataset -- the point is to catch wiring mistakes
(shape mismatches, label conventions, state that silently fails to round-trip), not to measure
anything.

The resume test is the one that matters most: a resume that restores weights but *not* the
optimizer and scheduler looks like it worked while actually restarting the LR schedule and
momentum, which trains a different model than the one you meant to continue.

Run:  python -m pytest tests/test_pipeline.py   (or)   python tests/test_pipeline.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    _HAS_TORCH = hasattr(torch, "nn")
except Exception:
    _HAS_TORCH = False


def _skip_if_no_torch(name=""):
    if not _HAS_TORCH:
        print(f"SKIP {name}: torch not available")
        return True
    return False


# ---------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------
def make_mini_dataset(root, size=64):
    """Two TB images (with boxes) and two negatives (without), in the real COCO layout."""
    from PIL import Image

    img_dir = os.path.join(root, "images", "train")
    ann_dir = os.path.join(root, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    images, anns, cls_records = [], [], []
    for i in range(4):
        name = f"img{i}.png"
        Image.new("RGB", (size, size), (40 + 30 * i, 40, 40)).save(os.path.join(img_dir, name))
        images.append({"id": i + 1, "file_name": name, "width": size, "height": size})
        is_tb = i < 2
        if is_tb:
            anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": 1,
                         "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0})
        cls_records.append({"image_id": i + 1, "file_name": name,
                            "class": "tb" if is_tb else "healthy"})

    tb_images = [im for im in images if im["id"] in {a["image_id"] for a in anns}]
    cats = [{"id": 1, "name": "TB"}]
    with open(os.path.join(ann_dir, "tb_train_agnostic.json"), "w") as f:
        json.dump({"images": tb_images, "annotations": anns, "categories": cats}, f)
    with open(os.path.join(ann_dir, "all_train_agnostic.json"), "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    with open(os.path.join(ann_dir, "cls_train.json"), "w") as f:
        json.dump({"images": cls_records,
                   "classes": ["healthy", "sick_non_tb", "tb"]}, f)
    return root


# ---------------------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------------------
def test_dataset_drops_negatives_by_default():
    """Stage-1 training must see TB images only -- the paper's §3.4 choice."""
    if _skip_if_no_torch("dataset_drops_negatives"):
        return
    from symformer_tb.tv_dataset import CocoDetectionDataset

    root = make_mini_dataset(tempfile.mkdtemp(prefix="sf_ds_"))
    ds = CocoDetectionDataset(os.path.join(root, "annotations", "all_train_agnostic.json"),
                              os.path.join(root, "images", "train"), keep_empty=False)
    assert len(ds) == 2, f"expected 2 TB images, got {len(ds)}"
    shutil.rmtree(root, ignore_errors=True)


def test_dataset_keeps_negatives_when_asked():
    """All-images evaluation must see the negatives, or false positives on them vanish."""
    if _skip_if_no_torch("dataset_keeps_negatives"):
        return
    from symformer_tb.tv_dataset import CocoDetectionDataset

    root = make_mini_dataset(tempfile.mkdtemp(prefix="sf_ds_"))
    ds = CocoDetectionDataset(os.path.join(root, "annotations", "all_train_agnostic.json"),
                              os.path.join(root, "images", "train"), keep_empty=True)
    assert len(ds) == 4, f"expected all 4 images, got {len(ds)}"

    empty = [ds[i] for i in range(len(ds)) if ds[i][1]["boxes"].shape[0] == 0]
    assert len(empty) == 2, "negatives should yield zero-row box tensors"
    # A zero-row tensor must still have the right shape, or the model's loss will error on it
    assert empty[0][1]["boxes"].shape == (0, 4), empty[0][1]["boxes"].shape
    assert empty[0][1]["labels"].shape == (0,), empty[0][1]["labels"].shape
    shutil.rmtree(root, ignore_errors=True)


def test_negatives_survive_collate():
    if _skip_if_no_torch("negatives_survive_collate"):
        return
    from symformer_tb.tv_dataset import CocoDetectionDataset, collate_fn

    root = make_mini_dataset(tempfile.mkdtemp(prefix="sf_ds_"))
    ds = CocoDetectionDataset(os.path.join(root, "annotations", "all_train_agnostic.json"),
                              os.path.join(root, "images", "train"), keep_empty=True)
    images, targets = collate_fn([ds[i] for i in range(4)])
    assert len(images) == 4 and len(targets) == 4
    assert sum(int(t["boxes"].shape[0]) for t in targets) == 2
    shutil.rmtree(root, ignore_errors=True)


def test_hflip_mirrors_boxes():
    """A flip that moves pixels but not boxes is a silent label corruption."""
    if _skip_if_no_torch("hflip_mirrors_boxes"):
        return
    from symformer_tb.tv_dataset import CocoDetectionDataset

    root = make_mini_dataset(tempfile.mkdtemp(prefix="sf_ds_"), size=64)
    ds = CocoDetectionDataset(os.path.join(root, "annotations", "tb_train_agnostic.json"),
                              os.path.join(root, "images", "train"), train=True)
    ds.hflip_prob = 1.0                                  # force the flip
    _img, target = ds[0]
    # original box is x in [10, 30] on a 64-wide image -> mirrored to [64-30, 64-10] = [34, 54]
    assert torch.allclose(target["boxes"][0][[0, 2]], torch.tensor([34.0, 54.0])), target["boxes"]
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------------------
def test_torchvision_adapter_interface():
    if _skip_if_no_torch("torchvision_adapter"):
        return
    from symformer_tb.adapters import DetAdapter, build_adapter

    adapter = build_adapter("torchvision", sas=None, image_size=128, pretrained_backbone=False)
    assert isinstance(adapter, DetAdapter)

    images = [torch.rand(3, 128, 128)]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]), "image_id": torch.tensor(1)}]

    losses = adapter.loss(images, targets)
    assert "box_loss" in losses and "cls_loss" in losses, losses.keys()
    assert all(torch.isfinite(v) for v in losses.values()), losses

    preds = adapter.predict(images)
    assert len(preds) == 1
    assert set(preds[0]) >= {"boxes", "scores", "labels"}
    assert preds[0]["boxes"].shape[-1] == 4


def test_adapter_with_sas_block():
    """The SAS block must be shared across FPN levels, not instantiated per level."""
    if _skip_if_no_torch("adapter_with_sas"):
        return
    from symformer_tb.adapters import build_adapter

    sas = dict(attention="symattention", pe="spe", use_stn=True, direction="r2l",
               num_heads=8, num_points=4)
    adapter = build_adapter("torchvision", sas=sas, image_size=128, pretrained_backbone=False)
    block = adapter.model.backbone.sas
    assert block is not None
    # one instance, so the parameter count does not scale with the number of pyramid levels
    ids = {id(m) for m in adapter.model.backbone.modules() if type(m) is type(block)}
    assert len(ids) == 1, "SAS block is not shared across levels"

    losses = adapter.loss([torch.rand(3, 128, 128)],
                          [{"boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
                            "labels": torch.tensor([1]), "image_id": torch.tensor(1)}])
    assert all(torch.isfinite(v) for v in losses.values())


def test_adapter_handles_image_with_no_boxes():
    """Negatives reach the loss in the all-images world; an empty target must not explode."""
    if _skip_if_no_torch("adapter_empty_target"):
        return
    from symformer_tb.adapters import build_adapter

    adapter = build_adapter("torchvision", sas=None, image_size=128, pretrained_backbone=False)
    losses = adapter.loss([torch.rand(3, 128, 128)],
                          [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
                            "image_id": torch.tensor(1)}])
    assert all(torch.isfinite(v) for v in losses.values()), losses


def test_mmdet_adapter_if_installed():
    if _skip_if_no_torch("mmdet_adapter"):
        return
    from symformer_tb.adapters import mmdet_available

    if not mmdet_available():
        print("SKIP mmdet_adapter: mmdet not installed (torchvision stack is unaffected)")
        return

    from symformer_tb.adapters import DetAdapter, build_adapter

    adapter = build_adapter("mmdet", sas=None, image_size=128,
                            config="configs/retinanet_r50_fpn_tbx11k_512.py")
    assert isinstance(adapter, DetAdapter)
    images = [torch.rand(3, 128, 128)]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]), "image_id": torch.tensor(1)}]
    losses = adapter.loss(images, targets)
    assert "box_loss" in losses and "cls_loss" in losses, losses.keys()
    assert all(torch.isfinite(v) for v in losses.values()), losses
    preds = adapter.predict(images)
    assert len(preds) == 1 and set(preds[0]) >= {"boxes", "scores", "labels"}


# ---------------------------------------------------------------------------------------
# checkpoint / resume
# ---------------------------------------------------------------------------------------
def test_resume_restores_optimizer_and_scheduler():
    """Resume must continue the schedule, not restart it.

    Restoring only the weights is the classic silent bug: training appears to resume while the LR
    jumps back to its initial value and momentum is zeroed.
    """
    if _skip_if_no_torch("resume_restores_state"):
        return
    import argparse

    from symformer_tb.adapters import build_adapter
    from symformer_tb.trainer import Trainer

    tmp = tempfile.mkdtemp(prefix="sf_resume_")
    args = argparse.Namespace(
        lr=0.005, momentum=0.9, weight_decay=1e-4, milestones=[1, 2], epochs=4,
        amp=False, ema=True, seed=0, grad_clip=10.0, fitness="ultralytics",
        eval_every=0, save_period=0, warmup_iters=10, image_size=128, limit_batches=1,
        batch_size=1, num_workers=0,
    )
    adapter = build_adapter("torchvision", sas=None, image_size=128, pretrained_backbone=False)
    device = torch.device("cpu")

    t1 = Trainer(adapter, [], None, None, args, os.path.join(tmp, "run"), device)
    # advance the schedule and the optimizer so there is real state to lose
    for _ in range(3):
        t1.optimizer.step()
        t1.scheduler.step()
    t1.best_fitness, t1.best_epoch = 12.5, 2
    lr_before = t1.optimizer.param_groups[0]["lr"]
    sched_before = t1.scheduler.last_epoch
    t1.save_ckpt(epoch=2, path=t1.last_pt)

    adapter2 = build_adapter("torchvision", sas=None, image_size=128, pretrained_backbone=False)
    t2 = Trainer(adapter2, [], None, None, args, os.path.join(tmp, "run2"), device)
    t2.resume_from(t1.last_pt)

    assert t2.start_epoch == 3, t2.start_epoch
    assert t2.best_fitness == 12.5, t2.best_fitness
    assert t2.best_epoch == 2, t2.best_epoch
    assert t2.scheduler.last_epoch == sched_before, (t2.scheduler.last_epoch, sched_before)
    assert abs(t2.optimizer.param_groups[0]["lr"] - lr_before) < 1e-12, "LR schedule restarted"

    # and the weights themselves must match
    for (k1, v1), (k2, v2) in zip(t1.model.state_dict().items(), t2.model.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"weight mismatch at {k1}"
    shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_records_provenance():
    """A checkpoint that does not say what produced it is useless three runs later."""
    if _skip_if_no_torch("checkpoint_provenance"):
        return
    import argparse

    from symformer_tb.adapters import build_adapter
    from symformer_tb.trainer import Trainer

    tmp = tempfile.mkdtemp(prefix="sf_prov_")
    sas = dict(attention="symattention", pe="spe", use_stn=True, direction="r2l",
               num_heads=8, num_points=4)
    args = argparse.Namespace(
        lr=0.005, momentum=0.9, weight_decay=1e-4, milestones=[1], epochs=2, amp=False, ema=False,
        seed=7, grad_clip=10.0, fitness="ap50", eval_every=0, save_period=0, warmup_iters=10,
        image_size=128, limit_batches=1, batch_size=1, num_workers=0,
    )
    adapter = build_adapter("torchvision", sas=sas, image_size=128, pretrained_backbone=False)
    t = Trainer(adapter, [], None, None, args, os.path.join(tmp, "run"), torch.device("cpu"))
    t.save_ckpt(epoch=0, path=t.last_pt)

    ckpt = torch.load(t.last_pt, map_location="cpu", weights_only=False)
    assert ckpt["sas_cfg"] == sas, ckpt["sas_cfg"]
    assert ckpt["stack"] == "torchvision"
    assert ckpt["seed"] == 7
    for key in ("torch", "git_sha", "date", "args", "epoch"):
        assert key in ckpt, f"missing provenance key {key}"
    shutil.rmtree(tmp, ignore_errors=True)


def test_strip_optimizer_shrinks_checkpoint():
    if _skip_if_no_torch("strip_optimizer"):
        return
    import argparse

    from symformer_tb.adapters import build_adapter
    from symformer_tb.trainer import Trainer, strip_optimizer

    tmp = tempfile.mkdtemp(prefix="sf_strip_")
    args = argparse.Namespace(
        lr=0.005, momentum=0.9, weight_decay=1e-4, milestones=[1], epochs=2, amp=False, ema=True,
        seed=0, grad_clip=10.0, fitness="ultralytics", eval_every=0, save_period=0,
        warmup_iters=10, image_size=128, limit_batches=1, batch_size=1, num_workers=0,
    )
    adapter = build_adapter("torchvision", sas=None, image_size=128, pretrained_backbone=False)
    t = Trainer(adapter, [], None, None, args, os.path.join(tmp, "run"), torch.device("cpu"))
    t.optimizer.step()
    t.save_ckpt(epoch=0, path=t.last_pt)
    before = t.last_pt.stat().st_size

    strip_optimizer(t.last_pt)
    after = t.last_pt.stat().st_size
    assert after < before, (before, after)
    ckpt = torch.load(t.last_pt, map_location="cpu", weights_only=False)
    assert "optimizer" not in ckpt and ckpt["stripped"] is True
    assert "model" in ckpt, "stripping must not remove the weights"
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------
# classification head
# ---------------------------------------------------------------------------------------
def test_cls_head_shapes_and_freezing():
    if _skip_if_no_torch("cls_head"):
        return
    from symformer_tb.adapters import build_adapter
    from symformer_tb.cls_head import ClassifierModel

    adapter = build_adapter("torchvision", sas=None, image_size=128, pretrained_backbone=False)
    model = ClassifierModel(adapter, tap="2")

    logits = model(torch.rand(2, 3, 128, 128))
    assert logits.shape == (2, 3), logits.shape

    assert all(not p.requires_grad for p in model.detector.parameters()), \
        "the stage-1 detector must be frozen"
    assert all(p.requires_grad for p in model.trainable_parameters()), \
        "the head must be trainable"

    # only the head is saved -- the detector came from the stage-1 checkpoint
    assert set(model.state_dict()) == set(model.head.state_dict())


def test_cls_head_gradients_reach_only_the_head():
    if _skip_if_no_torch("cls_head_grads"):
        return
    from symformer_tb.adapters import build_adapter
    from symformer_tb.cls_head import ClassifierModel

    adapter = build_adapter("torchvision", sas=None, image_size=128, pretrained_backbone=False)
    model = ClassifierModel(adapter, tap="2")
    logits = model(torch.rand(2, 3, 128, 128))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
    assert all(p.grad is None for p in model.detector.parameters()), \
        "gradients leaked into the frozen detector"


def test_cls_dataset_class_indices():
    """Index 2 must be TB: metrics.classification_report binarises on it."""
    if _skip_if_no_torch("cls_dataset"):
        return
    from symformer_tb.tv_dataset import ClassificationDataset

    root = make_mini_dataset(tempfile.mkdtemp(prefix="sf_cls_"))
    ds = ClassificationDataset(os.path.join(root, "annotations", "cls_train.json"),
                               os.path.join(root, "images", "train"))
    assert ds.CLASSES == ("healthy", "sick_non_tb", "tb")
    assert ds.TB_INDEX == 2
    assert len(ds) == 4
    _img, label, img_id = ds[0]
    assert int(label) == 2 and int(img_id) == 1, (label, img_id)
    _img, label, _ = ds[2]
    assert int(label) == 0
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
