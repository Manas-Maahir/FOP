#!/usr/bin/env python
"""Stage 2: train the image classification head on top of a frozen stage-1 detector.

Paper §3.4: freeze the backbone + FPN + SAS + detection head, then train only the 3-way classifier
(healthy / sick-non-TB / TB) on **all** images for 12 epochs. This is the half of SymFormer the
Colab PoC never ran, because it needs the 10,000 non-TB images that were out of scope there.

    python tools/train_cls.py --weights runs/detect/train/weights/best.pt \
        --data-root data/tbx11k_512 --epochs 12

Produces a run directory under ``runs/classify/`` with ``results.csv``, a confusion matrix, and
``weights/{best,last}.pt``. Feed that checkpoint to ``tools/val.py --mode all --cls-ckpt ...`` to see
the false-positive filtering it exists for.

Reported metrics are paper Table 3's columns: accuracy, AUC(TB), sensitivity, specificity, AP, AR.
Sensitivity and specificity are computed on the TB-vs-non-TB binarisation, which is the clinically
meaningful one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", required=True, help="stage-1 detector checkpoint")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-ann", default="annotations/cls_train.json")
    ap.add_argument("--train-img", default="images/train")
    ap.add_argument("--val-ann", default="annotations/cls_val.json")
    ap.add_argument("--val-img", default="images/val")

    ap.add_argument("--epochs", type=int, default=12, help="paper §3.4 stage 2")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="larger than detection: the backbone runs under no_grad, so activations "
                         "for the frozen trunk are not retained")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--milestones", type=int, nargs="+", default=[8, 11])
    ap.add_argument("--tap", default="2",
                    help="FPN level feeding the head. '2' = P5, the 1/32-stride level, which is the "
                         "paper's F4; see symformer_tb/cls_head.py")
    ap.add_argument("--class-weights", action="store_true",
                    help="inverse-frequency weighted CE. OFF by default: the paper does not mention "
                         "reweighting, and adding it would confound a Table 3 comparison")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--project", default="runs/classify")
    ap.add_argument("--name", default="train")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-batches", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    import torch.nn as nn
    from tqdm import tqdm

    from symformer_tb.adapters import build_adapter
    from symformer_tb.cls_head import ClassifierModel, normalise_batch
    from symformer_tb.metrics import classification_report
    from symformer_tb.trainer import colorstr, increment_path, _autocast, _grad_scaler, gpu_mem_str
    from symformer_tb.tv_dataset import ClassificationDataset, build_cls_loader
    from symformer_tb import plotting

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = increment_path(Path(args.project) / args.name)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    # -- frozen stage-1 detector ----------------------------------------------------------
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    adapter = build_adapter(stack=ckpt.get("stack", "torchvision"), sas=ckpt.get("sas_cfg"),
                            image_size=args.image_size, pretrained_backbone=False)
    adapter.model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    adapter.model.to(device)

    model = ClassifierModel(adapter, tap=args.tap).to(device)
    params = model.trainable_parameters()

    print(colorstr("bold", "\nSymFormer / TBX11K -- stage-2 classification"))
    print(f"  detector : {args.weights}  (stack {ckpt.get('stack', 'torchvision')}, frozen)")
    print(f"  SAS      : {ckpt.get('sas_cfg') or '(none -- baseline)'}")
    print(f"  tap      : FPN level {args.tap!r}")
    print(f"  device   : {device}")
    print(f"  trainable: {sum(p.numel() for p in params):,} params "
          f"(of {sum(p.numel() for p in model.parameters()):,} in the head)")

    root = Path(args.data_root)
    train_loader = build_cls_loader(str(root / args.train_ann), str(root / args.train_img),
                                    batch_size=args.batch_size, train=True,
                                    num_workers=args.num_workers, seed=args.seed)
    val_loader = build_cls_loader(str(root / args.val_ann), str(root / args.val_img),
                                  batch_size=args.batch_size, train=False,
                                  num_workers=args.num_workers)
    counts = train_loader.dataset.class_counts()
    print(f"  train    : {len(train_loader.dataset)} images  {dict(counts)}")
    print(f"  val      : {len(val_loader.dataset)} images")

    weight = None
    if args.class_weights:
        total = sum(counts.values())
        weight = torch.tensor(
            [total / max(counts.get(c, 1), 1) for c in ClassificationDataset.CLASSES],
            dtype=torch.float32, device=device)
        weight = weight / weight.mean()
        print(f"  CE weight: {weight.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                     gamma=0.1)
    amp = args.amp and device.type == "cuda"
    scaler = _grad_scaler(amp)

    csv_path = run_dir / "results.csv"
    best_acc, best_epoch = -1.0, -1
    t0 = time.time()

    for epoch in range(args.epochs):
        model.head.train()
        print(colorstr(("%11s" * 5) % ("Epoch", "GPU_mem", "loss", "train_acc", "Size")))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=118,
                    bar_format="{l_bar}{bar:12}{r_bar}")
        run_loss, correct, seen = 0.0, 0, 0

        for i, (images, labels, _ids) in pbar:
            if args.limit_batches and i >= args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(amp):
                logits = model(normalise_batch(images, adapter))
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += float(loss) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
            pbar.set_description(("%11s" * 2 + "%11.4g" * 3) % (
                f"{epoch + 1}/{args.epochs}", gpu_mem_str(),
                run_loss / max(seen, 1), correct / max(seen, 1), args.image_size))

        scheduler.step()

        # -- validation ------------------------------------------------------------------
        model.head.eval()
        y_true, y_pred, tb_score = [], [], []
        with torch.no_grad():
            for images, labels, _ids in val_loader:
                logits = model(normalise_batch(images.to(device), adapter))
                probs = torch.softmax(logits.float(), dim=1)
                y_true += labels.tolist()
                y_pred += logits.argmax(1).cpu().tolist()
                tb_score += probs[:, ClassificationDataset.TB_INDEX].cpu().tolist()

        rep = classification_report(y_true, y_pred, tb_score,
                                    tb_index=ClassificationDataset.TB_INDEX)
        print(colorstr(("%13s" * 6) % ("Acc", "AUC(TB)", "Sens", "Spec", "AP", "AR")))
        print(("%13.4g" * 6) % (rep["accuracy"], rep["auc_tb"], rep["sensitivity"],
                                rep["specificity"], rep["AP"], rep["AR"]))

        row = {"epoch": epoch + 1, "time": round(time.time() - t0, 1),
               "train/loss": round(run_loss / max(seen, 1), 5),
               "train/acc": round(correct / max(seen, 1), 5),
               "metrics/accuracy": round(rep["accuracy"], 3),
               "metrics/auc_tb": round(rep["auc_tb"], 3),
               "metrics/sensitivity": round(rep["sensitivity"], 3),
               "metrics/specificity": round(rep["specificity"], 3),
               "lr/0": optimizer.param_groups[0]["lr"]}
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

        payload = {"model": model.state_dict(), "epoch": epoch, "args": vars(args),
                   "stack": ckpt.get("stack", "torchvision"), "sas_cfg": ckpt.get("sas_cfg"),
                   "metrics": rep, "classes": list(ClassificationDataset.CLASSES)}
        torch.save(payload, run_dir / "weights" / "last.pt")
        if rep["accuracy"] > best_acc:
            best_acc, best_epoch = rep["accuracy"], epoch
            torch.save(payload, run_dir / "weights" / "best.pt")
            print(colorstr("green", "bold", f"  new best accuracy {best_acc:.2f} -> best.pt"))

        # 3x3 confusion matrix, rebuilt each epoch so an interrupted run still leaves one
        import numpy as np

        n = len(ClassificationDataset.CLASSES)
        mat = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            mat[t, p] += 1
        plotting.plot_cls_confusion(mat, ClassificationDataset.CLASSES,
                                    run_dir / "confusion_matrix.png")
        plotting.plot_results(csv_path, run_dir / "results.png")

    (run_dir / "final_metrics.json").write_text(json.dumps(rep, indent=2))
    print(colorstr("bold", f"\nStage-2 complete in {(time.time() - t0) / 60:.1f} min"))
    print(f"  best accuracy {best_acc:.2f} @ epoch {best_epoch + 1}")
    print(f"  final: Acc {rep['accuracy']:.1f}  AUC(TB) {rep['auc_tb']:.1f}  "
          f"Sens {rep['sensitivity']:.1f}  Spec {rep['specificity']:.1f}")
    print(f"Results saved to {run_dir}")
    print("\nNow use it to filter detections on the all-images split:")
    print(f"    python tools/val.py --weights {args.weights} --data-root {args.data_root} \\")
    print(f"        --mode all --cls-ckpt {run_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
