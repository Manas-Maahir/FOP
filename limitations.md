# Limitations

Two kinds of limitations are recorded here: **(A)** limitations of the paper itself, and
**(B)** limitations of *our* planned replication (Hybrid · Core-method PoC · Colab Free T4).
See [paper.md](paper.md) and [CLAUDE.md](CLAUDE.md) for context.

---

## A. Limitations of the paper

1. **Latent TB is detected poorly.** Latent-TB AP is very low for every method (e.g. SymFormer
   w/ RetinaNet ≈ 14.7 AP50 / 4.8 AP, TB-only). The authors attribute this to severe class
   imbalance — only **212 latent** vs **924 active** TB images — and note many latent targets are
   located correctly but **misclassified as active**.

2. **Imprecise localization at strict IoU.** `AP` (averaged over IoU 0.5–0.95) is much lower than
   `AP50` for all models, i.e. boxes are roughly right but not tightly localized. The authors argue
   `AP50` is the more meaningful clinical metric (a 0.5-IoU box already guides a radiologist), which
   is a reasonable but **task-specific** justification rather than a fix.

3. **Strong overall class imbalance.** The dataset is deliberately realistic (44.6% healthy /
   44.6% sick-non-TB / 10.7% TB), which makes training harder and is explicitly flagged as an open
   problem for future methods.

4. **The test set is withheld.** Test-set ground truth is private and scored only through an
   **online challenge**. This is good for fairness but means the headline test-set tables (3, 4, 7)
   **cannot be reproduced offline** by anyone — only the val-based results (ablation Table 8,
   cross-val Table 9, cross-dataset Tables 5–6) can.

5. **Single-region data source.** CXRs were collected from **top hospitals in China**. Scanner,
   population, and protocol shifts mean generalization to other regions/equipment is unverified
   in-paper (the cross-dataset tests on DA/DB/MC/Shenzhen are small and image-level only).

6. **Active-vs-latent confusion** is a recurring error mode, limiting the clinical value of the
   fine-grained TB typing the dataset enables.

7. **Symmetry assumption.** SymFormer's inductive bias presumes bilateral symmetry; pathologies
   that are themselves bilateral, or severe pose/rotation, can weaken the signal. SPE's STN
   mitigates but does not eliminate this.

8. **Reproducibility friction.** Results depend on PyTorch + a specific mmdetection version and
   2× TITAN XP; exact configs for every table are not all in the text (the paper defers to "our
   code"), and the TPAMI SymFormer code may lag the dataset release.

---

## B. Limitations of our replication

### Scope (Core-method PoC)
1. **Reduced data → trend, not absolute numbers.** We use only the **TB subset (~1,200 images)**
   resized to **512²**. Expect to reproduce the *direction* of the result (SymFormer > baseline;
   SPE/SymAttention help), **not** the paper's exact AP/AP50 values.
2. **No classification head / non-TB images.** We skip the 10,000 non-TB images and the
   classifier, so we **cannot** reproduce the classification tables (3–4), specificity / false-
   positive filtering, or the realistic clinical-distribution story. Only **detection** (Tables 7–8)
   is targeted, in **TB-only** mode.
3. **Validation, not test.** Because test GT is private, all PoC numbers come from the **val** set;
   they are not directly comparable to the paper's test-set tables.
4. **A subset of the grid.** No SSD/Faster R-CNN/FCOS/Deformable-DETR baselines, no **P2T-Small**
   backbone, no 4-fold cross-validation, no cross-dataset evaluation (these are Tier-2 extensions).

### Compute (Colab Free T4)
5. **Single T4 vs 2× TITAN XP**, with **session time-outs, idle disconnects, and daily quotas**.
   Free tier may even **refuse a GPU**. Training must checkpoint every epoch and auto-resume; long
   runs span multiple sessions.
6. **Memory/throughput.** Batch 8 at 512² may OOM on a 16 GB T4 for some configs; reducing batch
   (and rescaling LR) is itself a deviation from the paper that can shift results.

### Engineering / fidelity
7. **Reimplemented modules.** The official SymFormer (TPAMI) code was not usable for us, so **SPE**
   and **SymAttention** are reimplemented from the equations — small differences (mirror-coordinate
   handling, STN init, sampling op) can affect numbers.
8. **Different detection framework (torchvision, not mmdetection).** The paper used mmdetection, but
   OpenMMLab publishes `mmcv` wheels only up to ~torch 2.1 / Python 3.11 while Colab now runs
   Python 3.12 — `mim install mmcv` falls into a source build that fails, and mmdetection has been
   effectively unmaintained since 2023. We therefore build on **torchvision's
   `retinanet_resnet50_fpn`**, which is the same ResNet-50 + FPN + RetinaNet architecture. The
   science is unchanged (our SAS block is framework-agnostic pure torch), but anchor settings,
   loss details, NMS defaults, and the training loop differ from mmdetection's, so absolute numbers
   will not line up with the paper's even before the reduced-data effect. One concrete consequence:
   torchvision's RetinaNet spikes `bbox_regression` toward ~1e34 and NaNs out as the warmup LR nears
   its peak at this batch size, so we add **gradient clipping** (default max-norm 10, `--grad-clip`)
   — a standard detection-training guard that the paper's mmdetection recipe did not need. This is a
   deviation from the paper's stated settings, noted here for the record.
9. **Deformable sampling via `grid_sample`.** We use `F.grid_sample` rather than Deformable-DETR's
   custom CUDA op — equivalent at single scale and needs no compilation, but not bit-identical.
10. **"RPE" is an approximation.** Relative positional encoding is ill-defined for deformable
    attention (which predicts attention weights directly rather than from query-key dot products);
    we implement it as a learnable per-(head, point) bias on the attention logits. The paper's
    RPE < APE finding may therefore not reproduce faithfully.
9. **Unspecified details.** Hyper-parameters not stated in the paper are filled with mmdetection /
   Deformable-DETR defaults; each such choice is a potential divergence and will be noted in
   [CLAUDE.md](CLAUDE.md).

### Bottom line
This replication is a **proof of the mechanism**, not a bit-for-bit reproduction. A faithful
full reproduction would require the complete dataset, the full two-stage pipeline, the entire
baseline/backbone/CV/cross-dataset grid, and an **online-challenge submission** for the private
test set — beyond a free-tier Colab budget.
