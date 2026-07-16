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
7. **Reimplemented modules.** If the official SymFormer (TPAMI) code is unreleased or incompatible,
   we reimplement **SPE** and **SymAttention** from the equations on top of mmdetection — small
   differences (mirror-coordinate handling, STN init, sampling op) can affect numbers.
8. **Version drift.** torch / CUDA / mmcv / mmdetection compatibility is brittle; the exact stack we
   pin will differ from the authors', another source of small discrepancies.
9. **Unspecified details.** Hyper-parameters not stated in the paper are filled with mmdetection /
   Deformable-DETR defaults; each such choice is a potential divergence and will be noted in
   [CLAUDE.md](CLAUDE.md).

### Bottom line
This replication is a **proof of the mechanism**, not a bit-for-bit reproduction. A faithful
full reproduction would require the complete dataset, the full two-stage pipeline, the entire
baseline/backbone/CV/cross-dataset grid, and an **online-challenge submission** for the private
test set — beyond a free-tier Colab budget.
