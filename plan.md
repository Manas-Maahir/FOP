# plan.md — Execution Runbook: SymFormer / TBX11K Core-Method PoC

**Audience:** the executing agent (Sonnet). **Style:** instructions only — *no code in this file*;
you write the code as you go. Each phase ends with a **GATE** (must pass before the next phase) and a
**RECORD** (what to log). Work **one phase at a time, in order**.

**Companion docs (read them):**
- [paper.md](paper.md) — what the paper does, the method, and the target numbers (Tables 2, 7, 8).
- [CLAUDE.md](CLAUDE.md) — locked scope, target architecture diagram, training recipe, status checklist.
- [limitations.md](limitations.md) — caveats you must respect and restate in the write-up.

---

## Preface — goal & principles

**Goal.** Demonstrate the paper's central claim on a reduced budget:
1. **SymFormer w/ RetinaNet (ResNet-50) beats the RetinaNet baseline** on **category-agnostic TB
   detection** (COCO AP50 / AP), measured on the **TB-only validation** set.
2. The **Table 8 ablation** trend holds: **SymAttention > vanilla attention**, **SPE > APE > RPE**,
   **STN** adds a further gain, **right→left** ≥ left→right.

**Success = the trend (direction of the gains), NOT the paper's absolute numbers.** Reduced data and a
single T4 mean magnitudes will differ; the *ordering* is what we are verifying.

**Execution principles.**
- Do not start a phase until the previous phase's **GATE** passes. If a GATE fails, **stop and fix** —
  do not paper over it by moving on.
- **Persist everything to Google Drive**: the compact dataset, per-epoch checkpoints, logs, and results.
  Assume the Colab session can die at any moment; every run must be **resumable**.
- **Reproducibility:** pick ONE random seed in Phase 0 and reuse it everywhere; log it and all library
  versions with every run.
- **Record as you go** into `results.md` — do not rely on scrollback or notebook output that a
  disconnect will erase.
- When a detail is missing from the paper, use **mmdetection RetinaNet defaults** (or **Deformable-DETR
  sampling defaults** for the attention op) and **note the deviation** in `results.md`.

---

## Phase 0 — Repository & bookkeeping

- Create the working project layout on Drive (and optionally a Git repo): folders for **data**,
  **configs**, **checkpoints**, **logs**, **results**, **notebooks**.
- Create `results.md` with a table whose columns are: `run-name`, `config summary`, `seed`,
  `library versions`, `AP50`, `AP`, `notes`. Leave it empty for now.
- Choose ONE fixed random **seed** and write it at the top of `results.md`.
- **GATE:** the folder layout and the (empty) `results.md` exist on Drive.
- **RECORD:** the chosen seed and the absolute Drive path of the project root.

---

## Phase 1 — Colab environment

- Open a new Colab notebook; set the runtime to **GPU**; run the GPU check; confirm a **T4** is attached
  and note its VRAM. If Colab grants no GPU, stop and retry later (the free tier sometimes refuses).
- **Mount Google Drive.**
- **No mmcv/mmdetection.** (Tried and rejected: OpenMMLab ships `mmcv` wheels only up to ~torch 2.1 /
  Python 3.11 and is unmaintained since 2023; Colab runs Python 3.12, so `mim install mmcv` falls
  into a failing source build. `pip install -U openmim` also downgrades setuptools and breaks
  `pkg_resources` on 3.12.) We use **torchvision's `retinanet_resnet50_fpn`** — same architecture,
  preinstalled, **zero installs**.
- Print the python / torch / torchvision versions and confirm CUDA is visible; install `pycocotools`
  only if missing. Freeze versions into `requirements.lock.txt`.
- **GATE:** `tests/test_tv_model.py` passes — RetinaNet+SAS builds, the SAS weights are shared across
  pyramid levels, a training step yields finite losses with gradients reaching the SAS block, and
  inference returns detections.
- **RECORD:** the version list (python, torch, torchvision, pycocotools) in `results.md`.

---

## Phase 2 — Data acquisition (one-time)

- Download **TBX11K** from the links in the official repo (https://github.com/yun-liu/Tuberculosis)
  into Colab's **ephemeral `/content`** disk (~100GB), **not** Google Drive — the raw set is ~tens of
  GB and free Drive is only 15GB. Raw data is consumed by Phase 3 and then discarded with the
  session; only the compact 512² output is persisted to Drive.
- Map the dataset's on-disk layout: the image folders (healthy / sick / tb) and the **annotation
  format** — expect VOC-style XML and/or JSON giving TB **bounding boxes with active/latent type** —
  and the official **train / val / test list files**.
- Identify the **TB subset**: the ~1,200 TB images (active, latent, both, uncertain) and their box
  annotations. Determine which TB images are **train** vs **val** per the official split.
- Note that **test-set ground truth is withheld** (online challenge) — we will only ever evaluate on
  **val**, so you do not need test labels.
- **GATE:** you can enumerate every TB image and its boxes, and your per-split TB counts are consistent
  with [paper.md](paper.md) Table 2 (TB **train ≈ 600**, **val ≈ 200**; active≫latent).
- **RECORD:** the per-class TB counts you actually found for train and val, and the raw dataset location.

---

## Phase 3 — Build the compact TB-only COCO dataset (512²)

- **Filter** to TB images only (drop the 10,000 healthy / sick-non-TB images — out of scope for the PoC).
- **Resize** each image to **512×512** and **scale every bounding box** by the same x- and y-factors.
  Visually **spot-check** several resized images with their boxes overlaid to confirm the boxes still
  line up (this is the cheapest place to catch a coordinate bug).
- **Convert** the annotations to **COCO JSON**:
  - Categories = {**active TB**, **latent TB**}.
  - Produce **separate `train` and `val` JSON files** following the official TB split.
  - Also define a **category-agnostic view** (all TB boxes collapsed to a single "TB" class) to be used
    at evaluation time — this is the primary metric.
- **Save** the compact dataset (resized images + JSONs) to Drive. Confirm total size is only a few
  hundred MB (it must comfortably fit free Drive alongside checkpoints).
- **GATE:** a COCO loader opens both JSONs; image and annotation counts match Phase 2; the overlay
  visualization looks correct.
- **RECORD:** dataset path, on-disk size, and final image/box counts per split into `results.md`.

---

## Phase 4 — Smoke test (before any real training)

- Configure a **minimal RetinaNet** (ResNet-50, ImageNet-pretrained) pointed at the compact COCO
  dataset: input **512²**, tiny batch (2), **1 epoch**, on just a **handful of images**.
- Run **one training iteration** (forward + backward) and then **one evaluation pass** to produce a
  (deliberately poor) AP number.
- **GATE:** the whole path runs end-to-end with no crash and prints a **finite AP/AP50**. Fix any data,
  config, or path problems **here**, where iteration is cheap — not during the real runs.
- **RECORD:** confirmation the pipeline is green; note any config fixes made.

---

## Phase 5 — RetinaNet baseline (the reference number)

- Full config: **ResNet-50 + FPN (C=256)**, input **512²**, augmentation = **random horizontal flip**,
  optimizer **SGD**, **batch 8** (fallback to 4 or 2 if the T4 OOMs, scaling the learning rate linearly
  with batch size), **24 epochs**, the **fixed seed**. Train on **TB-train**.
- Enable **per-epoch checkpointing to Drive** and **auto-resume from the latest checkpoint**, so a
  disconnect only costs the current epoch.
- Evaluate on **TB-val**: **category-agnostic AP and AP50** (collapse categories at eval). Optionally
  also report per-type active/latent AP.
- **GATE:** all 24 epochs complete (across multiple sessions if needed) and the val AP is stable
  (not still climbing steeply / not diverging).
- **RECORD:** baseline **AP50 / AP** on TB-val in `results.md`. **This is the reference the primary claim
  is measured against.**

---

## Phase 6 — Implement the SAS module (the only novel code)

Implement the **Symmetric Abnormity Search (SAS)** module and insert it between FPN and the detection
head. Reuse mmdetection / the deformable-attention op wherever possible; the SAS wrapper is the only
genuinely new code. Follow the contract in [paper.md](paper.md) §3.2 and [CLAUDE.md](CLAUDE.md) §2.

- **SAS structure:** `SPE → SymAttention → FFN`, applied after **each** FPN level, with the module's
  **weights shared across all levels**.
- **SPE (Symmetric Positional Encoding):** build an absolute **sine/cosine** positional encoding for a
  feature map; split it into left/right halves at the **vertical centerline**; use an **STN** to predict
  an **affine transform initialised to the identity**, apply it to the right half, then **horizontally
  flip** and **concatenate** to form `P_sym`; recalibrate the feature as `F = F + P_sym`.
  Expose two config flags: **transfer direction** (`right→left` default, or `left→right`) and
  **STN on/off**.
- **SymAttention (Symmetric Search Attention):** deformable-DETR-style sparse sampling with
  **M = 8 heads** and **K = 4 sample points**, but the sampling **x-coordinates are mirrored across the
  centerline (x → W − x)** before sampling; follow with a **residual connection + MLP**. Expose an
  **attention-type** flag: `none` / `vanilla` (plain deformable attention, no mirroring) / `symattention`.
- **FFN:** a standard feed-forward block after the attention.
- **Required unit tests** (the mirror-coordinate math is the easy thing to get wrong — test on toy
  tensors you can check by hand):
  1. Reflection: mirroring a small known tensor across the centerline matches the hand-computed result.
  2. Identity STN: with the STN at its identity initialization, SPE leaves the positional encoding
     unchanged.
  3. Shape: each SAS output feature map has the **same shape** as its input.
  4. Gradients: gradients flow through the whole SAS module (no detached/blocked path).
- **GATE:** all four unit tests pass, and SAS plugs into the FPN→RetinaNet-head pipeline with a working
  forward pass.
- **RECORD:** unit-test results and any deviation from the paper's description (with the reason).

---

## Phase 7 — SymFormer w/ RetinaNet (primary result)

- Use the **exact same training recipe as Phase 5** (SGD, batch 8, 24 epochs, 512², random flip, same
  seed) but with the **SAS module inserted** between FPN and the RetinaNet head. Configure SAS in its
  full setting: **SymAttention + SPE (with STN), right→left**.
- Checkpoint / auto-resume to Drive; evaluate on **TB-val** (category-agnostic AP/AP50).
- **GATE (primary claim):** SymFormer's **AP50 > the Phase 5 baseline AP50**. (Direction is the pass
  condition; also note whether AP improves.)
- **RECORD:** SymFormer **AP50 / AP** and the **delta vs baseline** in `results.md`.

---

## Phase 8 — Table 8 ablation

- Sweep the matrix: **attention ∈ {none, vanilla, SymAttention}** × **PE ∈ {none, APE, RPE, SPE-w/o-STN,
  SPE}** × **symmetry ∈ {left→right, right→left}** (the symmetry flag applies only to the SPE variants).
  Train each cell on **TB-train**, evaluate on **TB-val**, using the **same schedule and seed** throughout.
- Run the **most informative cells first** so a partial run is still meaningful:
  1. `none / none` (this equals the Phase 5 baseline),
  2. `vanilla / APE`,
  3. `SymAttention / APE`,
  4. `SymAttention / SPE / right→left` (the full model),
  5. then the **±STN** pair and the **left→right vs right→left** pair.
- **Expected ordering** (from [paper.md](paper.md) Table 8): SymAttention > vanilla at matched PE;
  SPE > APE > RPE; STN adds a bump; right→left ≥ left→right. **With reduced data the ordering can be
  noisier than the paper** — if time permits, re-run the key cells with **2–3 different seeds** and report
  the spread to judge whether a small gap is real.
- **GATE:** every planned cell has a recorded AP50/AP.
- **RECORD:** the complete ablation table in `results.md`, mirroring the layout of paper Table 8.

---

## Phase 9 — Optional checkpoint sanity check

- **Only if** the authors have released a SymFormer checkpoint compatible with our eval setup: run it
  **inference-only** on **TB-val** to confirm our evaluation pipeline lands in the paper's ballpark.
- Keep this number clearly **separate** from our own trained results — it validates the *evaluator*, not
  our training.
- **RECORD:** the checkpoint's val AP under our pipeline, labelled as an external sanity check.

---

## Phase 10 — Write-up

- Fill `results.md` with the three blocks: **baseline vs SymFormer** (primary comparison) and the
  **full ablation table**.
- Update [CLAUDE.md](CLAUDE.md) **§0 / §6 / §8** with the actual numbers and tick the status checklist.
- Write a short results summary covering: the primary comparison and its delta; whether the ablation
  ordering held; every deviation from the paper (batch size, versions, filled-in defaults); and the
  honest caveats from [limitations.md](limitations.md) — **val not test**, and **reduced data → trend,
  not absolute numbers**.

---

## Appendix — cross-cutting instructions

- **Persistence & resume:** checkpoint every epoch to Drive; on session start, resume from the latest
  checkpoint automatically; never keep the only copy of anything in Colab-local storage.
- **Reproducibility:** set and log the fixed seed for every run; log the pinned library versions
  (from `requirements.txt`) alongside each result.
- **OOM handling:** if batch 8 OOMs on the T4, halve the batch and **scale the learning rate linearly**,
  and record this as a deviation.
- **Missing paper details:** default to mmdetection RetinaNet settings (and Deformable-DETR sampling
  settings for the attention op), and note each such choice.
- **What to record per run:** run-name, full config summary, seed, versions, AP50, AP, wall-clock /
  epochs completed, and any anomalies.
- **When to stop and ask the user:** if a GATE cannot be passed after a genuine fix attempt; if the
  dataset structure differs materially from what Phase 2 assumes; if the free-tier GPU is unavailable
  for an extended period; or if the primary claim (Phase 7) fails to reproduce even the *direction* of
  the gain — surface it rather than tuning silently.
- **Pointers:** method & target numbers → [paper.md](paper.md); scope, architecture & recipe →
  [CLAUDE.md](CLAUDE.md); caveats → [limitations.md](limitations.md).
