# CLAUDE.md — Replication Plan: SymFormer / TBX11K

This file is the working guide for **replicating** *Revisiting Computer-Aided Tuberculosis
Diagnosis* (Liu et al., TPAMI; arXiv:2307.02848). Read [paper.md](paper.md) for what the paper
does and [limitations.md](limitations.md) for caveats. **No training code exists yet** — this
documents the plan we agreed on before coding.

> Official code & data: https://github.com/yun-liu/Tuberculosis

---

## 0. Decisions (locked)
| Decision | Choice | Why |
|---|---|---|
| **Approach** | **Hybrid** | Use the official TBX11K dataset; build SymFormer's novel modules on a maintained detection stack rather than running the authors' old code verbatim or rewriting everything from scratch. |
| **Detection stack** | **both**, via `--stack` | *Colab era:* torchvision only — mmcv ships wheels only up to ~torch 2.1 / Py 3.11 and Colab is on Py 3.12, so `mim install mmcv` fell into a failing source build. *Local era:* OpenMMLab publishes prebuilt **`win_amd64`** wheels, so mmdet installs with no compiler given Py 3.11 + torch 2.1.0 + mmcv 2.1.0. Both stacks are pinned to the **same torch**, so a difference between them is attributable to the detector framework. The SAS block is pure torch and is shared verbatim. |
| **Scope** | **Core method + stage 2** | *Colab era:* core-method PoC only. *Local era:* the full 11,200-image dataset, so the **stage-2 classification head** (Tables 3–6, incl. specificity) and the **all-images evaluation mode** are in scope. Stage-1 detection still trains on TB images only — that is the paper's own §3.4 choice, not a compute compromise. |
| **Compute** | **local RTX 3070 Ti (8 GB)**, was Colab T4 | No session time-outs and 120 GB of disk, so the Drive-quota architecture is gone. Half the T4's VRAM, so AMP is on by default and batch 8 is near the ceiling. Multi-seed ablation is now affordable — that is the fix [report.md](report.md) §7 asks for. |

**Success = the trend, not the exact numbers.** We aim to show SymFormer beats the RetinaNet
baseline on category-agnostic TB detection (AP50/AP) and that **SPE** and **SymAttention** each
add gains — measured on the **TB-only validation** set. Matching the paper's absolute table
values is a non-goal (see [limitations.md](limitations.md)).

---

## 1. What we are (and aren't) building

**In scope (Tier 0 + Tier 1):**
- Data pipeline for the **TB subset only** (~1,200 CXRs), resized to **512×512**, in **COCO format**.
- **RetinaNet baseline** (ResNet-50 + FPN), stage-1 detection training.
- **SymFormer w/ RetinaNet** = RetinaNet + the **SAS** module (**SPE** + **SymAttention** + FFN).
- The **Table 8 ablation matrix**.
- COCO **AP / AP50** evaluation for category-agnostic TB on **TB-only val**.

**Out of scope (optional Tier-2 extensions, noted but not built now):**
- The classification head + the 10,000 non-TB images (→ specificity / false-positive filtering / Tables 3–6).
- Other baselines (SSD, Faster R-CNN, FCOS, Deformable DETR), the **P2T-Small** backbone.
- 4-fold cross-validation, cross-dataset eval, and **online-challenge** test-set scoring.

---

## 2. Target architecture (what to implement)

```
CXR 512×512
  └─ ResNet-50 backbone ── FPN (C=256) ─> {F1,F2,F3,F4}
        each Fi ─> SAS module (weights SHARED across levels)
                     ├─ SPE: absolute sin/cos PE
                     │        → STN affine (init = identity)
                     │        → mirror Right→Left across vertical centerline + h-flip
                     │        → concat = P_sym ; recalibrate  F = F + P_sym
                     ├─ SymAttention: deformable sampling (M=8 heads, K=4 points)
                     │        sampled AROUND the bilaterally-symmetric (mirrored) location
                     │        → residual + MLP
                     └─ FFN
        enhanced {F̂1..F̂4} ─> RetinaNet detection head (active TB / latent TB)
```

- **Baseline** = the same minus the SAS module (plain RetinaNet on ResNet-50+FPN).
- The SAS module is the *only* novel code; everything else comes from mmdetection.
- SymAttention can reuse Deformable DETR's CUDA `MultiScaleDeformableAttention` op, modified so
  reference points are reflected across the centerline (`x → W − x`) before sampling.

---

## 3. Training recipe (paper's stage-1 detection settings)
| Setting | Value |
|---|---|
| Images | **TB only** (active/latent/both/uncertain) |
| Input size | 512 × 512 |
| Augmentation | random horizontal flip |
| Optimizer | **SGD** (RetinaNet & SymFormer-w/-RetinaNet) |
| Batch size | 8 (drop to 4–2 if T4 OOM; scale LR accordingly) |
| Epochs | 24 |
| FPN channels | C = 256 |
| SymAttention | M = 8, K = 4 |
| Seed | fixed (record it) |

> Stage 2 (classification head) is **skipped** in this PoC. The classifier only filters false
> positives for the *all-images* evaluation mode; our PoC evaluates **TB-only**, so it isn't needed.

---

## 4. Execution phases (designed for T4 time-outs)
Run sequentially; each phase must finish (or checkpoint) before the next.

1. **Smoke test** — a handful of TB images, 1 epoch, tiny batch. Prove the data loads, the model
   does one forward+backward, and one AP number comes out end-to-end. **Do this before anything else.**
2. **Baseline** — train RetinaNet (ResNet-50) on TB-train; eval AP/AP50 on TB-val. Record numbers.
3. **SymFormer** — add the SAS module; train; eval. **Compare to baseline → primary result.**
4. **Ablation (Table 8)** — sweep `attention ∈ {none, vanilla, SymAttention}` ×
   `PE ∈ {none, APE, RPE, SPE-no-STN, SPE}` × `symmetry ∈ {L→R, R→L}`. Train on TB-train, eval on
   TB-val. (Run the most informative cells first: none/none, vanilla/APE, SymAttention/SPE-R→L.)
5. **(Optional) Sanity check** — run inference with the authors' released checkpoint to validate
   the eval pipeline independently of our training.

---

## 5. Environment & data (Colab Free T4)

**Storage split (free Drive is 15GB; the Colab VM has ~100GB of ephemeral disk):**

| What | Where | Why |
|---|---|---|
| code / configs / docs | **GitHub** (cloned to `/content/FOP`) | tiny, versioned |
| raw TBX11K (~tens of GB) | **`/content`** (ephemeral) | download → prep → discard; never on Drive |
| compact TB-only 512² dataset (~few hundred MB) | **Drive** | expensive to rebuild (needs the raw set back) |
| checkpoints (~300MB/epoch) | **`/content/work`** (ephemeral) | a full run is only ~15–20 min, so retraining is cheaper than storing — and Drive deletions go to Trash, which keeps counting against the quota |
| logs (`train_log.jsonl` / `eval_log.jsonl`, KB) — **the results** | **Drive** | tiny, and they carry every AP/AP50 |

> **Weights never go on Drive.** Colab's Drive mount turns every delete into a move to Drive's
> **Trash**, and trashed files count against the 15GB quota for 30 days. Since training writes
> ~300MB per epoch and prunes the previous one, a 24-epoch run silently trashes ~7GB — and the
> Table 8 ablation would trash ~90GB. Point `--work-dir` at `/content/work/<run>` and pass
> `--drive-sync <drive_dir>` to copy just the logs. `tv_train.py` warns loudly if `--work-dir`
> lands on Drive.
>
> `max_keep_ckpts=1` and the ablation loop's post-eval delete now bound only `/content` (~66GB
> free), not Drive. Total expected Drive usage: **~350MB**.
>
> Cost of this: cross-session resume is gone (weights die with the session). That's ~20 min of
> retraining versus 7GB of quota — worth it. `last.pth` still gives within-session resume. The
> headline number is the **final-epoch AP**, which is already in the logs, so the weights are
> genuinely disposable; `--sync-weights` opts one model back onto Drive if you want it for a figure.

**Setup checklist (per session):**
1. `nvidia-smi` — confirm a GPU is attached (free tier may deny one).
2. Mount Drive.
3. Install a **mutually compatible** torch / CUDA / **mmcv** / **mmdetection** stack and
   **record the exact versions** in this repo (a `requirements.txt` / setup cell). Version drift
   between torch, mmcv, and mmdet is the most common failure — pin it once it works.
4. Build/verify the deformable-attention CUDA op.

**Data pipeline:**
1. Download TBX11K from the official repo's links (one-time; needs a machine/Drive with room —
   the raw set is ~tens of GB at 3000²).
2. **Extract the TB subset only** and **pre-resize once to 512×512** → a compact copy
   (a few hundred MB) stored on Drive. Scale bounding boxes by the same factor.
3. Produce/keep annotations in **COCO format** with the **TB-train / TB-val** split from the paper.
4. For category-agnostic evaluation, collapse active+latent into a single class at eval time.

---

## 6. How we verify (definition of done for the PoC)
- **Pipeline:** the smoke test produces a finite AP number without crashing.
- **Primary claim:** SymFormer w/ RetinaNet shows **higher** category-agnostic AP50 (and AP) than
  the RetinaNet baseline on TB-only val (paper's val ablation: 72.7 → 76.6 AP50). Direction matters
  more than the absolute gap.
- **Mechanism:** in the ablation, **SymAttention > vanilla attention** and **SPE > APE > RPE**, with
  the STN adding a further bump and **R→L** ≥ L→R.
- Record all runs (config, seed, versions, AP/AP50) in a results table committed to the repo.

---

## 7. Notes & conventions
- Keep the SAS module isolated and unit-testable (mirror-coordinate math is the easy thing to get
  wrong — test it on a toy tensor where you can hand-check the reflection).
- Log GPU memory; if batch 8 OOMs on the T4, reduce batch and learning rate together.
- When in doubt about a detail not in the paper, prefer the **mmdetection RetinaNet defaults** and
  the **Deformable DETR** sampling defaults, and note the deviation.
- Update §0 and §6 with actual numbers as runs complete; this file is the living record.

---

## 8. Status

**Colab PoC — complete.** Full write-up in [report.md](report.md); headline in [README.md](README.md).
- [x] Paper read end-to-end; [paper.md](paper.md), this file, and [limitations.md](limitations.md) written.
- [x] Environment pinned on Colab (torchvision, after the mmdet pivot).
- [x] TB-subset compact dataset (512², COCO), train 599 / val 200.
- [x] Smoke test, RetinaNet baseline, SymFormer, 6 of 13 ablation cells.
- [x] **Result: null.** Every config within 77.7–80.0 AP50, a spread no larger than the
      single-config run-to-run noise. Underpowered, not broken — see [report.md](report.md) §6.

**Local pipeline — built, verification in progress.** Entry point:
[notebooks/local_runbook.ipynb](notebooks/local_runbook.ipynb) (drop on a PC, Run All).
- [x] `scripts/setup_env.py` — pinned Py 3.11 venv, both stacks, stdlib-only bootstrapper.
- [x] YOLO-style trainer: run dirs, progress bar, metric table, plots, AMP, EMA, SIGINT-safe resume.
- [x] `--stack {torchvision,mmdet}` behind one adapter interface; `SASFPN` restored from `0c294a0`.
- [x] `--scope all` prep: 11,200 images, all-images COCO JSONs, stage-2 class labels.
- [x] Stage-2 classification head + the classifier's false-positive veto in `val.py --mode all`.
- [x] `tools/ablate.py` — 13 cells × seeds, resumable, reports **mean ± std**.
- [x] Synthetic-fixture smoke test that runs *before* any download.
- [ ] Full dataset downloaded and prepared.
- [ ] Baseline + SymFormer re-run locally (reference: torchvision baseline ≈ 79.1 AP50).
- [ ] Stage-2 numbers (paper Table 3 shape).
- [ ] **Multi-seed ablation — the open question.** [report.md](report.md) §7 item 1.
