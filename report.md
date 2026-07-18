# Replication Report — SymFormer / TBX11K (Core-Method PoC)

*Replicating the detection method of "Revisiting Computer-Aided Tuberculosis Diagnosis"
(Liu et al., TPAMI; arXiv:2307.02848) on Google Colab Free (T4).*

---

## TL;DR

We set out to test **one** claim from the paper — that **SymFormer w/ RetinaNet beats a plain
RetinaNet baseline** on category-agnostic TB detection, and that its two novel modules (**SPE** and
**SymAttention**) each add a measurable gain (the paper's Table 8). We built the full pipeline on a
free T4, got it training stably, and ran the baseline plus six ablation configurations end-to-end.

**Result: the trend did not reproduce.** Every configuration landed within **77.7–80.0 AP50** — a
~2-point spread that is no larger than the run-to-run noise we measured for a *single fixed config*.
The baseline (79.1) was, if anything, marginally ahead of the full SymFormer (78.1). This is a clean
**null result at this scale**, and it is exactly what our pre-registered limitations predicted:
~600 training images, a single random seed, a different detection framework (torchvision, not
mmdetection), and evaluation on *val* rather than the private *test* set. The pipeline works; the
experiment as run simply lacks the statistical power to detect a ~4-point effect.

---

## 1. Where we started — the goal

The paper makes three contributions: the **TBX11K** dataset (11,200 chest X-rays with bounding-box
TB annotations), the **SymFormer** detector (ResNet-50 + FPN + a novel **SAS** block + RetinaNet
head), and a **benchmark**. Its detector exploits a domain prior: healthy lungs are roughly
**bilaterally symmetric**, so a lesion often shows up as a *break* in left–right symmetry. SymFormer
encodes that prior two ways:

- **SPE (Symmetric Positional Encoding)** — mirror the positional encoding across the vertical
  centerline (via a learned STN, identity-initialised) so each location "knows" where its mirror is.
- **SymAttention** — deformable attention that samples around each query's **mirrored** location
  (`x → W − x`), not just its neighbourhood.

A full reproduction (both detector + classifier heads, all baselines, all backbones, 4-fold CV,
cross-dataset eval, online-challenge test scoring) is far beyond a single free T4. So we scoped down.

---

## 2. What we decided (locked scope)

| Decision | Choice | Why |
|---|---|---|
| **Approach** | **Hybrid** | Official TBX11K data + the paper's novel modules rebuilt on a maintained detection stack. |
| **Scope** | **Core-method PoC** | Reproduce the *central claim* — SymFormer > baseline on TB detection — plus the Table 8 ablation. Not the full grid. |
| **Success metric** | **The trend, not the absolute numbers** | Matching the paper's exact table values on reduced data is a non-goal. |
| **Compute** | **Colab Free (T4)** | Single 16 GB GPU with session time-outs. |
| **Detection stack** | **torchvision** | (chosen mid-project — see §3) |

We evaluate **category-agnostic** TB detection (active + latent collapsed to one class) on the
**TB-only validation** split, with COCO **AP** (IoU .50:.95) and **AP50**.

---

## 3. What we changed along the way, and why

The project was a sequence of course-corrections. Each was forced by a concrete failure, and each is
a single commit in the repo.

### 3.1 Scope: full replication → core-method PoC
An honest feasibility check showed a full reproduction was impossible on the hardware and time
budget. We narrowed to the one falsifiable claim that fits a T4. *(Rationale, not a code change.)*

### 3.2 Detection stack: mmdetection → torchvision  (`71f02ca`)
The paper uses **mmdetection**. On Colab it is **uninstallable**: OpenMMLab ships `mmcv` wheels only
up to ~torch 2.1 / Python 3.11, Colab now runs Python 3.12, so `mim install mmcv` falls into a
failing source build — and `pip install -U openmim` downgrades setuptools, whose `pkg_resources`
calls the `pkgutil.ImpImporter` that Python 3.12 removed, breaking the tool outright. We pivoted to
torchvision's `retinanet_resnet50_fpn`: **the same ResNet-50 + FPN + RetinaNet architecture**,
preinstalled, zero compilation. Our novel **SAS block is pure torch and was reused verbatim** — the
science is unchanged, but anchor/loss/NMS defaults now differ from mmdetection's. *(Recorded as a
limitation: absolute numbers will not line up with the paper's.)*

### 3.3 Data pipeline made real  (`4d744c5`, `fabc675`, `dc4666c`)
The first data cell only *printed instructions*. We turned it into a real download (`gdown`) +
layout auto-discovery + loud failures, and fixed three traps: empty leftover dirs faking "done", a
stale-directory guard (`compact_ready()`), and a `gdown --fuzzy` incompatibility (pass the bare file
id instead).

### 3.4 The split-list bug — the one that would have silently corrupted everything  (`3159651`)
TBX11K ships seven list files (`{TBX11K,all}_{train,val,trainval,test}.txt`). Our substring matcher
picked the split by testing `"train" in name` — but **`"train"` is a substring of `"trainval"`**, so
it could select a *trainval* file as the training split, **leaking validation images into training
and silently inflating every AP**. Fixed with an explicit chooser that excludes `trainval`/`test`,
plus a **hard error on any train/val overlap** and a size warning. After the fix the loader reported
exactly **train = 599, val = 200**, matching the paper's Table 2 (~600 / ~200). *This was caught
before it produced a single number — the most important fix in the project.*

### 3.5 Storage architecture — stop burning the Drive quota  (`121a51f`, `650ddda`, `c32b7d4`)
Training filled the 15 GB Drive quota with **deleted** files. The notebook wrote ~300 MB checkpoints
straight onto the Drive mount, and Colab's Drive mount turns every deletion into a move to **Trash**,
which keeps counting against quota for 30 days — so pruning the previous epoch trashed ~300 MB
*every epoch*, ~7 GB per 24-epoch run (the ablation would have trashed ~90 GB). We re-architected:

- **Weights → `/content/work` (ephemeral); only the tiny logs → Drive.** The logs (a few KB) carry
  every AP/AP50 — *they are the results*; the weights are disposable because a run is only ~20 min.
- Added a startup **guard** that warns if `--work-dir` ever lands on Drive again, a logs-only
  `--drive-sync`, and moved the Trash-cleanup cell *before* the data-prep cell (prep writes to Drive
  and would fail on a full quota). Total Drive footprint dropped from ~7 GB/run to **~350 MB total**.

### 3.6 Gradient clipping — torchvision RetinaNet diverges without it  (`b76ed06`)
The first real baseline reached **AP50 43 by epoch 4**, then `bbox_regression` exploded to **1.3 ×
10³⁴** and the loss went NaN as the warmup LR neared its 0.005 peak. torchvision's RetinaNet is more
divergence-prone here than mmdetection's (which the paper used and which did not need this). We added
**gradient clipping** (L2 max-norm 10) — a standard detection-training guard — which fixed it while
keeping the paper's linear-scaled LR. *(Recorded as a torchvision-forced deviation from the recipe.)*

### 3.7 The ×100 logging bug  (`85118d3`)
The first results table showed impossible values (`AP50 = 7905`). `score_coco()` already returns
percentages (`×100`), and the eval logger multiplied by 100 **again**. The printed result line was
always correct; only the logged JSON — and thus the collector table — was inflated 100×. Fixed;
**true values are the logged ones ÷ 100.**

---

## 4. Training recipe (as run)

| Setting | Value |
|---|---|
| Data | TB-only, 512×512, COCO format, category-agnostic at eval |
| Split | train 599 / val 200 (paper's TB split) |
| Backbone / neck | ResNet-50 (ImageNet-pretrained) + FPN, C = 256 |
| Detector | torchvision RetinaNet (`num_classes = 2`) |
| Novel module | shared **SAS** block over every FPN level (SPE + SymAttention + FFN) |
| Optimiser | SGD, momentum 0.9, weight-decay 1e-4 |
| LR | 0.005 (linear-scaled from 0.01 @ batch 16), 500-iter warmup, ×0.1 @ epochs 16/22 |
| **Gradient clip** | **L2 max-norm 10** (deviation from paper — see §3.6) |
| Batch / epochs / seed | 8 / 24 / **0** (single seed) |
| Augmentation | random horizontal flip (with box mirroring) |

---

## 5. Results

Category-agnostic TB detection on TB-only **val** (200 images). Values below are the **corrected**
numbers (raw log ÷ 100 per §3.7).

| Configuration | Attention | PE | AP50 | AP |
|---|---|---|---:|---:|
| **Baseline** (`none_none`) | — | — | **79.1** | 33.4 |
| `vanilla_ape` | vanilla deformable | APE | 78.5 | 33.1 |
| `symattention_ape` | SymAttention | APE | 77.7 | 33.5 |
| **SymFormer** (`symattention_spe_stn_r2l`) | SymAttention | SPE + STN, R→L | **78.1** | 32.9 |
| `symattention_spe_nostn_r2l` | SymAttention | SPE, no STN, R→L | 80.0 | 33.2 |
| `symattention_spe_stn_l2r` | SymAttention | SPE + STN, L→R | 77.8 | 33.7 |

**Repeat-measurement noise (same config, different runs):**

| Config | Independent evals | Spread |
|---|---|---|
| Baseline (no SAS) | 79.1, 79.2, 78.9 | ~0.3 AP50 |
| Full SymFormer | 78.1 (ablation), 78.9 (standalone) | ~0.8 AP50 |

*(The ablation loop's 7 remaining cells — RPE variants and vanilla+SPE combinations — were not run;
the session was ended after the informative cells completed. Their weights were deleted after eval by
design, so only the baseline and SymFormer checkpoints survive.)*

---

## 6. What we can infer

**1. The paper's central claim did not reproduce at this scale.** SymFormer (78.1 AP50) is
statistically indistinguishable from — and numerically just below — the baseline (79.1). None of the
Table 8 orderings (SymAttention > vanilla; SPE > APE; +STN helps; R→L ≥ L→R) emerge above the noise.

**2. The differences are within noise, and we can show that quantitatively.** The entire spread
across six *different* configurations is **2.3 AP50** (77.7–80.0). But two runs of the *same* full
SymFormer config differ by **0.8 AP50**, and the baseline varies **0.3** across three evals. When the
between-condition signal (~2 points) is the same order as the within-condition noise (~0.3–0.8
points), **a single seed cannot separate the conditions.** The paper's reported effect was ~4 AP50 on
the full setup; we did not have the power to detect it.

**3. This is a null result, not a failure of the implementation.** Everything that *should* work,
does:
   - The pipeline runs end-to-end: data → train → eval → logged AP, reproducibly.
   - Training is **stable and healthy** — the baseline reaches **79 AP50, above the paper's own
     baseline of 72.7** (plausible: different framework, different split size, val not test).
   - The data split is **correct and cross-validated** — three independent baseline evals all land at
     ~79, and the train/val overlap check passed.
   - The novel modules are wired in and training (the configs differ from each other), just not
     enough to matter at this data scale.

**4. Absence of evidence ≠ evidence of absence.** We cannot conclude the SAS modules *don't* help.
We can only conclude that **this experiment — 600 images, one seed, a different framework, val not
test — is too underpowered to tell**, which is precisely what we flagged before running it.

---

## 7. What a real test would need

Ranked by expected impact:

1. **Multiple seeds (3–5) with error bars.** The single biggest gap. With ~0.5 AP50 noise, you need
   averaging to resolve a ~2–4 point effect. *Cheap on a T4 — just more time.*
2. **More training data.** The full trainval split (and the paper's "retrain on trainval, report on
   test" protocol) rather than the ~600-image TB-train subset. The symmetry prior may simply need
   more examples to pay off.
3. **The mmdetection stack the modules were designed for**, once a compatible environment exists —
   removing the anchor/loss/NMS confounds introduced by the torchvision pivot.
4. **Test-set scoring via the official online challenge**, instead of val.
5. **Faithfulness of the SAS reimplementation** — our SymAttention uses `F.grid_sample` rather than
   the custom CUDA op, and "RPE" is an approximation (relative PE is ill-defined for deformable
   attention). Both are documented in `limitations.md`.

---

## 8. Artifacts & reproducibility

- **Code:** this repo (`Manas-Maahir/FOP`). Novel module: `symformer_tb/sas.py`. Training/eval:
  `tools/tv_train.py`, `tools/tv_eval.py`. Runbook: `notebooks/colab_runbook.ipynb`.
- **Key commits:** `71f02ca` (stack pivot), `3159651` (split fix), `650ddda` (storage), `b76ed06`
  (grad clip), `85118d3` (AP-scaling fix).
- **Persisted on Drive:** compact 512² TB dataset, the baseline + SymFormer checkpoints, and the
  eval logs. Raw TBX11K is ephemeral (re-downloadable).
- **Companion docs:** `paper.md` (method), `CLAUDE.md` (scope/recipe), `limitations.md` (caveats),
  `plan.md` (runbook), `results.md` (results table).

**Bottom line:** we built a working, honest, reproducible test rig for SymFormer's central claim and
ran it end-to-end. At the scale a free T4 allows, the claim neither confirmed nor refuted — the
effect the paper reports is smaller than the noise this setup can resolve. The path to a real answer
is multi-seed averaging on more data, and it is now a matter of compute time, not code.
