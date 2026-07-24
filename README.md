# SymFormer / TBX11K — Core-Method Replication

A reduced-scope, honest replication of the detection method from ***Revisiting Computer-Aided
Tuberculosis Diagnosis*** (Liu et al., TPAMI 2023; [arXiv:2307.02848](https://arxiv.org/abs/2307.02848)).
It tests the paper's **central claim** — that the novel **SAS** block (Symmetric Positional Encoding
+ Symmetric-Search Attention) improves tuberculosis detection over a plain RetinaNet — on the
**TB-only** subset of TBX11K, at 512×512, on a single free Colab **T4**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manas-Maahir/FOP/blob/main/notebooks/colab_runbook.ipynb)
&nbsp;·&nbsp; Paper: [arXiv:2307.02848](https://arxiv.org/abs/2307.02848)
&nbsp;·&nbsp; Official code/data: [yun-liu/Tuberculosis](https://github.com/yun-liu/Tuberculosis)

> **Design goal — the trend, not the absolute numbers.** Matching the paper's exact table values on
> a ~600-image subset with a different detection framework is a non-goal by construction. See
> [limitations.md](limitations.md).

---

## Headline result

The pipeline was built, trained stably on a free T4, and run end-to-end across the baseline and six
ablation configurations. **The paper's trend did not reproduce at this scale — a clean null result.**

| Configuration | Attention | Positional encoding | AP50 | AP |
|---|---|---|---:|---:|
| **Baseline** | — | — | **79.1** | 33.4 |
| `vanilla_ape` | vanilla deformable | APE | 78.5 | 33.1 |
| `symattention_ape` | SymAttention | APE | 77.7 | 33.5 |
| **SymFormer** | SymAttention | SPE + STN, R→L | **78.1** | 32.9 |
| `symattention_spe_nostn_r2l` | SymAttention | SPE, no STN, R→L | 80.0 | 33.2 |
| `symattention_spe_stn_l2r` | SymAttention | SPE + STN, L→R | 77.8 | 33.7 |

*Category-agnostic TB detection on the TB-only validation split (200 images).*

Every configuration lands within **77.7–80.0 AP50** — a ~2.3-point spread that is **no larger than
the run-to-run noise measured for a single fixed config** (~0.3 AP50 for the baseline across three
evals, ~0.8 for SymFormer). With one random seed, ~600 training images, a different detection stack,
and evaluation on *val* rather than the private *test* set, the setup is **underpowered to resolve
the ~4-point effect the paper reports** — exactly as the pre-registered limitations predicted. The
implementation is sound; the experiment as run simply lacks statistical power.

**Read the full write-up:** [report.md](report.md) — goal, every course-correction, results, and what
a conclusive test would require.

---

## The idea being tested

Healthy lungs are roughly **bilaterally symmetric**, so a lesion often appears as a *break* in
left–right symmetry. SymFormer encodes that prior in a shared **SAS block** applied after the FPN:

- **SPE (Symmetric Positional Encoding)** — mirror the positional encoding across the vertical
  centerline (via an identity-initialised STN) so each location "knows" where its mirror is.
- **SymAttention** — deformable attention that samples around each query's **mirrored** location
  (`x → W − x`), not just its local neighbourhood.

```
CXR 512²  ─►  ResNet-50 + FPN {P3..P7}  ─►  shared SAS block  ─►  RetinaNet head  ─►  boxes
                                             (SPE + SymAttention + FFN)
```

The **SAS block is the only novel code**; everything around it is a standard detector. The baseline
is the identical model with the SAS block removed.

---

## Two stacks: torchvision *and* mmdetection

The paper uses **mmdetection**. On Colab it is uninstallable — OpenMMLab ships `mmcv` wheels only up
to ~torch 2.1 / Python 3.11, and Colab runs 3.12, so `mim install mmcv` falls into a failing source
build. That forced the pivot to **torchvision's `retinanet_resnet50_fpn`**: the same
ResNet-50 + FPN + RetinaNet architecture, at the cost of different anchor/loss/NMS defaults
([limitations.md](limitations.md)).

Locally that constraint is gone. OpenMMLab publishes prebuilt **`win_amd64`** wheels, so mmdet
installs with **no compiler** given the right pins — which `scripts/setup_env.py` handles:

| | pin | why |
|---|---|---|
| Python | 3.11 | mmcv wheels cover cp38–cp311 only |
| torch / torchvision | 2.1.0+cu121 / 0.16.0+cu121 | last torch with an mmcv Windows wheel |
| mmcv | 2.1.0 | `mmdet/__init__.py` asserts `>=2.0.0rc4, <2.2.0` |
| mmdet | 3.3.0 | |

So `--stack {torchvision,mmdet}` selects the framework, both pinned to the *same* torch — a
difference between them is attributable to the detector, not the tensor library. mmdet's wheel also
ships the CUDA `MultiScaleDeformableAttention` op, retiring the `grid_sample` approximation.

mmdet is installed **last and optionally**: if it fails, the torchvision stack is already complete
and only `--stack mmdet` becomes unavailable. `SASBlock` is pure torch and is reused verbatim by
both — `SASBackbone` for torchvision, `SASFPN` for mmdet. The novel code has one implementation.

---

## Repository layout

```
symformer_tb/            the ONLY novel science (the SAS block); the rest is plumbing
  sas.py                 SPE, SymAttention, FFN, SASBlock                    [pure torch]
  adapters.py            one interface over torchvision + mmdet detectors
  trainer.py             YOLO-style loop: run dirs, progress bar, AMP, EMA, resume
  metrics.py             P/R/F1 sweep + COCO mAP + classification metrics    [numpy only]
  plotting.py            results.png, PR/F1 curves, confusion matrices, batch mosaics
  cls_head.py            stage-2 classification head on a frozen detector
  tv_model.py            RetinaNet-R50-FPN + shared SAS after the FPN        [torchvision]
  mmdet_plugin.py        SASFPN neck registered with mmdet's registry        [mmdet]
  tv_dataset.py          COCO detection + classification datasets
  evaluate.py            TB-only / all-images COCO scoring
  _numpy_ref.py          numpy oracle for the mirror/PE math (torch-free)
scripts/
  setup_env.py           builds the pinned .venv; stdlib-only, runs under any Python
  download_tbx11k.py     fetch + verify + extract the raw archive
tools/
  prepare_tbx11k.py      TBX11K -> compact 512² COCO (--scope tb|all); has --selftest
  make_dummy_data.py     synthetic fixture so the smoke test runs before any download
  train.py  val.py       stage-1 training and evaluation
  train_cls.py           stage-2 classification head
  ablate.py              Table 8 sweep over cells x seeds -> mean ± std
  tv_train.py tv_eval.py LEGACY -- kept so the Colab notebook still runs
tests/
  test_numpy_ref.py      mirror/PE math                            [numpy only]
  test_metrics.py        IoU/PR/F1/AUC, hand-computed              [numpy only]
  test_sas.py            the 4 required SAS tests + numpy cross-checks
  test_tv_model.py       RetinaNet+SAS wiring, weight sharing
  test_pipeline.py       adapters, dataset negatives, resume round-trip, cls head
notebooks/
  local_runbook.ipynb    the local pipeline -- drop it on a PC and Run All
  colab_runbook.ipynb    the original Colab PoC
```

The ablation is driven by **CLI flags**, not config files:
`--attention {vanilla,symattention} --pe {none,ape,rpe,spe} --stn/--no-stn --direction {r2l,l2r}`,
with `--no-sas` for the baseline. (`configs/` holds the equivalent mmdet configs.)

---

## Running it

### Locally — the full pipeline, including the whole dataset (recommended)

Open [`notebooks/local_runbook.ipynb`](notebooks/local_runbook.ipynb), pick **any** Python kernel,
and Run All. There is no setup step to do first.

The notebook builds its own pinned `.venv` (fetching a standalone Python 3.11 if the machine has
none), so **the kernel's Python version is irrelevant** — it runs every real step as a subprocess
and uses only the standard library itself. Needs Windows + an NVIDIA GPU (≥6 GB) + ~70 GB free.

It runs a **full smoke test on generated data before downloading anything**: build → train →
checkpoint → kill and resume → evaluate → plots, in about a minute. If that cell is green, the rest
is data and patience.

| Phase | Cold machine |
|---|---|
| setup + unit tests + smoke test | ~10 min |
| download + prepare all 11,200 images | 1–3 h |
| baseline + SymFormer (24 epochs each) | ~30 min |
| stage-2 classifier + all-images eval | ~30 min |
| Table 8 ablation (opt-in, 13 cells × seeds) | overnight |

Scripts can also be driven directly:

```bash
python scripts/setup_env.py                                   # build .venv (idempotent)
python tools/train.py --data-root data/tbx11k_512 --no-sas    # baseline
python tools/train.py --data-root data/tbx11k_512 \
    --attention symattention --pe spe --stn --direction r2l   # SymFormer
python tools/val.py --weights runs/detect/train/weights/best.pt \
    --data-root data/tbx11k_512 --mode all --cls-ckpt ...     # all-images eval
python tools/ablate.py --data-root data/tbx11k_512 --seeds 0 1 2
```

Each run writes a self-contained directory:

```
runs/detect/symformer/
  weights/{best,last}.pt   results.csv   results.png       args.yaml
  PR_curve.png   F1_curve.png   confusion_matrix.png
  labels.jpg     train_batch0.jpg        val_batch0_pred.jpg
```

### Without a GPU or PyTorch — verify the framework-independent logic

```bash
pip install -r requirements.txt           # numpy / pillow / matplotlib only
python tests/test_numpy_ref.py            # mirror + positional-encoding math
python tests/test_metrics.py              # IoU / PR / F1 / AUC, hand-checked
python tools/prepare_tbx11k.py --selftest # resize + box-scaling + COCO output (synthetic)
```

### On Colab — the original reduced-scope PoC

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manas-Maahir/FOP/blob/main/notebooks/colab_runbook.ipynb)

[`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb) still works and is the record of how
the results above were produced. It is TB-subset-only and torchvision-only, and its architecture is
shaped entirely by Colab's constraints — ephemeral `/content`, a 15 GB Drive quota, weights
deliberately kept off Drive because Colab's mount turns every delete into a quota-consuming Trash
move. None of that applies locally, which is why the local runbook exists.

---

## Key deviations from the paper

All are documented in code comments and [limitations.md](limitations.md):

- **torchvision instead of mmdetection** — same architecture, but different anchor/loss/NMS defaults,
  so absolute numbers differ from the paper's.
- **Gradient clipping (L2 max-norm 10)** — torchvision's RetinaNet diverges without it at the paper's
  linear-scaled LR; mmdetection's did not need it.
- Deformable sampling uses `F.grid_sample` rather than the Deformable-DETR CUDA op (no compile step).
- 1-class (category-agnostic) detector, so COCO AP/AP50 is the primary metric directly.
- TB-only data at 512²; the classification head and the 10k non-TB images are out of scope;
  evaluated on **val** (test GT is private).
- "RPE" is approximated as a learnable per-(head, point) attention bias.

---

## Documentation

| File | Purpose |
|---|---|
| [report.md](report.md) | **The write-up** — journey, decisions, results, and inference. Start here. |
| [paper.md](paper.md) | What the paper does; method details; target numbers (Tables 2/7/8). |
| [CLAUDE.md](CLAUDE.md) | Locked scope, architecture, and training recipe. |
| [plan.md](plan.md) | Step-by-step execution runbook. |
| [limitations.md](limitations.md) | Caveats of the paper and of this replication. |
| [results.md](results.md) | Raw results log. |

---

## Citation

```bibtex
@article{liu2023revisiting,
  title   = {Revisiting Computer-Aided Tuberculosis Diagnosis},
  author  = {Liu, Yun and Wu, Yu-Huan and Zhang, Shi-Chen and Liu, Li and Wu, Min and Cheng, Ming-Ming},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2023}
}
```

This repository is an independent, reduced-scope replication for study purposes and is not affiliated
with the paper's authors.
