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

## Stack: torch + torchvision (not mmdetection)

The paper uses **mmdetection**, but OpenMMLab publishes `mmcv` wheels only up to ~torch 2.1 / Python
3.11 and has been effectively unmaintained since 2023 — on Colab's Python 3.12, `mim install mmcv`
falls into a source build that fails. This project uses **torchvision's `retinanet_resnet50_fpn`**:
the same ResNet-50 + FPN + RetinaNet architecture, preinstalled on Colab, **zero installs**. The SAS
block is framework-agnostic pure torch and is reused verbatim. The trade-off — different anchor/loss/
NMS defaults, so absolute numbers won't line up with the paper's — is documented in
[limitations.md](limitations.md).

---

## Repository layout

```
symformer_tb/            the ONLY novel code (the SAS block)
  sas.py                 SPE, SymAttention (grid_sample-based), FFN, SASBlock   [pure torch]
  tv_model.py            RetinaNet-R50-FPN + shared SAS after the FPN           [torchvision]
  tv_dataset.py          COCO dataset + random hflip for the torchvision API
  _numpy_ref.py          numpy reference for the mirror/PE math (torch-free oracle)
tools/
  prepare_tbx11k.py      TBX11K -> compact TB-only 512² COCO; has --selftest
  tv_train.py            training loop; checkpoint + auto-resume for T4 time-outs
  tv_eval.py             COCO AP/AP50 via pycocotools
tests/
  test_numpy_ref.py      mirror/PE math — runs locally with numpy only
  test_sas.py            the 4 required SAS tests + numpy cross-checks           [Colab]
  test_tv_model.py       RetinaNet+SAS wiring, weight sharing, train/infer       [Colab]
notebooks/
  colab_runbook.ipynb    drives the full pipeline on Colab
```

The ablation is driven by **CLI flags**, not config files:
`--attention {vanilla,symattention} --pe {none,ape,rpe,spe} --stn/--no-stn --direction {r2l,l2r}`,
with `--no-sas` for the baseline.

---

## Running it

### Locally — verify the framework-independent logic (no GPU, no PyTorch)

```bash
pip install -r requirements.txt           # numpy / pillow / opencv / matplotlib only
python tests/test_numpy_ref.py            # mirror + positional-encoding math   (5 tests)
python tools/prepare_tbx11k.py --selftest # resize + box-scaling + COCO output  (synthetic)
```

These are the parts checkable without torch, and both pass. Everything torch-based (the SAS forward
pass, training, evaluation) runs on Colab.

### On Colab — the full pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manas-Maahir/FOP/blob/main/notebooks/colab_runbook.ipynb)

Open [`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb) (badge above), set the runtime
to **GPU**, and run top-to-bottom. It clones the repo, runs the SAS unit tests, builds the compact
dataset, smoke-tests, then trains and evaluates the baseline, SymFormer, and the ablation cells.
Training cells auto-resume within a session, so after a time-out just re-run the same cell. A full
24-epoch run is ~15–20 min.

**Storage split** — free Drive is 15 GB, but the Colab VM has ~100 GB of ephemeral disk:

| What | Where | Size |
|---|---|---|
| code / docs | GitHub → `/content/FOP` | ~200 KB |
| raw TBX11K | **`/content`** (ephemeral — never on Drive) | ~11–30 GB |
| compact TB-only 512² dataset | **Drive** | ~250–400 MB |
| checkpoints | **`/content/work`** (ephemeral — never on Drive) | ~300 MB transient |
| logs (`train_log.jsonl` / `eval_log.jsonl`) — *the results* | **Drive** | a few KB |

Weights are deliberately kept off Drive: Colab's Drive mount turns every delete into a move to
**Trash**, which counts against the 15 GB quota for 30 days, so pruning ~300 MB/epoch would silently
trash gigabytes. Because a run is only ~20 min and every AP/AP50 lands in the logs, the **logs are
the results** and the weights are disposable. Total Drive footprint: **~350 MB**.

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
