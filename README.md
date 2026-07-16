# SymFormer / TBX11K — Core-Method PoC

A reduced-scope replication of *Revisiting Computer-Aided Tuberculosis Diagnosis*
(Liu et al., TPAMI; arXiv:2307.02848). We reproduce the paper's **central claim** — that the
**SAS** block (Symmetric Positional Encoding + Symmetric Search Attention) improves TB detection
over a plain RetinaNet — on the **TB-only** subset of TBX11K, at 512², on a free Colab **T4**.

**Success = the trend, not the paper's absolute numbers.** See `limitations.md`.

## Document chain
| File | Purpose |
|---|---|
| [paper.md](paper.md) | What the paper does; method details; target numbers (Tables 2/7/8). |
| [CLAUDE.md](CLAUDE.md) | Locked scope, architecture, training recipe, status checklist. |
| [plan.md](plan.md) | Step-by-step execution runbook (Phases 0–10, each with a GATE). |
| [limitations.md](limitations.md) | Caveats of the paper and of this PoC. |
| [results.md](results.md) | Where run numbers get recorded. |

## Stack: torch + torchvision (no mmdetection)
The paper used mmdetection, but OpenMMLab publishes `mmcv` wheels only up to ~torch 2.1 / Python
3.11 and has been effectively unmaintained since 2023 — on Colab's Python 3.12 `mim install mmcv`
falls into a source build that fails. We use **torchvision's `retinanet_resnet50_fpn`**: the same
ResNet-50 + FPN + RetinaNet architecture, preinstalled on Colab, **zero installs**. The SAS block is
framework-agnostic pure torch, so the novel code is unchanged. See `limitations.md` for what this
costs us in comparability.

## Repository layout
```
symformer_tb/            the ONLY novel code (the SAS block)
  sas.py                 SPE, SymAttention (grid_sample-based), FFN, SASBlock  [pure torch]
  tv_model.py            RetinaNet-R50-FPN + shared SAS after the FPN          [torchvision]
  tv_dataset.py          COCO dataset + random hflip for the torchvision API
  _numpy_ref.py          numpy reference for the mirror/PE math (torch-free oracle)
tools/
  prepare_tbx11k.py      TBX11K -> compact TB-only 512 COCO (Phase 3); has --selftest
  tv_train.py            training loop; checkpoint + auto-resume for T4 time-outs
  tv_eval.py             COCO AP/AP50 via pycocotools
tests/
  test_numpy_ref.py      mirror/PE math — runs locally with numpy only
  test_sas.py            the 4 required SAS tests + numpy cross-checks   [Colab]
  test_tv_model.py       RetinaNet+SAS wiring, weight sharing, train/infer [Colab]
notebooks/
  colab_runbook.ipynb    drives Phases 1-10 on Colab
```

The ablation is driven by **CLI flags**, not config files:
`--attention {vanilla,symattention} --pe {none,ape,rpe,spe} --stn/--no-stn --direction {r2l,l2r}`,
with `--no-sas` for the baseline.

## Run it

### Locally (no GPU, no PyTorch needed) — verify the framework-independent logic
```
pip install -r requirements.txt          # numpy / pillow / opencv / matplotlib only
python tests/test_numpy_ref.py           # mirror + positional-encoding math  (5 tests)
python tools/prepare_tbx11k.py --selftest # resize + box-scaling + COCO output (synthetic)
```
These two are the parts that can be checked without torch/mmdet, and both pass. Everything torch/
mmdet (the SAS forward pass, training, evaluation) runs on Colab.

### On Colab — the full pipeline
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manas-Maahir/FOP/blob/main/notebooks/colab_runbook.ipynb)

Open `notebooks/colab_runbook.ipynb` (badge above), set the runtime to **GPU**, and run top-to-bottom.
It clones this repo, installs the mm stack, runs the SAS unit tests, builds the compact dataset,
smoke-tests, then trains/evaluates the baseline, SymFormer, and the ablation. Training cells use
`--resume`, so after a session time-out just re-run the same cell.

**Storage split** — free Drive is 15GB, but the Colab VM has ~100GB of ephemeral disk:

| What | Where | Size |
|---|---|---|
| code / configs / docs | GitHub → `/content/FOP` | ~200 KB |
| raw TBX11K | **`/content`** (ephemeral — never on Drive) | ~11–30 GB |
| compact TB-only 512² dataset | **Drive** | ~250–400 MB |
| checkpoints (baseline + SymFormer) + logs | **Drive** | ~600 MB + ~300 MB transient per ablation cell |
| **Drive total** | | **~1.5 GB peak** |

Checkpoints are ~300 MB each (model + optimizer), so configs keep only the latest
(`max_keep_ckpts=1`) and the ablation loop deletes weights after each cell is evaluated. Set
`DELETE_CKPTS_AFTER_EVAL = False` to keep them all (~5 GB).

## Status
- [x] Novel code implemented: SAS = SPE + SymAttention + FFN (`symformer_tb/`).
- [x] Verified **locally** (numpy/cv2): mirror-coordinate math (5/5) and data-prep (self-test).
- [x] Torch SAS unit tests written (the 4 required + numpy cross-checks) — run on Colab (Phase 6).
- [x] Configs (baseline, SymFormer, 12 ablation cells), runners, and Colab notebook ready.
- [ ] Colab: environment pinned, dataset built, smoke test, baseline, SymFormer, ablation (Phases 1-8).
- [ ] Results recorded in `results.md` (Phase 10).

## Key deviations from the paper (see code comments + `limitations.md`)
- **torchvision instead of mmdetection** (mmcv is unbuildable on Colab's Python 3.12) — same
  architecture, but anchors/loss/NMS defaults differ, so absolute numbers won't match the paper.
- Deformable sampling uses `F.grid_sample` instead of the Deformable-DETR CUDA op (no compile step).
- 1-class (category-agnostic) detector so COCO AP/AP50 is the primary metric directly.
- TB-only data at 512²; classification head + non-TB images out of scope; evaluated on **val**.
- "RPE" is approximated as a learnable per-(head, point) attention bias.
