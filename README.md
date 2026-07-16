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

## Repository layout
```
symformer_tb/            the ONLY novel code (the SAS block)
  sas.py                 SPE, SymAttention (grid_sample-based), FFN, SASBlock  [torch]
  mmdet_plugin.py        SASFPN neck = FPN + shared SAS; registers with mmdet  [torch+mmdet]
  _numpy_ref.py          numpy reference for the mirror/PE math (torch-free oracle)
configs/
  retinanet_r50_fpn_tbx11k_512.py          RetinaNet baseline (Phase 5)
  symformer_retinanet_r50_fpn_tbx11k_512.py  full SymFormer (Phase 7)
  ablation/              12 Table-8 ablation configs (auto-generated) + README
tools/
  prepare_tbx11k.py      TBX11K -> compact TB-only 512 COCO (Phase 3); has --selftest
  make_ablation_configs.py  regenerate the ablation configs
  train_runner.py        mmengine training runner (with --resume for T4 time-outs)
  test_runner.py         mmengine eval runner (COCO AP/AP50)
tests/
  test_numpy_ref.py      mirror/PE math — runs locally with numpy only
  test_sas.py            the 4 required SAS tests + numpy cross-checks — runs on Colab (torch)
notebooks/
  colab_runbook.ipynb    drives Phases 1-10 on Colab
```

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
Open `notebooks/colab_runbook.ipynb`, set the runtime to **GPU**, and run top-to-bottom. It installs
the mm stack, runs the SAS unit tests, builds the compact dataset from your TBX11K copy on Drive,
smoke-tests, then trains/evaluates the baseline, SymFormer, and the ablation. Training cells use
`--resume`, so after a session time-out just re-run the same cell.

## Status
- [x] Novel code implemented: SAS = SPE + SymAttention + FFN (`symformer_tb/`).
- [x] Verified **locally** (numpy/cv2): mirror-coordinate math (5/5) and data-prep (self-test).
- [x] Torch SAS unit tests written (the 4 required + numpy cross-checks) — run on Colab (Phase 6).
- [x] Configs (baseline, SymFormer, 12 ablation cells), runners, and Colab notebook ready.
- [ ] Colab: environment pinned, dataset built, smoke test, baseline, SymFormer, ablation (Phases 1-8).
- [ ] Results recorded in `results.md` (Phase 10).

## Key deviations from the paper (see code comments + `limitations.md`)
- Deformable sampling uses `F.grid_sample` instead of the Deformable-DETR CUDA op (no compile step).
- 1-class (category-agnostic) detector so COCO AP/AP50 is the primary metric directly.
- TB-only data at 512²; classification head + non-TB images out of scope; evaluated on **val**.
