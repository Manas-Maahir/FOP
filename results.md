# Results log

> Fill this in as runs complete (plan.md Phases 5, 7, 8, 10). Success = the **trend**
> (SymFormer > baseline; SPE/SymAttention help), not the paper's absolute numbers. All numbers
> are **category-agnostic TB detection on TB-only val** unless noted (test GT is private).

## Run environment
- **Seed:** `0` (set in configs; keep constant across all runs)
- **Library versions:** _paste from `requirements.lock.txt` after Phase 1_ — torch `?`, mmcv `?`,
  mmdet `?`, mmengine `?`
- **GPU:** _e.g. Tesla T4 16GB_
- **Deviations from paper:** batch size `?` (paper 8), LR `?`; deformable sampling via
  `F.grid_sample` (not the CUDA op); 2-D sine/cos PE; 1-class (agnostic) detector; TB-only data
  at 512²; evaluated on **val** not test.

## Primary comparison (Phases 5 & 7)
| Run | Config | AP50 | AP | Notes |
|---|---|---|---|---|
| Baseline | `retinanet_r50_fpn_tbx11k_512.py` |  |  | reference |
| SymFormer | `symformer_retinanet_r50_fpn_tbx11k_512.py` |  |  | **Δ vs baseline = ?** |

**GATE (Phase 7):** SymFormer AP50 > baseline AP50 → _pass / fail_.

## Table 8 ablation (Phase 8)
Mirrors paper.md Table 8 (paper val numbers shown in parentheses for reference only).

| Attention | Positional Encoding | Symmetry | AP50 | AP | (paper AP50) |
|---|---|---|---|---|---|
| None | None | — |  |  | (72.7) |
| Vanilla | APE | — |  |  | (73.4) |
| Vanilla | RPE | — |  |  | (72.7) |
| Vanilla | SPE w/o STN | l→r |  |  | (74.0) |
| Vanilla | SPE w/o STN | r→l |  |  | (74.3) |
| Vanilla | SPE | l→r |  |  | (75.1) |
| Vanilla | SPE | r→l |  |  | (75.7) |
| SymAttention | APE | — |  |  | (74.9) |
| SymAttention | RPE | — |  |  | (73.6) |
| SymAttention | SPE w/o STN | l→r |  |  | (75.3) |
| SymAttention | SPE w/o STN | r→l |  |  | (75.5) |
| SymAttention | SPE | l→r |  |  | (76.3) |
| SymAttention | SPE | r→l |  |  | (76.6) |

**Expected trend:** SymAttention > Vanilla (matched PE); SPE > APE > RPE; STN adds a bump;
r→l ≥ l→r. Note noise from reduced data — re-run key cells with 2–3 seeds if time permits.

## Optional sanity check (Phase 9)
| Source | AP50 | AP | Notes |
|---|---|---|---|
| Authors' released checkpoint (inference only) |  |  | validates the eval pipeline |

## Full run log
| run-name | config | seed | versions | AP50 | AP | wall-clock / epochs | notes |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |
