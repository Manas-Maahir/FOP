# Table 8 ablation configs (auto-generated)

Baseline (No/No) = `../retinanet_r50_fpn_tbx11k_512.py` (no file here).

Suggested run order (most-informative first, per plan.md Phase 8):
1. baseline (retinanet config)
2. `abl_vanilla_ape.py`
3. `abl_symattention_ape.py`
4. `abl_symattention_spe_stn_r2l.py`  (= the full SymFormer neck)
5. STN pair: `abl_symattention_spe_nostn_r2l.py` vs `abl_symattention_spe_stn_r2l.py`
6. direction pair: `..._spe_stn_l2r.py` vs `..._spe_stn_r2l.py`
7. the remaining cells

All configs: `abl_vanilla_ape.py`, `abl_vanilla_rpe.py`, `abl_vanilla_spe_nostn_l2r.py`, `abl_vanilla_spe_nostn_r2l.py`, `abl_vanilla_spe_stn_l2r.py`, `abl_vanilla_spe_stn_r2l.py`, `abl_symattention_ape.py`, `abl_symattention_rpe.py`, `abl_symattention_spe_nostn_l2r.py`, `abl_symattention_spe_nostn_r2l.py`, `abl_symattention_spe_stn_l2r.py`, `abl_symattention_spe_stn_r2l.py`
