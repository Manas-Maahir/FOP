"""Torch unit tests for the SAS block (run on Colab, where PyTorch is installed).

Covers the four tests required by plan.md Phase 6:
  1. reflection on a toy tensor matches the hand-computed mirror,
  2. the identity-initialised STN leaves the positional encoding unchanged,
  3. each SAS output has the same shape as its input,
  4. gradients flow through the whole SAS block.
Plus: the torch PE / P_sym match the independent NumPy reference numerically, SymAttention differs
from vanilla attention, and one shared block works across differently-sized FPN levels.

Run:  python -m pytest tests/test_sas.py     (or)   python tests/test_sas.py
If torch is not installed the whole module is skipped (the math is still covered by
tests/test_numpy_ref.py).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    _HAS_TORCH = hasattr(torch, "nn") and hasattr(torch, "randn")
except Exception:
    _HAS_TORCH = False

if _HAS_TORCH:
    from symformer_tb.sas import (
        SASBlock, SPE, DeformableSymAttention, sine_cosine_pe, mirror_x,
    )
    from symformer_tb._numpy_ref import sine_cosine_pe_np, build_psym_np


def _skip_if_no_torch():
    if not _HAS_TORCH:
        print("SKIP: torch not available in this environment")
        return True
    return False


# 1 -----------------------------------------------------------------------------------
def test_reflection_toy_tensor():
    if _skip_if_no_torch():
        return
    t = torch.arange(1 * 1 * 2 * 4, dtype=torch.float32).view(1, 1, 2, 4)
    m = mirror_x(t)
    assert torch.allclose(m[..., 0], t[..., 3])
    assert torch.allclose(m[..., 3], t[..., 0])
    assert torch.allclose(m, torch.flip(t, dims=[-1]))


# 2 -----------------------------------------------------------------------------------
def test_identity_stn_leaves_pe_unchanged():
    if _skip_if_no_torch():
        return
    torch.manual_seed(0)
    feat = torch.randn(2, 8, 6, 8)
    spe_with_stn = SPE(8, pe="spe", use_stn=True, direction="r2l")
    spe_no_stn = SPE(8, pe="spe", use_stn=False, direction="r2l")
    with torch.no_grad():
        psym_stn = spe_with_stn.build_psym(feat)
        psym_plain = spe_no_stn.build_psym(feat)
    # the STN is identity-initialised, so it must not change P_sym
    assert torch.allclose(psym_stn, psym_plain, atol=1e-5), \
        (psym_stn - psym_plain).abs().max().item()


# 3 -----------------------------------------------------------------------------------
def test_output_shape_matches_input():
    if _skip_if_no_torch():
        return
    block = SASBlock(channels=32, num_heads=8, num_points=4,
                     attention="symattention", pe="spe", use_stn=True)
    for h, w in [(16, 16), (12, 20), (7, 9)]:
        x = torch.randn(2, 32, h, w)
        y = block(x)
        assert y.shape == x.shape, (y.shape, x.shape)


# 4 -----------------------------------------------------------------------------------
def test_gradients_flow():
    if _skip_if_no_torch():
        return
    block = SASBlock(channels=32, num_heads=8, num_points=4,
                     attention="symattention", pe="spe", use_stn=True)
    x = torch.randn(2, 32, 14, 14, requires_grad=True)
    y = block(x).sum()
    y.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    n_with_grad = sum(1 for p in block.parameters()
                      if p.grad is not None and torch.isfinite(p.grad).all())
    assert n_with_grad > 0


# 5 -----------------------------------------------------------------------------------
def test_torch_pe_matches_numpy():
    if _skip_if_no_torch():
        return
    pe_t = sine_cosine_pe(16, 9, 13).numpy()[0]
    pe_n = sine_cosine_pe_np(16, 9, 13)
    assert np.allclose(pe_t, pe_n, atol=1e-5)


# 6 -----------------------------------------------------------------------------------
def test_torch_psym_matches_numpy_ref():
    if _skip_if_no_torch():
        return
    torch.manual_seed(1)
    feat = torch.zeros(1, 16, 6, 10)  # feature content irrelevant when STN is off
    spe = SPE(16, pe="spe", use_stn=False, direction="r2l")
    with torch.no_grad():
        psym_t = spe.build_psym(feat).numpy()[0]
    pe_n = sine_cosine_pe_np(16, 6, 10)
    psym_n = build_psym_np(pe_n, direction="r2l")
    assert np.allclose(psym_t, psym_n, atol=1e-5)


# 7 -----------------------------------------------------------------------------------
def test_symattention_differs_from_vanilla():
    if _skip_if_no_torch():
        return
    torch.manual_seed(2)
    x = torch.randn(1, 32, 16, 16)
    van = DeformableSymAttention(32, attention="vanilla", pe="none")
    sym = DeformableSymAttention(32, attention="symattention", pe="none")
    sym.load_state_dict(van.state_dict())  # identical weights -> only the reference point differs
    with torch.no_grad():
        yv, ys = van(x), sym(x)
    assert not torch.allclose(yv, ys), "mirroring the reference point should change the output"


# 8 -----------------------------------------------------------------------------------
def test_shared_block_across_levels():
    if _skip_if_no_torch():
        return
    block = SASBlock(channels=32, attention="symattention", pe="spe")
    levels = [torch.randn(1, 32, s, s) for s in (64, 32, 16, 8)]
    outs = [block(l) for l in levels]  # same weights, different sizes (weight sharing)
    for o, l in zip(outs, levels):
        assert o.shape == l.shape


def _run():
    if _skip_if_no_torch():
        print("0 torch tests run (torch unavailable) — use tests/test_numpy_ref.py locally")
        return
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} torch SAS tests passed")


if __name__ == "__main__":
    _run()
