"""Local, torch-free verification of the mirror + positional-encoding math.

Run with:  python -m pytest tests/test_numpy_ref.py     (or)   python tests/test_numpy_ref.py

These tests validate the reflection used by SPE — the piece plan.md flags as the easy thing to
get wrong — using only NumPy, so they run on the local machine without PyTorch/mmdet.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from symformer_tb._numpy_ref import (  # noqa: E402
    sine_cosine_pe_np, mirror_x_np, build_psym_np,
)


def test_mirror_reverses_columns():
    t = np.arange(3 * 2 * 4).reshape(3, 2, 4).astype(float)
    m = mirror_x_np(t)
    assert np.allclose(m[..., 0], t[..., 3])
    assert np.allclose(m[..., 1], t[..., 2])
    assert np.allclose(m, t[..., ::-1])


def test_pe_shape_and_range():
    pe = sine_cosine_pe_np(256, 16, 24)
    assert pe.shape == (256, 16, 24)
    assert pe.max() <= 1.0 + 1e-9 and pe.min() >= -1.0 - 1e-9


def test_psym_r2l_preserves_right_and_reflects_left():
    # even width -> exact bilateral symmetry expected
    pe = sine_cosine_pe_np(8, 5, 8)
    psym = build_psym_np(pe, direction="r2l")
    w = pe.shape[-1]
    mid = w // 2
    # right half is preserved
    assert np.allclose(psym[..., mid:], pe[..., mid:])
    # left column j equals the far-right column w-1-j  (transfer of right side, flipped)
    for j in range(mid):
        assert np.allclose(psym[..., j], pe[..., w - 1 - j]), f"mismatch at col {j}"
    # for even width P_sym is perfectly mirror-symmetric about the centerline
    assert np.allclose(psym, mirror_x_np(psym))


def test_psym_l2r_preserves_left_and_reflects_right():
    pe = sine_cosine_pe_np(8, 5, 8)
    psym = build_psym_np(pe, direction="l2r")
    w = pe.shape[-1]
    mid = w // 2
    assert np.allclose(psym[..., :mid], pe[..., :mid])
    for j in range(mid, w):
        assert np.allclose(psym[..., j], pe[..., w - 1 - j]), f"mismatch at col {j}"
    assert np.allclose(psym, mirror_x_np(psym))


def test_psym_odd_width_shapes_ok():
    pe = sine_cosine_pe_np(8, 4, 7)
    psym = build_psym_np(pe, direction="r2l")
    assert psym.shape == pe.shape


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} numpy-reference tests passed")


if __name__ == "__main__":
    _run()
