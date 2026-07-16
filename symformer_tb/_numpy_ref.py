"""NumPy reference implementation of the mirror-coordinate + positional-encoding math.

This exists so the *trickiest* part of the SAS block (the reflection used by SPE, which
plan.md flags as "the easy thing to get wrong") can be verified locally with only NumPy,
and so the Colab torch tests have an independent oracle to compare against numerically.

The functions here intentionally mirror ``symformer_tb/sas.py`` line-for-line in behaviour,
with the STN taken to be the identity (its initial state).
"""

from __future__ import annotations

import math
import numpy as np


def sine_cosine_pe_np(channels: int, height: int, width: int,
                      temperature: float = 10000.0) -> np.ndarray:
    """2-D absolute sine/cosine PE, shape [channels, height, width]. Mirrors sas.sine_cosine_pe."""
    pad = (4 - channels % 4) % 4
    c = channels + pad
    c_half = c // 2

    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, c_half, 2, dtype=np.float64) * (-math.log(temperature) / c_half))

    pe_y = np.zeros((height, c_half))
    pe_y[:, 0::2] = np.sin(y * div)
    pe_y[:, 1::2] = np.cos(y * div)

    pe_x = np.zeros((width, c_half))
    pe_x[:, 0::2] = np.sin(x * div)
    pe_x[:, 1::2] = np.cos(x * div)

    pe_y = np.broadcast_to(pe_y[:, None, :], (height, width, c_half))
    pe_x = np.broadcast_to(pe_x[None, :, :], (height, width, c_half))
    pe = np.concatenate([pe_y, pe_x], axis=-1)      # [H, W, c]
    pe = np.transpose(pe, (2, 0, 1))                # [c, H, W]
    if pad:
        pe = pe[:channels]
    return np.ascontiguousarray(pe)


def mirror_x_np(t: np.ndarray) -> np.ndarray:
    """Reflect across the vertical centerline (flip last axis)."""
    return t[..., ::-1].copy()


def _match_width_np(t: np.ndarray, target_w: int) -> np.ndarray:
    cur = t.shape[-1]
    if cur == target_w:
        return t
    if cur > target_w:
        return t[..., :target_w]
    pad = target_w - cur
    return np.pad(t, [(0, 0)] * (t.ndim - 1) + [(0, pad)])


def build_psym_np(pe: np.ndarray, direction: str = "r2l") -> np.ndarray:
    """Construct P_sym from an absolute PE with the STN as identity. pe is [C, H, W]."""
    w = pe.shape[-1]
    mid = w // 2
    left, right = pe[..., :mid], pe[..., mid:]
    if direction == "r2l":
        transferred = mirror_x_np(right)
        transferred = _match_width_np(transferred, left.shape[-1])
        return np.concatenate([transferred, right], axis=-1)
    else:
        transferred = mirror_x_np(left)
        transferred = _match_width_np(transferred, right.shape[-1])
        return np.concatenate([left, transferred], axis=-1)
