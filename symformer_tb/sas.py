"""Symmetric Abnormity Search (SAS) block and its parts.

Contract (see plan.md Phase 6 / paper.md §3.2):

    SAS(F) = FFN( SymAttention( SPE(F) ) )      # residual connections inside each part

applied to every FPN level with **shared weights**.

Flags exposed for the Table-8 ablation:
    attention : "vanilla" | "symattention"      (baseline = do not use SAS at all)
    pe        : "none" | "ape" | "rpe" | "spe"
    use_stn   : bool                            (only meaningful for pe="spe")
    direction : "r2l" | "l2r"                   (only meaningful for pe="spe"; paper default r2l)

Design notes / deviations from the paper (documented on purpose — see plan.md appendix):
  * Absolute PE is a standard 2D sine/cosine encoding (DETR-style: half the channels encode
    the row, half encode the column). The paper's Eq.1 is written for a 1-D position; the 2-D
    lift is the conventional one.
  * Deformable sampling uses ``F.grid_sample`` instead of the Deformable-DETR CUDA op. It is
    the single-scale equivalent and needs no compilation, which also removes a Colab build risk.
  * "rpe" is implemented as a learnable per-(head,point) relative-position bias added to the
    attention logits — a light, explicit stand-in for relative positional encoding.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Positional encoding + mirroring utilities
# --------------------------------------------------------------------------------------
def sine_cosine_pe(channels: int, height: int, width: int,
                   temperature: float = 10000.0,
                   device=None, dtype=torch.float32) -> torch.Tensor:
    """Standard 2-D absolute sine/cosine positional encoding.

    Returns a tensor of shape ``[1, channels, height, width]``. Half of ``channels`` encode the
    y (row) position and half encode the x (column) position, each split again into sin/cos.
    """
    if channels % 4 != 0:
        # fall back gracefully: pad to a multiple of 4 then crop
        pad = (4 - channels % 4) % 4
    else:
        pad = 0
    c = channels + pad
    c_half = c // 2  # channels for each of the two spatial axes

    y = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)  # [H,1]
    x = torch.arange(width, device=device, dtype=dtype).unsqueeze(1)   # [W,1]

    div = torch.exp(
        torch.arange(0, c_half, 2, device=device, dtype=dtype) * (-math.log(temperature) / c_half)
    )  # [c_half/2]

    pe_y = torch.zeros(height, c_half, device=device, dtype=dtype)
    pe_y[:, 0::2] = torch.sin(y * div)
    pe_y[:, 1::2] = torch.cos(y * div)

    pe_x = torch.zeros(width, c_half, device=device, dtype=dtype)
    pe_x[:, 0::2] = torch.sin(x * div)
    pe_x[:, 1::2] = torch.cos(x * div)

    # broadcast to [H, W, c]
    pe_y = pe_y.unsqueeze(1).expand(height, width, c_half)
    pe_x = pe_x.unsqueeze(0).expand(height, width, c_half)
    pe = torch.cat([pe_y, pe_x], dim=-1)          # [H, W, c]
    pe = pe.permute(2, 0, 1).unsqueeze(0)          # [1, c, H, W]
    if pad:
        pe = pe[:, :channels]
    return pe.contiguous()


def mirror_x(t: torch.Tensor) -> torch.Tensor:
    """Reflect a feature/PE map across the vertical centerline (flip the width axis).

    For a map of width ``W`` the column at index ``ix`` moves to index ``W-1-ix``.
    """
    return torch.flip(t, dims=[-1])


# --------------------------------------------------------------------------------------
# Spatial Transformer Network (predicts an affine, initialised to identity)
# --------------------------------------------------------------------------------------
class STN(nn.Module):
    """Predict an affine matrix conditioned on the input feature, initialised to identity.

    Two alternating (max-pool + Conv-ReLU) stages, adaptive-pool to a fixed grid, then an MLP
    that outputs the 6 affine parameters. The final linear layer is zero-initialised with an
    identity bias so that, at init, the predicted transform is the identity mapping.
    """

    def __init__(self, channels: int, hidden: int = 64, pooled: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(pooled)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * pooled * pooled, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
        )
        # identity initialisation
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        h = self.features(feat)
        h = self.pool(h).flatten(1)
        theta = self.mlp(h).view(b, 2, 3)
        return theta


# --------------------------------------------------------------------------------------
# Symmetric Positional Encoding
# --------------------------------------------------------------------------------------
class SPE(nn.Module):
    """Symmetric Positional Encoding + feature recalibration.

    pe="none": returns F unchanged.
    pe="ape" : F + absolute sine/cosine PE.
    pe="spe" : F + P_sym, where P_sym transfers one side of the absolute PE to the other via an
               STN affine (optional) + horizontal flip, then concatenates the two halves.
    pe="rpe" : returns F unchanged here (RPE is applied inside the attention as a bias).
    """

    def __init__(self, channels: int,
                 pe: Literal["none", "ape", "rpe", "spe"] = "spe",
                 use_stn: bool = True,
                 direction: Literal["r2l", "l2r"] = "r2l"):
        super().__init__()
        assert pe in ("none", "ape", "rpe", "spe")
        assert direction in ("r2l", "l2r")
        self.pe = pe
        self.use_stn = use_stn
        self.direction = direction
        self.stn = STN(channels) if (pe == "spe" and use_stn) else None

    def _affine_transform(self, half: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(theta, list(half.shape), align_corners=False)
        return F.grid_sample(half, grid, align_corners=False, padding_mode="border")

    def build_psym(self, feat: torch.Tensor) -> torch.Tensor:
        """Construct the symmetric positional encoding P_sym for a feature map ``feat``."""
        b, c, h, w = feat.shape
        pe = sine_cosine_pe(c, h, w, device=feat.device, dtype=feat.dtype)  # [1,c,h,w]
        pe = pe.expand(b, c, h, w)
        mid = w // 2
        left, right = pe[..., :mid], pe[..., mid:]

        # choose which side is the "source" that gets transferred to the other side
        if self.direction == "r2l":
            source = right
        else:
            source = left

        if self.stn is not None:
            theta = self.stn(feat)
            source = self._affine_transform(source, theta)
        transferred = mirror_x(source)  # horizontal flip

        # crop/pad transferred to the width of the destination half, then assemble full-width P_sym
        if self.direction == "r2l":
            # transferred fills the LEFT half; keep the real right half
            dest_w = left.shape[-1]
            transferred = self._match_width(transferred, dest_w)
            psym = torch.cat([transferred, right], dim=-1)
        else:
            dest_w = right.shape[-1]
            transferred = self._match_width(transferred, dest_w)
            psym = torch.cat([left, transferred], dim=-1)
        return psym

    @staticmethod
    def _match_width(t: torch.Tensor, target_w: int) -> torch.Tensor:
        cur = t.shape[-1]
        if cur == target_w:
            return t
        if cur > target_w:
            return t[..., :target_w]
        pad = target_w - cur
        return F.pad(t, (0, pad))  # pad on the right

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pe in ("none", "rpe"):
            return feat
        if self.pe == "ape":
            b, c, h, w = feat.shape
            pe = sine_cosine_pe(c, h, w, device=feat.device, dtype=feat.dtype)
            return feat + pe
        # spe
        return feat + self.build_psym(feat)


# --------------------------------------------------------------------------------------
# Symmetric Search Attention (deformable, via grid_sample)
# --------------------------------------------------------------------------------------
class DeformableSymAttention(nn.Module):
    """Deformable attention that samples around each location (vanilla) or around its
    bilaterally symmetric location (symattention).

    M heads, K points. Offsets and attention weights are predicted from the (recalibrated)
    feature map; values are sampled with ``F.grid_sample``.
    """

    def __init__(self, channels: int, num_heads: int = 8, num_points: int = 4,
                 attention: Literal["vanilla", "symattention"] = "symattention",
                 pe: Literal["none", "ape", "rpe", "spe"] = "spe",
                 offset_scale: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        assert attention in ("vanilla", "symattention")
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.attention = attention
        self.pe = pe
        self.offset_scale = offset_scale

        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.offset_proj = nn.Conv2d(channels, num_heads * num_points * 2, 1)
        self.attn_proj = nn.Conv2d(channels, num_heads * num_points, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # RPE: a learnable relative-position bias per (head, point), added to attn logits
        self.rel_bias = (nn.Parameter(torch.zeros(num_heads, num_points))
                         if pe == "rpe" else None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.zeros_(self.attn_proj.weight)
        nn.init.zeros_(self.attn_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _base_reference(h: int, w: int, device, dtype):
        """Normalised base reference grid in [-1, 1], shape [H, W, 2] as (x, y)."""
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype) if h > 1 else torch.zeros(1, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype) if w > 1 else torch.zeros(1, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1)  # [H, W, 2]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        m, k = self.num_heads, self.num_points
        hc = c // m

        value = self.value_proj(feat).view(b * m, hc, h, w)

        offsets = self.offset_proj(feat).view(b, m, k, 2, h, w)
        offsets = offsets.permute(0, 1, 2, 4, 5, 3)          # [b, m, k, h, w, 2]

        attn = self.attn_proj(feat).view(b, m, k, h, w)
        if self.rel_bias is not None:
            attn = attn + self.rel_bias.view(1, m, k, 1, 1)
        attn = torch.softmax(attn, dim=2)                     # softmax over K points

        ref = self._base_reference(h, w, feat.device, feat.dtype)  # [h, w, 2] as (x, y)
        ref = ref.view(1, 1, h, w, 2).expand(b, m, h, w, 2)
        if self.attention == "symattention":
            # reflect the x reference across the centerline: x -> -x in normalised coords
            ref = ref.clone()
            ref[..., 0] = -ref[..., 0]

        out = feat.new_zeros(b * m, hc, h, w)
        for kk in range(k):
            off = offsets[:, :, kk] * self.offset_scale                # [b, m, h, w, 2]
            loc = ref + off                                            # [b, m, h, w, 2]
            grid = loc.reshape(b * m, h, w, 2).clamp(-2.0, 2.0)
            sampled = F.grid_sample(value, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)  # [b*m, hc, h, w]
            weight = attn[:, :, kk].reshape(b * m, 1, h, w)
            out = out + sampled * weight

        out = out.view(b, c, h, w)
        out = self.out_proj(out)
        return feat + out  # residual (Eq. 8)


# --------------------------------------------------------------------------------------
# Feed-forward network
# --------------------------------------------------------------------------------------
class FFN(nn.Module):
    """Position-wise feed-forward block with a residual connection (1x1 convs)."""

    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# --------------------------------------------------------------------------------------
# The SAS block
# --------------------------------------------------------------------------------------
class SASBlock(nn.Module):
    """SPE -> SymAttention -> FFN, applied to a single feature map.

    A single instance is meant to be *shared* across all FPN levels (see SASFPN in
    mmdet_plugin.py); it works for any (H, W) because the PE and reference grids are built
    dynamically from the input shape.
    """

    def __init__(self, channels: int = 256,
                 num_heads: int = 8, num_points: int = 4,
                 attention: Literal["vanilla", "symattention"] = "symattention",
                 pe: Literal["none", "ape", "rpe", "spe"] = "spe",
                 use_stn: bool = True,
                 direction: Literal["r2l", "l2r"] = "r2l",
                 offset_scale: float = 0.1):
        super().__init__()
        self.spe = SPE(channels, pe=pe, use_stn=use_stn, direction=direction)
        self.attn = DeformableSymAttention(
            channels, num_heads=num_heads, num_points=num_points,
            attention=attention, pe=pe, offset_scale=offset_scale)
        self.ffn = FFN(channels)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.spe(feat)
        x = self.attn(x)
        x = self.ffn(x)
        return x
