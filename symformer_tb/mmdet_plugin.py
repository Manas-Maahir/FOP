"""mmdetection integration for the SAS block.

Defines ``SASFPN`` — an FPN whose outputs are each passed through a *single, shared* SAS block
(SPE + SymAttention + FFN) — and registers it with the mmdet model registry so it can be used
from a config via ``neck=dict(type='SASFPN', ...)``.

This module is imported for its side effect (registration). It targets **mmdetection 3.x**
(``mmdet.registry.MODELS``, ``mmdet.models.necks.FPN``). It is written so that merely importing
``symformer_tb`` does not require mmdet; import this module explicitly (or set
``custom_imports = dict(imports=['symformer_tb.mmdet_plugin'], allow_failed_imports=False)`` in
the mmdet config) once mmdet is installed on Colab.

The baseline (paper Table 8 "No attention / No PE") uses a plain ``FPN`` neck, i.e. it simply
does not use ``SASFPN``.
"""

from __future__ import annotations

from typing import Optional

try:
    from mmdet.registry import MODELS
    from mmdet.models.necks import FPN
    _MMDET_OK = True
except Exception:  # pragma: no cover - mmdet only present on Colab
    _MMDET_OK = False


if _MMDET_OK:
    from .sas import SASBlock

    @MODELS.register_module()
    class SASFPN(FPN):
        """FPN + a shared Symmetric Abnormity Search block on every output level.

        Args:
            out_channels: FPN output channels (C in the paper, 256). Also the SAS width.
            sas: dict of SAS options, e.g.
                 ``dict(attention='symattention', pe='spe', use_stn=True, direction='r2l',
                        num_heads=8, num_points=4)``.
                 Set ``sas=None`` to fall back to a plain FPN (useful for A/B configs).
            All other args are forwarded to ``mmdet.models.necks.FPN``.
        """

        def __init__(self, *args, out_channels: int = 256,
                     sas: Optional[dict] = None, **kwargs):
            super().__init__(*args, out_channels=out_channels, **kwargs)
            if sas is None:
                self.sas = None
            else:
                self.sas = SASBlock(channels=out_channels, **sas)

        def forward(self, inputs):
            outs = super().forward(inputs)  # tuple of feature maps, one per pyramid level
            if self.sas is None:
                return outs
            # weights are shared: the *same* self.sas is applied to every level
            return tuple(self.sas(o) for o in outs)

else:  # pragma: no cover
    SASFPN = None  # type: ignore
