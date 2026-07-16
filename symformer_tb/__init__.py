"""symformer_tb — Symmetric Abnormity Search (SAS) module for TB detection.

This package implements the *only* novel component of SymFormer (see paper.md §3.2):
the SAS block = Symmetric Positional Encoding (SPE) + Symmetric Search Attention
(SymAttention) + a feed-forward network (FFN), applied to each FPN level with weights
shared across levels.

Everything here is framework-light: the modules are plain ``torch.nn`` and the deformable
sampling is done with ``F.grid_sample`` (no custom CUDA op required), so the unit tests run
on CPU and the same code runs on a Colab GPU. mmdetection integration lives in
``mmdet_plugin.py`` and is imported lazily so this package is usable without mmdet installed.

See plan.md (Phase 6) for the module contract and required tests.
"""

__all__ = [
    "SASBlock",
    "SPE",
    "DeformableSymAttention",
    "FFN",
    "STN",
    "sine_cosine_pe",
    "mirror_x",
]

# The torch modules are imported lazily so that the torch-free parts of this package
# (e.g. ``symformer_tb._numpy_ref`` used by the local math tests, and the data-prep tool)
# remain usable on machines without PyTorch installed.
try:  # pragma: no cover - depends on environment
    from .sas import (  # noqa: F401
        SASBlock,
        SPE,
        DeformableSymAttention,
        FFN,
        STN,
        sine_cosine_pe,
        mirror_x,
    )
except Exception as _e:  # torch not available
    import warnings

    warnings.warn(
        f"symformer_tb: torch modules unavailable ({_e!r}); "
        "only the torch-free helpers (_numpy_ref) are importable.",
        stacklevel=2,
    )
