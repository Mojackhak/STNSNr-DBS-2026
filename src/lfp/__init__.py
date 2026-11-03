"""LFP time-frequency and connectivity toolbox.

This package intentionally keeps imports lightweight at package import time.
Heavy dependencies (e.g. MNE connectivity backends) are imported lazily when
their specific submodules are used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # TFR
    "tfr_grid": ("src.lfp.tfr.grid", "grid"),
    "decompose": ("src.lfp.tfr.decompose", "decompose"),
    # Connectivity
    "conn_grid": ("src.lfp.connectivity.grid", "grid"),
    "swap_reciprocal_pairs_on_channel_axis": (
        "src.lfp.connectivity.utils",
        "swap_reciprocal_pairs_on_channel_axis",
    ),
    # PSI
    "psi_grid": ("src.lfp.psi.grid", "grid"),
    # Burst
    "burst_grid": ("src.lfp.burst.grid", "grid"),
}


def __getattr__(name: str) -> Any:
    """Lazily import selected convenience symbols.

    This keeps ``import src.lfp`` cheap and avoids importing heavy optional
    dependencies unless the corresponding functionality is actually used.
    """

    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        mod = import_module(module_name)
        value = getattr(mod, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'src.lfp' has no attribute {name!r}")


__all__ = sorted(_LAZY_EXPORTS.keys())
