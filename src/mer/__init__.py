"""mer

Main purpose
------------
A small, modular toolkit for reading and analyzing intraoperative MER data,
with an emphasis on NeuroOmega exports and reproducible workflows.

Submodules
----------
* ``nex_convert``: MAP/MPX -> NEX conversion (requires NeuroExplorer `nex` API)
* ``neuroomega_mat``: Read NeuroOmega MER `.mat` exports (CLFP/CSPK)
* ``metrics``: Feature extraction and spike-train metrics
* ``stn_boundaries``: STN entry/exit estimation helpers
* ``plotting``: Convenience visualization
* ``spikeinterface_adapter``: Optional SpikeInterface integration
"""

from __future__ import annotations

from .nex_convert import convert_map_mpx_to_nex, convert_one_map_mpx_to_nex
from .neuroomega_mat import MerMatReadConfig, MerReadResult, MerSignal, read_neuroomega_mer_mat_folder, read_oa_mer
from .metrics import compute_burst_index, compute_lvr, compute_lvr_from_analyzer, normalize_rms
from .stn_boundaries import get_STN_entry, get_STN_exit, get_stn_entry_depth, get_stn_exit_depth

__all__ = [
    "MerMatReadConfig",
    "MerReadResult",
    "MerSignal",
    "read_neuroomega_mer_mat_folder",
    "read_oa_mer",
    "convert_one_map_mpx_to_nex",
    "convert_map_mpx_to_nex",
    "normalize_rms",
    "compute_lvr",
    "compute_lvr_from_analyzer",
    "compute_burst_index",
    "get_stn_entry_depth",
    "get_stn_exit_depth",
    "get_STN_entry",
    "get_STN_exit",
]
