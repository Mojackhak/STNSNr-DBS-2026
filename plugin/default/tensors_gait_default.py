"""
# Necessary parameters for lfp/tensors pipeline
1. bands
2. notches_base_hz
3. freq_interp_cfg
4. tfr_grid_cfg
5. decompose_cfg
6. conn_grid_cfg
7. psi_grid_cfg
8. burst_grid_cfg
9. morlet_cfg
10. k_sigma
11. dynamic_mask_cfg

"""

import os
import numpy as np
from src.lfp.connectivity.pairs import build_within_group_pairs as get_pairs
from plugin.default.preproc_default import notches_base_hz


bands = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 35.0),
    "gamma": (35.0, 75.0),
}

freqs_grid = {
    'fmin': 1.0,
    'fmax': 75.0,
    'n_freqs': 149,
}

notch_bandwidth_hz = 2.0  # radius (± Hz) around each notch center

freq_interp_cfg = {
    "axis": -2,
    "method": "linear",
    # "savgol_window": 13,
    # "savgol_polyorder": 2,
}

time_smooth_cfg = {
    'kernel_size': 13,
    'method': 'median',
    'axis': -1,
    'transform_mode': 'dB',
}

specparam_qc_thr = 0.6

grid_base_cfg = {
    "time_resolution_s": 0.3,
    "hop_s": 0.025,
    "decim": None,  
    "min_cycles": 3.0,
    "max_cycles": 13.0,
}

morlet_cfg = {
    "fwhm_time_s": grid_base_cfg["time_resolution_s"],
    "min_cycles": grid_base_cfg["min_cycles"],
    "max_cycles": grid_base_cfg["max_cycles"],
}

k_sigma = 5.0  # wavelets of mne extend to ±5 standard deviations

dynamic_mask_cfg = {
    "drop": ["bad", "edge"],
    "mode": "substring",
    "clip_to_raw": True,
}

# TFR grid parameters
tfr_grid_cfg = {
    "method": "morlet",
    "fmin": 1,
    "fmax": 80.0, # larger than max band freq (75 Hz)
    "n_freqs": 159,  # ~0.5 Hz resolution if linear
    "log_grid": False,
    "average": False,
    **grid_base_cfg,
}

decompose_cfg = {
    "freq_range": (1.0, 80.0),
    "aperiodic_mode": "fixed",
    "peak_width_limits": (2.0, 12.0),
    "max_n_peaks": 8,
    "min_peak_height": 0.15,
    "peak_threshold": 2,
}
    
conn_grid_cfg = {
    "gc_n_lags": 20,
    "gc_pad_scale": 10,
    **grid_base_cfg,
}   

psi_grid_cfg = {
    "bands": bands,
    "ordered_pairs": True,
    **grid_base_cfg,
}
    
burst_grid_cfg = {
    "bands": bands,
    "percentile": 75.0,
    "min_cycles": 2,
    "hop_s": None,
    "decim": 1,
    "baseline_keep": ["stand"],
    "baseline_match": "exact",
    "baseline_fallback": "raise",
}

conn_methods = ["coh", "plv", "ciplv", "pli", "wpli", "gc", "gc_tr"]
