"""
Interactive step-by-step pipeline for LFP time-frequency features on a stitched MNE Raw.

This script is intentionally written as a sequence of IPython/VSCode "#%%" cells to
run incrementally. Workflow:

1) Load stitched Raw (.fif)
2) Define shared tensor parameters (TFR/Connectivity/PSI/Burst)
3) Compute unwarped tensors once on the full concatenated Raw and save to stitch/tensor/
4) Build paradigm-specific masks (pain, gait) and save to stitch/masked/
"""

#%% 0) Imports & interactive setup (paths, configs, matplotlib backend)
from __future__ import annotations
import os
import time
import sys
from pathlib import Path

SCRIPT_START_TIME = time.perf_counter()

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
    
import src.utils.paths as paths
import src.utils.fileops as fileops
import plugin.default.tensors_sit_default as cfg
from src.lfp.common.timefreq import make_frequency_grid
from src.lfp.common.timefreq import morlet_n_cycles_from_time_fwhm_vector as get_morlet_n_cycles
from src.lfp.common.timefreq import morlet_mask_radius_time_s_from_freqs_n_cycles as get_morlet_mask_radius
from src.lfp.interp.freq import interpolate_tensor_with_metadata_transformed as interp_tensor
from src.lfp.smooth.smooth import smooth_axis
from src.utils.freqs import cut_notched_freqs, split_bands_by_notches
from src.lfp.burst.grid import grid as burst_grid
from src.lfp.connectivity.grid import grid as conn_grid
from src.lfp.connectivity.utils import swap_reciprocal_pairs_on_channel_axis as swap_channels
from src.lfp.psi.grid import grid as psi_grid
from src.lfp.tfr.decompose import decompose, make_gof_rsquared_masker
from src.lfp.tfr.grid import grid as tfr_grid
from src.io.pkl_io import load_pkl, save_pkl
from src.lfp.connectivity.pairs import build_within_group_pairs as get_pairs
from src.lfp.pipelines.masking import mask_tensors_dynamic
from src.qc.tensor_qc import export_tensor_qc
from src.utils.transforms import apply_transform_array

# %matplotlib qt

import importlib
importlib.reload(cfg)

#%% 1) Define project paths and dataset identifiers (subject/paradigm/sync settings)
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = Path(r"derivatives/mne")  # Lead-DBS derivatives folder (BIDS-style)
deri = project / deri

subject = os.environ["MYLFP_SUBJECT"]  # must start with "sub-"
wd = deri / subject

stitch_dir = wd / "sit"
preproc_dir = stitch_dir / "preproc"

fifpath = preproc_dir / "raw.fif"
print("Reading:", fifpath)

run_methods_in_parallel = True

#%% 2.1) Load stitched Raw
raw = mne.io.read_raw_fif(fifpath, preload=True)
print(raw)
print("sfreq (Hz):", raw.info["sfreq"])
print("n_times:", raw.n_times)
print("Duration (s):", raw.times[-1] - raw.times[0])
print("n_annotations:", len(raw.annotations))

#%% 2.2) Pair settings
channels = raw.ch_names
nchs = len(channels)

ch_dict = {
    'L': channels[:nchs//2],
    'R': channels[nchs//2:],
}

pairs_is_undirected = get_pairs(
    ch_dict, 
    ch_names=sum(ch_dict.values(), []), 
    ordered=False)[2]

pairs_is_directed = get_pairs(
    ch_dict, 
    ch_names=sum(ch_dict.values(), []), 
    ordered=True)[2]

# #%% 2.3) Optional: Plot Raw overview
# raw.plot()

#%% 3.1) Create output folders
tensor_dir = fileops.create_subfolder(stitch_dir, "tensor")
tensor_pkl = tensor_dir / "tensors_raw.pkl"
qc_dir = fileops.create_subfolder(tensor_dir, "qc")
print("tensor_pkl:", tensor_pkl)

#%% 4) Compute tensors ONCE on the full stitched Raw
if tensor_pkl.exists():
    tensors = load_pkl(tensor_pkl)
    print("Loaded cached tensors:", tensor_pkl)
else:
    # 4.1) Build frequency grids (full grid + notch-cut compute grid)
    kind = "log" if bool(cfg.tfr_grid_cfg.get("log_grid", False)) else "linear"
    freqs_full = make_frequency_grid(
        float(cfg.tfr_grid_cfg["fmin"]),
        float(cfg.tfr_grid_cfg["fmax"]),
        int(cfg.tfr_grid_cfg["n_freqs"]),
        kind=kind,
    )
    
    freqs_final = make_frequency_grid(**cfg.freqs_grid, kind=kind)
    idx_full_to_final = [i for i, f in enumerate(freqs_full) if f in freqs_final]

    notch_cut = cut_notched_freqs(freqs_full, cfg.notches_base_hz, bandwidth_hz=float(cfg.notch_bandwidth_hz))
    freqs_compute = notch_cut.freqs_kept    
    
    notch_cut_final = cut_notched_freqs(freqs_final, cfg.notches_base_hz, bandwidth_hz=float(cfg.notch_bandwidth_hz))
    freqs_compute_final = notch_cut_final.freqs_kept
    
    # 4.2) TFR (raw) computed on the notch-cut grid
    print("Computing TFR (notch-cut freqs)...")
    tfr_raw_4d, tfr_meta = tfr_grid(
        raw,
        n_jobs=-1,
        freqs=freqs_compute,
        **cfg.tfr_grid_cfg,
    )

    # Raw -> single "epoch": (1, ch, freq, time)
    tfr_raw_compute = np.asarray(tfr_raw_4d[0], dtype=np.float64)  # (ch, freq, time)
    freqs_compute = np.asarray(tfr_meta["axes"]["freq"], dtype=float)
    times = np.asarray(tfr_meta["axes"]["time"], dtype=float)
    hop_s = float(tfr_meta["params"]["hop_s"])
    decim_eff = int(tfr_meta["params"]["decim_eff"])
    target_n_times = int(tfr_raw_compute.shape[-1])
    
    qc_tfr_compute_dir = fileops.create_subfolder(qc_dir, "tfr_compute")
    export_tensor_qc(
        apply_transform_array(
            tfr_raw_compute[None, ...], 
            mode='dB'), 
        tfr_meta, 
        out_dir=qc_tfr_compute_dir
        )

    # 4.3) Interpolate notch-removed frequency bins (restores full grid)
    print(f"Interpolating notch bins (method={cfg.freq_interp_cfg.get('method')})...")
    tfr_raw, tfr_meta = interp_tensor(
        tfr_raw_compute,
        tfr_meta,
        freqs_out=freqs_full,
        transform_mode='dB',
        **cfg.freq_interp_cfg,
    )
        
    tfr_raw = np.clip(tfr_raw, a_min=np.min(tfr_raw_4d[tfr_raw_4d>0]), a_max=None)  # avoid zeros for log computations later
    freqs = np.asarray(tfr_meta["axes"]["freq"], dtype=float)
    channels = tfr_meta["axes"]["channel"]
    print("TFR:", tfr_raw.shape, "decim_eff:", decim_eff, "target_n_times:", target_n_times)

    qc_tfr_full_dir = fileops.create_subfolder(qc_dir, "tfr_full")
    export_tensor_qc(
        apply_transform_array(
            tfr_raw[None, ...], 
            mode='dB'), 
        tfr_meta, 
        out_dir=qc_tfr_full_dir
        )
    
    # 4.4.a) Smooth the SpecParam input. Frequency smoothing is applied first,
    # followed by the existing time-axis smoothing.
    
    tfr_smoothed = smooth_axis(tfr_raw, **cfg.freq_smooth_cfg)
    tfr_smoothed = smooth_axis(tfr_smoothed, **cfg.time_smooth_cfg)
    
    # 4.4.b) SpecParam decomposition (on interpolated full-grid TFR)
    # SpecParam expects 4D: (epoch, ch, freq, time)
    qc_specparam_dir = fileops.create_subfolder(qc_dir, "specparam")
    print("Computing SpecParam decomposition...")
    (
        tfr_aperiod_4d,
        tfr_period_4d,
        tfr_full_4d,
        params_tensor_4d,
        params_meta,
    ) = decompose(
        tfr_smoothed[None, ...],
        freqs=freqs,
        times=times,
        ch_names=channels,
        report_dir=qc_specparam_dir / "specparam_report",
        n_jobs=-1,
        **cfg.decompose_cfg,
    )

    masker = make_gof_rsquared_masker(params_tensor_4d, params_meta, threshold=cfg.specparam_qc_thr)
    tfr_aperiod_4d = masker(tfr_aperiod_4d)
    tfr_period_4d = masker(tfr_period_4d)
    tfr_full_4d = masker(tfr_full_4d)
    params_tensor_4d = masker(params_tensor_4d)
        
    tfr_aperiod = np.asarray(tfr_aperiod_4d[0], dtype=np.float64)
    tfr_period = np.asarray(tfr_period_4d[0], dtype=np.float64)
    tfr_full = np.asarray(tfr_full_4d[0], dtype=np.float64)
    params_tensor = np.asarray(params_tensor_4d[0], dtype=np.float64)

    # tuncate frequency params to match final freq grid; reduce edge artifacts   

    tfr_raw = tfr_raw[:, idx_full_to_final, :]
    tfr_aperiod = tfr_aperiod[:, idx_full_to_final, :]
    tfr_period = tfr_period[:, idx_full_to_final, :]
    tfr_full = tfr_full[:, idx_full_to_final, :]
        
    tfr_meta['axes']['freq'] = freqs_final
    tfr_meta['axes']['shape'] = tfr_raw.shape
    tfr_meta['params']['freqs_final'] = freqs_final
    
    export_tensor_qc(
        apply_transform_array(
            tfr_period[None, ...], 
            mode='dB'), 
        tfr_meta, 
        out_dir=qc_specparam_dir
        )    

    # 4.5) Connectivity tensors (computed on notch-cut freqs, then interpolated)

    print("Computing connectivity ...")
    conn_undirected: dict[str, np.ndarray] = {}
    conn_directed: dict[str, np.ndarray] = {}

    conn_meta = None
    gc_meta = None    
        
    if run_methods_in_parallel:
        method_specs = []
        for method in cfg.conn_methods:
            is_directed = method in ("gc", "gc_tr")
            ordered_pairs = True if is_directed else False
            pairs = pairs_is_directed if is_directed else pairs_is_undirected
            method_specs.append((method, is_directed, ordered_pairs, pairs))
            
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(conn_grid)(
                raw,
                freqs=freqs_compute_final,
                target_n_times=target_n_times,
                method=method,
                pairs=pairs,
                ordered_pairs=ordered_pairs,
                multivariate=is_directed,
                outer_n_jobs=1,          # already parallelized at higher level
                **cfg.conn_grid_cfg,
            )
            for method, is_directed, ordered_pairs, pairs in method_specs
        )

        for (method, is_directed, _, _), (con_tensor_4d, con_meta) in zip(method_specs, results):
            tensor3d = np.asarray(con_tensor_4d[0], dtype=np.float64)  # (channel, freq, time)
            if is_directed:
                conn_directed[method] = tensor3d
                gc_meta = con_meta if gc_meta is None else gc_meta
            else:
                conn_undirected[method] = tensor3d
                conn_meta = con_meta if conn_meta is None else conn_meta

    else:
        for method in cfg.conn_methods:
            print(f"  - {method}")
            is_directed = method in ("gc", "gc_tr")
            ordered_pairs = True if is_directed else False
            pairs = pairs_is_directed if is_directed else pairs_is_undirected

            con_tensor_4d, con_meta = conn_grid(
                raw,
                freqs=freqs_compute_final,
                target_n_times=target_n_times,
                method=method,
                pairs=pairs,
                ordered_pairs=ordered_pairs,
                multivariate=is_directed,
                outer_n_jobs=-1,
                **cfg.conn_grid_cfg,
            )
            tensor3d = np.asarray(con_tensor_4d[0], dtype=np.float64)  # (channel, freq, time)

            if is_directed:
                conn_directed[method] = tensor3d
                gc_meta = con_meta if gc_meta is None else gc_meta
            else:
                conn_undirected[method] = tensor3d
                conn_meta = con_meta if conn_meta is None else conn_meta

    # 4.6) Interpolate connectivity frequency axis to the full grid
    print("Interpolating connectivity tensors to full frequency grid...")

    conn_meta_full = None
    gc_meta_full = None

    if conn_meta is not None and len(conn_undirected) > 0:
        for k, v in list(conn_undirected.items()):
            v_full, conn_meta_full = interp_tensor(
                v,
                conn_meta,
                freqs_out=freqs_final,
                **cfg.freq_interp_cfg,
            )

            conn_undirected[k] = v_full

    if gc_meta is not None and len(conn_directed) > 0:
        for k, v in list(conn_directed.items()):
            v_full, gc_meta_full = interp_tensor(
                v,
                gc_meta,
                freqs_out=freqs_final,
                **cfg.freq_interp_cfg,
            )

            conn_directed[k] = v_full

    conn_meta = conn_meta_full if conn_meta_full is not None else conn_meta
    gc_meta = gc_meta_full if gc_meta_full is not None else gc_meta 
                   
    # 4.7) Derived GC tensors
    
    if "gc" in conn_directed and "gc_tr" in conn_directed:   
        print("Computing derived GC tensors...")     
        pair_axis = gc_meta_full["axes"]["channel"]
        pair_names = [tuple(p) for p in pair_axis]
        
        gc = conn_directed["gc"][None, ...]  # (1, channel, freq, time)
        gc_swap = swap_channels(gc, pair_names=pair_names)
        net_gc = (gc - gc_swap)[0]
        conn_directed["net_gc"] = np.asarray(net_gc, dtype=np.float64)
        
        gc_tr = conn_directed["gc_tr"][None, ...]
        gc_tr_swap = swap_channels(gc_tr, pair_names=pair_names)
        net_gc_tr = (gc_tr - gc_tr_swap)[0]
        conn_directed["net_gc_tr"] = np.asarray(net_gc_tr, dtype=np.float64)

        delta_gc = conn_directed["gc"] - conn_directed["gc_tr"]
        conn_directed["delta_gc"] = np.asarray(delta_gc, dtype=np.float64)   

        delta_net_gc = conn_directed["net_gc"] - conn_directed["net_gc_tr"]
        conn_directed["delta_net_gc"] = np.asarray(delta_net_gc, dtype=np.float64)         

    else:
        print("Skipping derived GC tensors (missing 'gc' or 'gc_tr').")

    # QC exports
    qc_conn_dir = fileops.create_subfolder(qc_dir, "connectivity")
    for k, v in conn_undirected.items():
        export_tensor_qc(
            apply_transform_array(
                v[None, ...], 
                mode='none'), 
            conn_meta, 
            out_dir=qc_conn_dir / k
            )
    
    qc_gc_dir = fileops.create_subfolder(qc_dir, "gc")    
    for k, v in conn_directed.items():
        export_tensor_qc(
            apply_transform_array(
                v[None, ...], 
                mode='none'), 
            gc_meta, 
            out_dir=qc_gc_dir / k
            )
    
    
    # 4.8) PSI
    print("Computing PSI...")
    bands_notch_safe = split_bands_by_notches(
        cfg.bands,
        cfg.notches_base_hz,
        bandwidth_hz=cfg.notch_bandwidth_hz,
        include_edges=True,
        drop_empty=True,
    )
    
    psi_cfg = cfg.psi_grid_cfg
    psi_cfg["bands"] = bands_notch_safe
    
    psi_tensor_4d, psi_meta = psi_grid(
        raw,
        pairs=pairs_is_undirected,
        target_n_times=target_n_times,
        n_jobs=-1,
        **psi_cfg,
    )
    
    time_resolution_s = float(psi_meta["params"]["time_resolution_s"])
    psi_tensor = np.asarray(psi_tensor_4d[0], dtype=np.float64)
    print("PSI:", psi_tensor.shape)
    
    qc_psi_dir = fileops.create_subfolder(qc_dir, "psi")
    export_tensor_qc(
        apply_transform_array(
            psi_tensor[None, ...], 
            mode='none'), 
        psi_meta, 
        y_scale_heatmap='linear',
        x_scale_spectrum='linear',        
        out_dir=qc_psi_dir / "psi"
        )

    # 4.9) Burst (baseline-driven threshold)
    print("Computing Burst...")
    burst_cfg = dict(cfg.burst_grid_cfg)
    burst_cfg["bands"] = bands_notch_safe
    burst_tensor_4d, burst_meta = burst_grid(
        raw,
        **burst_cfg,
    )
    burst_tensor = np.asarray(burst_tensor_4d[0], dtype=np.float64)
    print("Burst:", burst_tensor.shape)

    qc_burst_dir = fileops.create_subfolder(qc_dir, "burst")
    export_tensor_qc(
        apply_transform_array(
            burst_tensor[None, ...], 
            mode='none'), 
        burst_meta, 
        y_scale_heatmap='linear',
        x_scale_spectrum='linear',        
        out_dir=qc_burst_dir / "burst"
        )
    
    # 4.10)  #--- Build and dynamically edge-mask tensors before save ---
    tensors = dict(
        tfr=dict(
            raw=tfr_raw,
            aperiod=tfr_aperiod,
            period=tfr_period,
            full=tfr_full,
            meta=tfr_meta,
        ),
        params=dict(
            params=params_tensor,  # (channel, param, time)
            meta=params_meta,
        ),
        conn=dict(
            **conn_undirected,
            meta=conn_meta,
        ),
        gc=dict(
            **conn_directed,
            meta=gc_meta,
        ),
        psi=dict(
            psi=psi_tensor,
            meta=psi_meta,
        ),
        burst=dict(
            burst=burst_tensor,
            meta=burst_meta,
        ),
    )

    freqs = tensors["tfr"]["meta"]["axes"]["freq"]
    n_cycles = get_morlet_n_cycles(freqs, **cfg.morlet_cfg)

    bands_anno = tensors["psi"]["meta"]["axes"]["freq"]
    bands_freqs_min = np.array([cfg.bands[band][0] for band in cfg.bands], dtype=float)
    n_cycles_bands = get_morlet_n_cycles(bands_freqs_min, **cfg.morlet_cfg)

    params_anno = tensors["params"]["meta"]["axes"]["freq"]
    params_freqs = np.full(len(params_anno), np.min(freqs), dtype=float)
    n_cycles_params = get_morlet_n_cycles(params_freqs, **cfg.morlet_cfg)

    freqs_label_all = np.concatenate([freqs, bands_anno, params_anno])
    freqs_all = np.concatenate([freqs, bands_freqs_min, params_freqs])
    n_cycles_all = np.concatenate([n_cycles, n_cycles_bands, n_cycles_params])

    time_mask = get_morlet_mask_radius(
        freqs_all,
        n_cycles=n_cycles_all,
        k_sigma=cfg.k_sigma,
    )

    # Burst should remain unmasked; keep original burst group and restore it after dynamic masking.
    burst_unmasked = tensors["burst"]
    _, tensors = mask_tensors_dynamic(
        raw,
        tensors,
        freqs=freqs_label_all,
        time=time_mask,
        **cfg.dynamic_mask_cfg,
    )
    tensors["burst"] = burst_unmasked

    print("Saving tensors to:", tensor_pkl)
    save_pkl(tensors, tensor_pkl)
    print("Saved tensors:", tensor_pkl)

elapsed_seconds = time.perf_counter() - SCRIPT_START_TIME
elapsed_minutes, elapsed_seconds_remainder = divmod(elapsed_seconds, 60.0)
elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60.0)
print(
    "Total script runtime:"
    f" {int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{elapsed_seconds_remainder:05.2f}"
)

#%%
