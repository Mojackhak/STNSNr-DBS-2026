#%%
import os
import pandas as pd
import numpy as np
from pathlib import Path
import mne
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import src.utils.paths as paths
import src.utils.fileops as fileops
import src.viz.visualdf as vdf
import plugin.default.viz_defaults as cfg_viz
import plugin.default.tensors_default as cfg_tensors
from src.viz.multi_exposure import multi_exposure_motion
from src.preproc.channels import derive_channels_from_raw
from src.tabular.tensor_slices import split_tensor4d_to_nested_df as tensor2df
from src.stats.preproc.transform import transform_df
import plugin.default.derive_default as cfg_derive
from plugin.default.tabular_default import standardize_df
from src.lfp.burst.grid import grid as burst_grid
from src.stats.preproc.normalize import normalize_df, normalize_df_by_baseline
from src.lfp.common.timefreq import make_frequency_grid
from src.lfp.interp.freq import interpolate_tensor_with_metadata_transformed as interp_tensor
from src.lfp.smooth.smooth import smooth_tensor_time as smooth_time
from src.utils.freqs import cut_notched_freqs, split_bands_by_notches
from src.lfp.burst.grid import grid as burst_grid
from src.lfp.connectivity.grid import grid as conn_grid
from src.lfp.connectivity.utils import swap_reciprocal_pairs_on_channel_axis as swap_channels
from src.lfp.psi.grid import grid as psi_grid
from src.lfp.tfr.decompose import decompose, make_gof_rsquared_masker
from src.lfp.tfr.grid import grid as tfr_grid
from src.io.pkl_io import load_pkl, save_pkl
from src.lfp.connectivity.pairs import build_within_group_pairs as get_pairs
from src.qc.tensor_qc import export_tensor_qc
from src.utils.transforms import apply_transform_array


import importlib
importlib.reload(cfg_viz)

%matplotlib qt

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
summary_dir = project / r'summary'
stats_dir = summary_dir / r'stats'
section_dir = stats_dir / r"fog"
video_dir = section_dir / r'videos'
video_path = video_dir / r'fog_SNr008.mp4'
video_name = video_path.stem
max_frames = 10

#%%
img = multi_exposure_motion(
    video_path, 
    out_path=video_dir / f"{video_name}_out_darken.jpg", 
    fg_blend="darken", 
    alpha=0.5,
    bg_max_frames=200,
    blur_ksize=0,
    morph_ksize=5,
    thresh=16, 
    min_area=2000,
    dilate_iter=3,
    background_gray=True,
    background_dim=0.8,    
    max_frames=max_frames)

# %%
raw_file = os.environ["MYLFP_FOG_RAW"]
raw = mne.io.read_raw_fif(raw_file, preload=True)

# %%
#tmin=4*60+34+6/25
tmin=4*60+31+18/25
tmax=4*60+45+15/25
chs = raw.ch_names
type2dbs = {ch: 'dbs' for ch in chs}
type2eeg = {ch: 'eeg' for ch in chs}

# 1) Derive everything (mapping + bipolar pairs)
derived = derive_channels_from_raw(raw, drop_non_adjacent_pairs=True)

# 2) Use mapping
ch2eeg = derived.ch2eeg
eeg2ch = derived.eeg2ch

# 3) Bipolar pairs
anodes = derived.anodes
cathodes = derived.cathodes
new_names = derived.pair_names
dropped_idx = derived.dropped_pair_idx

raw_fog = raw.copy().crop(tmin=tmin, tmax=tmax)
annotations = mne.Annotations(
    (raw_fog.annotations.onset - raw_fog.first_samp / raw_fog.info['sfreq']),
    raw_fog.annotations.duration,
    raw_fog.annotations.description,
)
raw_fog_data = raw_fog.get_data()
# create mne raw
raw_fog = mne.io.RawArray(raw_fog_data, raw_fog.info)

raw_fog.set_annotations(annotations)
raw_fog = mne.set_bipolar_reference(
    raw_fog,
    anode=list(anodes),
    cathode=list(cathodes),
    ch_name=list(new_names),
    drop_refs=True)

# %%
raw_fog.plot()

# %%
kind = "log" if bool(cfg_tensors.tfr_grid_cfg.get("log_grid", False)) else "linear"
freqs_full = make_frequency_grid(
    float(cfg_tensors.tfr_grid_cfg["fmin"]),
    float(cfg_tensors.tfr_grid_cfg["fmax"]),
    int(cfg_tensors.tfr_grid_cfg["n_freqs"]),
    kind=kind,
)

freqs_final = make_frequency_grid(**cfg_tensors.freqs_grid, kind=kind)
idx_full_to_final = [i for i, f in enumerate(freqs_full) if f in freqs_final]

notch_cut = cut_notched_freqs(
    freqs_full, 
    cfg_tensors.notches_base_hz, 
    bandwidth_hz=float(cfg_tensors.notch_bandwidth_hz))
freqs_compute = notch_cut.freqs_kept    

notch_cut_final = cut_notched_freqs(
    freqs_final, 
    cfg_tensors.notches_base_hz, 
    bandwidth_hz=float(cfg_tensors.notch_bandwidth_hz))
freqs_compute_final = notch_cut_final.freqs_kept

# 4.2) TFR (raw) computed on the notch-cut grid
print("Computing TFR (notch-cut freqs)...")
tfr_raw_4d, tfr_meta = tfr_grid(
    raw_fog,
    n_jobs=-1,
    freqs=freqs_compute,
    **cfg_tensors.tfr_grid_cfg,
)

# Raw -> single "epoch": (1, ch, freq, time)
tfr_raw_compute = np.asarray(tfr_raw_4d[0], dtype=np.float64)  # (ch, freq, time)
freqs_compute = np.asarray(tfr_meta["axes"]["freq"], dtype=float)
times = np.asarray(tfr_meta["axes"]["time"], dtype=float)
hop_s = float(tfr_meta["params"]["hop_s"])
decim_eff = int(tfr_meta["params"]["decim_eff"])
target_n_times = int(tfr_raw_compute.shape[-1])

# 4.3) Interpolate notch-removed frequency bins (restores full grid)
print(f"Interpolating notch bins (method={cfg_tensors.freq_interp_cfg.get('method')})...")
tfr_raw, tfr_meta = interp_tensor(
    tfr_raw_compute,
    tfr_meta,
    freqs_out=freqs_full,
    transform_mode='dB',
    **cfg_tensors.freq_interp_cfg,
)
    
tfr_raw = np.clip(tfr_raw, a_min=np.min(tfr_raw_4d[tfr_raw_4d>0]), a_max=None)  # avoid zeros for log computations later
freqs = np.asarray(tfr_meta["axes"]["freq"], dtype=float)
channels = tfr_meta["axes"]["channel"]
print("TFR:", tfr_raw.shape, "decim_eff:", decim_eff, "target_n_times:", target_n_times)


# 4.4.a) Smooth to reduce noise. Important for stability of SpecParam decomposition.
tfr_smoothed = smooth_time(tfr_raw, **cfg_tensors.smooth_cfg)

# 4.4.b) SpecParam decomposition (on interpolated full-grid TFR)
# SpecParam expects 4D: (epoch, ch, freq, time)

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
    report_dir=video_dir / "specparam_report",
    n_jobs=-1,
    **cfg_tensors.decompose_cfg,
)

masker = make_gof_rsquared_masker(
    params_tensor_4d, 
    params_meta, 
    threshold=0,)
    # threshold=cfg_tensors.specparam_qc_thr)
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
    
# %%
df = tensor2df(
    tfr_period[None, ...],  # add back epoch dim: (1, ch, freq, time)
    tfr_meta['axes']['epoch'],
    tfr_meta['axes']['channel'],
    tfr_meta['axes']['freq'],
    tfr_meta['axes']['time'],
)
df.rename(columns={'channel': 'Contact'}, inplace=True)
df = standardize_df(df)
df = transform_df(df, mode='dB', **cfg_derive.transform_func_cfg)
baseline_normalize_cfg = {
    'mode_baseline': 'mean',
    'mode': 'mean',
    'eps': 1e-120,
    'slice_mode': 'percent'
}
df = normalize_df_by_baseline(
    df,
    baseline = [list(np.array([0.559, 4.039])/13.879 * 100)], 
    **baseline_normalize_cfg
)

# %%

config_updated = cfg_viz.update_raw_cfg(1, 'default')
# config_updated['cmap'] = 'viridis'
# config_updated['vmode'] = 'auto'
config_updated['colorbar_label'] = "Period power (dB)"
config_updated['x_label'] = 'Time (s)'
#config_updated['vertical_lines'] = [2.20, 9.48]
config_updated['vertical_lines'] = [4.72, 12.00]

for ch in df['Contact'].unique():
    df_ch = df[df['Contact'] == ch]
    fig = vdf.plot_single_effect_df(
        df=df_ch,
        **config_updated
    ) 
    save_path = video_dir / f'fog_SNr008_TFR_period_{ch}.pdf'
    fig.savefig(save_path, bbox_inches="tight", dpi=600)

chs = [['0_1', '8_9'], ['2_3', '10_11'], ['1_2', '2_3', '9_10', '10_11']]
for ch_group in chs:
    df_ch = df[df['Contact'].isin(ch_group)]
    fig = vdf.plot_single_effect_df(
        df=df_ch,
        **config_updated
    ) 
    ch_names = '_'.join(ch_group)
    save_path = video_dir / f'fog_SNr008_TFR_period_{ch_names}.pdf'
    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    
# %%
burst_cfg = dict(cfg_tensors.burst_grid_cfg)
burst_cfg['baseline_keep'] = ['walk',]
bands_notch_safe = split_bands_by_notches(
    cfg_tensors.bands,
    cfg_tensors.notches_base_hz,
    bandwidth_hz=cfg_tensors.notch_bandwidth_hz,
    include_edges=True,
    drop_empty=True,
)
burst_cfg["bands"] = bands_notch_safe
burst_tensor_4d, burst_meta = burst_grid(
    raw_fog,
    **burst_cfg,
)

df_burst = tensor2df(
    burst_tensor_4d,
    burst_meta['axes']['epoch'],
    burst_meta['axes']['channel'],
    np.arange(len(burst_meta['axes']['freq'])),
    burst_meta['axes']['time'],
)
df_burst.rename(columns={'channel': 'Contact'}, inplace=True)
df_burst = standardize_df(df_burst)
df_burst = transform_df(df_burst, mode='log', **cfg_derive.transform_func_cfg)

# %%
config_updated = cfg_viz.update_raw_cfg(1, 'default')
config_updated['cmap'] = 'viridis'
config_updated['vmode'] = 'auto'
config_updated['colorbar_label'] = r"$\operatorname{ln}$ burst amplitude (V)"
#config_updated['vertical_lines'] = [2.20, 9.48]
config_updated['vertical_lines'] = [4.72, 12.00]
config_updated['y_log'] = False
config_updated['x_label'] = 'Time (s)'
config_updated['y_limits'] = [-0.5, 5.5]
for ch in df_burst['Contact'].unique():
    df_ch = df_burst[df_burst['Contact'] == ch]
    fig = vdf.plot_single_effect_df(
        df=df_ch,
        **config_updated
    ) 
    save_path = video_dir / f'fog_SNr008__burst_{ch}.pdf'
    fig.savefig(save_path, bbox_inches="tight", dpi=600)

chs = [['0_1', '8_9'], ['2_3', '10_11'], ['1_2', '2_3', '9_10', '10_11']]
for ch_group in chs:
    df_ch = df_burst[df_burst['Contact'].isin(ch_group)]
    fig = vdf.plot_single_effect_df(
        df=df_ch,
        **config_updated
    ) 
    ch_names = '_'.join(ch_group)
    save_path = video_dir / f'fog_SNr008__burst_{ch_names}.pdf'
    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    
# %%
