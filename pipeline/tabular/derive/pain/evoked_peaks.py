"""
Pain-evoked peak property detection (paradigm-specific downstream analysis).

This script is an out-of-band, pain-paradigm utility that extracts evoked-peak features
from precomputed pain-locked band-power traces stored as summary tables.

Workflow overview
-----------------
1) Load paradigm-specific raw/derived summary DataFrames (Pickle).
2) Row-filter each DataFrame according to `cfg.slice_cfg` (e.g., select subjects/contacts/bands).
3) Run evoked-peak detection on the per-row time series stored in the "Value" column.
4) Apply QC filtering to remove unreliable detections (see `cfg.filter_df`).
5) Export selected feature groups (defined in `cfg.out_params`) into long-format tables
   via `cfg.melt_df`, then save as Pickle for downstream statistics and visualization.

Notes
-----
- This module is intentionally paradigm-specific and is not part of the core LFP pipeline.
  It should live under a derive/pain namespace (e.g., `src/derive/pain/...`) rather than
  in general-purpose signal-processing modules.
- Peak detection behavior is controlled entirely by `cfg.detect_peaks_cfg`.
- QC criteria are defined in `cfg.filter_df` and should be kept conservative to avoid
  polluting downstream statistics with spurious events.
"""


#%%
import os
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import src.utils.paths as paths
import src.utils.fileops as fileops
from src.tabular.selection import slice_df
from src.derive.pain.evoked_peaks import apply_detect_evoked_peaks_to_df as detect_peaks
import plugin.default.derive.pain.evoked_peaks_default as cfg
from src.io.pkl_io import load_pkl, save_pkl

import importlib
importlib.reload(cfg)

#%% 1) Project file/folder configuration

project = Path(os.environ["MYLFP_PROJECT_ROOT"]) # Folder path of project constructed in BIDS
summary_dir = project / 'summary'
table_dir = summary_dir / 'table'
raw_dir = table_dir / 'raw'
derive_dir = table_dir / 'derive'
pain_dir = derive_dir / 'pain'
pain_evoked_dir = derive_dir / 'pain_evoked'

#%% 2) Process each dataframe

for p in cfg.evoked_params_paths:
    p = Path(p)
    df_path = pain_dir / p
    print(f"Processing data: {df_path.name}")
    
    df = load_pkl(df_path)
    df = slice_df(df, cfg.slice_cfg[df_path.stem])
    
    # filp the value sign for trough detection
    if df_path.stem in cfg.peak_or_trough:
        for band, is_peak in cfg.peak_or_trough[df_path.stem].items():
            if not is_peak:
                print(f"Flipping sign for trough detection: {band}")
                mask = df['Band'] == band
                df.loc[mask, 'Value'] = -df.loc[mask, 'Value']
    
    # Detect peaks
    df_peaks = detect_peaks(
        df,
        **cfg.detect_peaks_cfg       
    )
    
    for param in cfg.out_params:
        df0 = pd.DataFrame()
        for phase in cfg.phase:
            # Filter peaks based on QC
            df_peaks_filtered = cfg.filter_df(df_peaks, phase)
            
            cols_basic = df.columns.tolist()
            cols_basic.remove("Value")
            col = f"{phase}-{param}"
            cols = cols_basic + [col]
            df_extracted = df_peaks_filtered[cols].copy()
            df_extracted = cfg.melt_df(df_extracted, [col])
            df_extracted = df_extracted.dropna()
            df0 = pd.concat([df0, df_extracted.copy()], 
                            axis=0, ignore_index=True)
        save_name = df_path.stem.rsplit('-', 1)[0] + f'-{param}.pkl'
        save_path = pain_evoked_dir / p.parent / save_name
        save_pkl(df0, save_path)
        
    #%%