"""
Derive analysis-ready summary tables from raw nested-value DataFrames.

This pipeline:
- discovers raw summary DataFrames stored as .pkl files (typically one per
  paradigm/param/subparam folder),
- optionally applies value transforms (e.g., log/dB/Fisher-z) driven by config,
- normalizes table columns via a project-wide standardizer,
- derives secondary feature tables by collapsing nested values:
    * spectral: PSD-like traces over time (collapse along time axis)
    * trace: band-limited time traces over frequency (collapse along freq axis)
    * scalar: band × time grid reduced to scalars
- writes derived DataFrames to disk and records per-file run logs.

Outputs are saved under:
    <project>/summary/table/derive/<paradigm>/<param>/<subparam>/

Notes
-----
- Raw tables are expected to contain a nested "value" column (scalar / pd.Series /
  pd.DataFrame) compatible with src.tabular.grid helpers.
- Derivation behavior (axes, reducers, enabled outputs) is controlled by config modules
  in config/.
"""


#%% 0) Imports & interactive setup (paths, configs, matplotlib backend)
import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.utils.paths as paths
import src.utils.fileops as fileops
from src.stats.preproc.transform import transform_df
import src.tabular.grid as grid
from src.utils.parallel import parallel_process_files_tqdm as parallel_tqdm
# import plugin.default.derive_default as cfg
import plugin.default.derive_posetrack_default as cfg
# import plugin.default.derive_turn_default as cfg
from plugin.default.tabular_default import standardize_df
from src.io.pkl_io import load_pkl, save_pkl
from src.utils.runlog import with_run_log, save_logs
from src.utils.config import get_param
from src.tabular.missing import drop_missing_value_rows
from datetime import datetime

import importlib
importlib.reload(cfg)

#%% 1) Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
deri = project / 'derivatives' / 'mne'

summary_dir = project / 'summary'
table_dir = summary_dir / 'table'
raw_dir = table_dir / 'raw'
derive_dir = fileops.create_subfolder(table_dir, 'derive')

overwrite = True  # whether to overwrite existing derived dataframes

#%% 2) get path of all the raw dataframes
df_paths = paths.get_files_list(raw_dir, filetypes='.pkl', recursive=True, contain=cfg.contain)
print(f"Found {len(df_paths)} raw dataframes in {raw_dir}")

#%% 3) Define processing function
# for df_path in df_paths:
@with_run_log()
def process_derive_df(df_path: Path):
    print("Processing:", df_path)
    subparam_path = df_path.parent
    param_path = subparam_path.parent
    paradigm_path = param_path.parent
    subparam = subparam_path.name
    param = param_path.name
    paradigm = paradigm_path.name
    print(f"  paradigm: {paradigm}, param: {param}, subparam: {subparam}")
    df = load_pkl(df_path)
    df = drop_missing_value_rows(df, value_col='Value', mode='all')
    save_dir = derive_dir / paradigm / param / subparam
    
    # params for derivation
    mode = get_param(cfg.pre_transform_mode_cfg, param, subparam)
    freqs_axis = cfg.freqs_cfg.get(param, cfg.freqs_cfg['default'])
    time_axis = cfg.time_cfg.get(paradigm, cfg.time_cfg['default'])
    reducer_list = get_param(cfg.reducer_cfg, param, subparam)
    derive_paradigm = cfg.derive_paradigm_cfg.get(
        paradigm, cfg.derive_paradigm_cfg['default'])

    print(f"  pre-transform mode: {mode}")
    print(f"  freq axis: {freqs_axis}")
    print(f"  time axis: {time_axis}")
    print(f"  reducer: {reducer_list}")
    print(f"  derives: {derive_paradigm}")

    # Apply the transform first to improve robustness to noise if needed
    df = transform_df(df, mode=mode, **cfg.transform_func_cfg)
    df = standardize_df(df)

    if cfg.is_derivable(paradigm, param, 'raw'):
        print("  deriving transformed raw features...")
        save_path = save_dir / f"{subparam}-transformed-raw.pkl"
        if save_path.exists() and not overwrite:
            print(f"  transformed raw dataframe already exists. Skipping...")
        else:
            mode_post = get_param(cfg.post_transform_mode_cfg, subparam, 'default')
            df_transformed = transform_df(df, mode=mode_post, **cfg.transform_func_cfg)
            df_transformed = standardize_df(df_transformed)
            save_pkl(df_transformed, save_path)
            print(f"  applying post-transform mode: {mode_post}")
            save_pkl(df_transformed, save_path)

    # derive spectral features
    if cfg.is_derivable(paradigm, param, 'spectral'):
        print("  deriving spectral features...")
        for reducer in reducer_list:
            save_path = save_dir / f"{subparam}-{reducer}-spectral.pkl"
            if save_path.exists() and not overwrite:
                print(f"    spectral dataframe with reducer {reducer} already exists. Skipping...")
                continue
            print(f"    applying reducer: {reducer}")
            df_spec = grid.split_nested_values(
                df,
                bands = None,
                times = time_axis,
                axis = "time",
                reducer = reducer,
                **cfg.collapse_base_cfg,
                )
            mode_post = get_param(cfg.post_transform_mode_cfg, subparam, reducer)
            print(f"    applying post-transform mode: {mode_post}")
            df_spec = transform_df(df_spec, mode=mode_post, **cfg.transform_func_cfg)
            df_spec = standardize_df(df_spec)
            save_pkl(df_spec, save_path)
    
    # derive trace features
    if cfg.is_derivable(paradigm, param, 'trace'):
        print("  deriving trace features...")
        for reducer in reducer_list:
            save_path = save_dir / f"{subparam}-{reducer}-trace.pkl"
            if save_path.exists() and not overwrite:
                print(f"    trace dataframe with reducer {reducer} already exists. Skipping...")
                continue
            print(f"    applying reducer: {reducer}")
            df_trace = grid.split_nested_values(
                df,
                bands = freqs_axis,
                times = None,
                axis = "freq",
                reducer = reducer,
                **cfg.collapse_base_cfg,
                )
            mode_post = get_param(cfg.post_transform_mode_cfg, subparam, reducer)
            print(f"    applying post-transform mode: {mode_post}")
            df_trace = transform_df(df_trace, mode=mode_post, **cfg.transform_func_cfg)
            df_trace = standardize_df(df_trace)
            save_pkl(df_trace, save_path)

    # derive scalar features
    if cfg.is_derivable(paradigm, param, 'scalar'):
        print("  deriving scalar features...")
        for reducer in reducer_list:
            save_path = save_dir / f"{subparam}-{reducer}-scalar.pkl"
            if save_path.exists() and not overwrite:
                print(f"    scalar dataframe with reducer {reducer} already exists. Skipping...")
                continue
            print(f"    applying reducer: {reducer}")
            df_scalar = grid.grid_nested_values(
                df,
                bands = freqs_axis,
                times = time_axis,
                reducer = reducer,
                **cfg.collapse_base_cfg,
                )
            mode_post = get_param(cfg.post_transform_mode_cfg, subparam, reducer)
            print(f"    applying post-transform mode: {mode_post}")
            df_scalar = transform_df(df_scalar, mode=mode_post, **cfg.transform_func_cfg)
            df_scalar = standardize_df(df_scalar)
            save_pkl(df_scalar, save_path)

#%% 8) parallel processing
logs = parallel_tqdm(
    df_paths,
    func = process_derive_df,
    n_jobs=-1,
    verbose=True,
)

time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = save_logs(logs, derive_dir / f"run_logs_{time_now}.jsonl")
print("Saved:", log_path)    

# %%
