

#%% 0) Imports & interactive setup (paths, configs, matplotlib backend)
import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import importlib

import src.utils.paths as paths
import src.utils.fileops as fileops
from src.stats.preproc.outliers import filter_df
from src.stats.preproc.aggregate import summarize_df
from src.tabular.missing import drop_missing_value_rows
from src.tabular.selection import slice_df
from src.tabular.reshape import split_rows_by_columns
from src.tabular.df_schema import normalize_df_categories
from src.utils.parallel import parallel_process_files_tqdm as parallel_tqdm
# import plugin.default.stats_default as cfg
# import plugin.default.stats_posetrack_default as cfg
# import plugin.default.stats_posetrack_percent_default as cfg
# import plugin.default.stats_turn_default as cfg
_cfg_module = os.environ.get("MYLFP_STATS_CFG", "plugin.default.stats_turn_percent_default")
cfg = importlib.import_module(_cfg_module)
from src.io.pkl_io import load_pkl, save_pkl
from src.utils.runlog import with_run_log, save_logs
from pyreadr import write_rds
from datetime import datetime
from typing import Sequence

importlib.reload(cfg)


def _env_list(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}

#%% 1) Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
deri = project / 'derivatives' / 'mne'

summary_dir = project / 'summary'
table_dir = summary_dir / 'table'
derive_dir = table_dir / 'derive'
stats_dir = fileops.create_subfolder(summary_dir, 'stats')
overwrite = True  # whether to overwrite existing derived dataframes

#%% 2) get path of all the raw dataframes
df_paths = paths.get_files_list(derive_dir, filetypes='.pkl', recursive=True, contain=cfg.contain)
target_stems = _env_list("MYLFP_STATS_STEMS")
if target_stems:
    df_paths = [df_path for df_path in df_paths if df_path.stem in target_stems]
print(f"Found {len(df_paths)} raw dataframes in {derive_dir}")

#%% 3) Define processing function
                
@with_run_log()
def process_stats_df(df_path: Path):
    print("Processing:", df_path)
    subparam_path = df_path.parent
    param_path = subparam_path.parent
    paradigm_path = param_path.parent
    subparam = subparam_path.name
    param = param_path.name
    paradigm = paradigm_path.name
    feature = df_path.stem.split('-', 2)[-1]
    print(f"  paradigm: {paradigm}")
    print(f"  param: {param}")
    print(f"  subparam: {subparam}")
    print(f"  feature: {feature}")
    # get sections to process
    sections = cfg.sections.get(paradigm, [])
    target_sections = _env_list("MYLFP_STATS_SECTIONS")
    if target_sections:
        sections = [section for section in sections if section in target_sections]

    def save_df_with_check(df: pd.DataFrame, save_path: Path):
        #standardize the table
        df = normalize_df_categories(df, **cfg.normalize_df_category_cfg)
        df = cfg.standardize_df_stats(df)
        
        if save_path.exists() and not overwrite:
            print(f"  {save_path.name} already exists. Skipping...")
        else:
            save_pkl(df, save_path)
            print(f"  saved dataframe to: {save_path}")
            if cfg.to_extra_format_cfg.get(feature, cfg.to_extra_format_cfg['default']):
                rds_path = save_path.with_suffix('.rds')
                write_rds(rds_path, df)
                print(f"  also saved RDS to: {rds_path}")
                csv_path = save_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                print(f"  also saved CSV to: {csv_path}")
                
    for section in sections:
        print(f" Processing section: {section}")
        df = load_pkl(df_path) 
        save_dir = stats_dir / section / param / subparam / df_path.stem
        df = drop_missing_value_rows(df, value_col='Value', mode='all')
        subset = cfg.get_param(cfg.slices_cfg, paradigm, section)
        if subset is not None:
            df = slice_df(df, subset)
        df = df.reset_index(drop=True)
        cols_basic = [col for col in df.columns if col in cfg.cols_basic]
        cols_with_coords = cols_basic + [col for col in df.columns if col in cfg.cols_coords]
        
        cols_filter_drop = cfg.get_param(cfg.cols_filter_cfg, paradigm, section)
        
        cols_filter =  [col for col in cols_basic if col not in cols_filter_drop]

        # filter outliers
        df_filtered, report = filter_df(
            df, group_cols=cols_filter, 
            **cfg.filter_outliers_cfg)

        print("Dropped %d outliers (%.2f%%) from %s" % 
            (report['overall']['n_removed'], 
             report['overall']['pct_removed']*100, 
             df_path.stem))
        
        df_filtered = df_filtered.reset_index(drop=True)
        
        # normalize values
        df_normalized = cfg.normalize_df_func(
            df_filtered, paradigm, section, feature)

        def aggregate_df_and_save(df: pd.DataFrame, df_norm: pd.DataFrame, 
                                  cols_cfg: dict, cols: Sequence, level: str):
                cols_agg_drop = cfg.get_param(cols_cfg, paradigm, section)
                cols_agg = [col for col in cols if col not in cols_agg_drop]
                print(f"  aggregating to {level} level by {cols_agg} ...")
                
                df_agg = summarize_df(df, group_cols=cols_agg, **cfg.summarize_df_cfg)
                df_agg_normalized = summarize_df(df_norm, group_cols=cols_agg, **cfg.summarize_df_cfg)
                
                save_path = save_dir / f"{df_path.stem}_{level}_raw.pkl"
                save_df_with_check(df_agg, save_path)
                
                save_path = save_dir / f"{df_path.stem}_{level}_normalized.pkl"
                save_df_with_check(df_agg_normalized, save_path)
                
                return df_agg, df_agg_normalized
        
        # aggregate to contact level (mean over epochs)
        df_contact, df_contact_normalized = aggregate_df_and_save(
            df_filtered, df_normalized, 
            cfg.cols_contact_agg_cfg, cols_with_coords, 'contact')

        # aggregate to region level (mean over contacts)
        df_region, df_region_normalized = aggregate_df_and_save(
            df_contact, df_contact_normalized, 
            cfg.cols_region_agg_cfg, cols_basic, 'region')
        
        # aggregate to subject level (mean over regions)
        df_subject, df_subject_normalized = aggregate_df_and_save(
            df_region, df_region_normalized, 
            cfg.cols_subject_agg_cfg, cols_basic, 'subject')
        
        # split connectivity features
        is_split = param in cfg.split_features_cfg.get(paradigm, cfg.split_features_cfg['default'])
        if is_split:
            col_old = list(cfg.split_col_bak.keys())[0]
            col_bak = cfg.split_col_bak[col_old]
            df_conn_split = df_contact.copy()
            df_conn_split[col_bak] = df_conn_split[col_old]
            df_conn_split = split_rows_by_columns(
                df_conn_split,
                **cfg.split_cfg,
            )
            df_conn_normalized = df_contact_normalized.copy()
            df_conn_normalized[col_bak] = df_conn_normalized[col_old]
            df_conn_normalized_split = split_rows_by_columns(
                df_conn_normalized,
                **cfg.split_cfg,
            )
            save_path = save_dir / f"{df_path.stem}_contact_split.pkl"
            save_df_with_check(df_conn_split, save_path)
            save_path = save_dir / f"{df_path.stem}_contact_normalized_split.pkl"
            save_df_with_check(df_conn_normalized_split, save_path)

 
#%% 8) parallel processing
logs = parallel_tqdm(
    df_paths,
    func = process_stats_df,
    n_jobs=int(os.environ.get("MYLFP_STATS_JOBS", "-1")),
    verbose=True,
)

time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = save_logs(logs, stats_dir / f"run_logs_{time_now}.jsonl")
print("Saved:", log_path)
if any(not log["ok"] for log in logs):
    raise RuntimeError(f"Statistics input generation failed; inspect {log_path}")

# %%
