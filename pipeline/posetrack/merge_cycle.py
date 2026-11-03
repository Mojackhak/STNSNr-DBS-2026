"""
Merge the stand (baseline) and the gait cycle feature to gait_feature derived lfp
"""

#%% Importing modules
# importing modules
import os
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.utils.paths as paths
import src.utils.fileops as fileops
from src.tabular.selection import slice_df
from src.io.pkl_io import load_pkl, save_pkl
from src.posetrack.gait_burst_time import (
    CYCLE_DURATION_SEC_COL,
    attach_cycle_duration,
    convert_percent_axis_burst_time,
)
import plugin.default.merge_cycle_default as cfg
from plugin.default.tabular_default import standardize_df

import importlib
importlib.reload(cfg)


def _env_list(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}

#%% Set the project folder
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

summary_dir = project / 'summary'
table_dir = summary_dir / 'table'
derive_dir = table_dir / 'derive'
posetrack_dir = table_dir / "posetrack"

df_features_path = posetrack_dir / r'gait_cycle_features.pkl'
df_feature = load_pkl(df_features_path)

#%% 1) Melt df_feature first (tidy), then merge

cols_info = ["Subject", "Task", "Epoch"]
df_feature_long = df_feature.melt(
    id_vars=cols_info,
    var_name="Feature",
    value_name="Value_feature",
)
    
#%% merge with gait cycle features and add stand baseline feature value
gait_feature_dir = derive_dir / 'gait_feature'
gait_feature_merged_dir = fileops.create_subfolder(derive_dir, 'gait_feature_merged')
pose_dir = derive_dir / 'pose'
df_paths = paths.get_files_list(gait_feature_dir, filetypes='.pkl', recursive=True)
target_stems = _env_list("MYLFP_MERGE_CYCLE_STEMS")
if target_stems:
    df_paths = [df_path for df_path in df_paths if df_path.stem in target_stems]

for df_path in df_paths:
    df = load_pkl(df_path)

    df_long = df.merge(
        df_feature_long,
        on=cols_info,
        how="left",
        validate="many_to_many",
    )

    df_long = attach_cycle_duration(
        df_long,
        df_feature,
        keys=cols_info,
    )
    df_long = convert_percent_axis_burst_time(
        df_long,
        source_stem=df_path.stem,
    )
    df_long.drop(columns=[CYCLE_DURATION_SEC_COL], inplace=True)
        
    # 2) Load baseline table and reduce it to one value per key
    path_rel = df_path.relative_to(gait_feature_dir)
    df_pose = load_pkl(pose_dir / path_rel)

    cols_basic = ["ID", "Contact", "Band", "Task"]

    df_baseline = (
        slice_df(df_pose, cfg.baseline_pose_subset)
        .loc[:, cols_basic + ["Value"]]
        .groupby(cols_basic, as_index=False)["Value"]
        .mean()
    )

    # 3) Build two conditions (Walk vs Stand) without copy+manual juggling
    df_walk = df_long.assign(Pose="Walk")

    df_stand = (
        df_long.drop(columns=["Value"], errors="ignore")
        .merge(df_baseline, on=cols_basic, how="left", validate="many_to_one")
        .assign(Pose="Stand")
    )
        
    df_merged = pd.concat([df_walk, df_stand], ignore_index=True)
    df_merged.drop(columns=["Subject"], inplace=True)
    df_merged['Value_stand'] = pd.concat([df_stand, df_stand], ignore_index=True)['Value']
    df_merged = standardize_df(df_merged)

    save_path = gait_feature_merged_dir / path_rel
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(df_merged, save_path)
    print("Saved merged gait feature dataframe to:", save_path)
        
    
# %%
