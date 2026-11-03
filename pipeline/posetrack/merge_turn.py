"""
Merge the stand (baseline) and the turn feature to turn_feature derived lfp
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
import plugin.default.merge_turn_default as cfg
from plugin.default.tabular_default import standardize_df

import importlib
importlib.reload(cfg)

#%% Set the project folder
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

summary_dir = project / 'summary'
table_dir = summary_dir / 'table'
derive_dir = table_dir / 'derive'
posetrack_dir = table_dir / "posetrack"

df_features_path = posetrack_dir / r'turn_features.pkl'
df_feature = load_pkl(df_features_path)

#%% 1) Melt df_feature first (tidy), then merge

cols_info = ["Subject", "Pose", "Side_pose", "Task", "Epoch"]
df_feature_long = df_feature.melt(
    id_vars=cols_info,
    var_name="Feature",
    value_name="Value_feature",
)
    
#%% merge with turn features and add stand baseline feature value
turn_feature_dir = derive_dir / 'turn_feature'
turn_feature_merged_dir = fileops.create_subfolder(derive_dir, 'turn_feature_merged')
pose_dir = derive_dir / 'pose'
df_paths = paths.get_files_list(turn_feature_dir, filetypes='.pkl', recursive=True)

for df_path in df_paths:
    
    df = load_pkl(df_path)
    
    df_long = df.merge(
        df_feature_long,
        on=cols_info,
        how="left",
        validate="many_to_many",
    )
    
    if df_path.stem == 'burst-duration-scalar':
        df_long['Value'] = df_long['Value'] - np.log(100 / df_long['Value_feature'])
    elif df_path.stem == 'burst-rate-scalar.pkl':
        df_long['Value'] = np.arcsinh(np.sinh(df_long['Value']) * 100 / df_long['Value_feature'])

    # 2) Load baseline table and reduce it to one value per key
    path_rel = df_path.relative_to(turn_feature_dir)
    df_pose = load_pkl(pose_dir / path_rel)

    cols_basic = ["ID", "Contact", "Band", "Task"]

    df_baseline = (
        slice_df(df_pose, cfg.baseline_pose_subset)
        .loc[:, cols_basic + ["Value"]]
        .groupby(cols_basic, as_index=False)["Value"]
        .mean()
    )

    # 3) Build two conditions (Turn vs Stand) without copy+manual juggling
    df_turn = df_long.assign(Pose="Turn")

    df_stand = (
        df_long.drop(columns=["Value"], errors="ignore")
        .merge(df_baseline, on=cols_basic, how="left", validate="many_to_one")
        .assign(Pose="Stand")
    )

    df_merged = pd.concat([df_turn, df_stand], ignore_index=True)
    df_merged.drop(columns=["Subject"], inplace=True)
    df_merged['Value_stand'] = pd.concat([df_stand, df_stand], ignore_index=True)['Value']
    df_merged = standardize_df(df_merged)

    save_path = turn_feature_merged_dir / path_rel
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(df_merged, save_path)
    print("Saved merged turn feature dataframe to:", save_path)
    
    
        
    
# %%
