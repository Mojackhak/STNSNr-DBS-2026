
#%% Importing modules
# importing modules
import os
import numpy as np
import pandas as pd
import mne
import json
from pathlib import Path

import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import src.utils.paths as paths
import src.utils.fileops as fileops
from src.posetrack.track_io import load_csv_3d
from src.posetrack.pose_sync import crop_posetrack_df_to_filtered_raw as sync_df_to_raw
from src.posetrack.gait_feature import extract_gait_cycle_features as extract_features
from src.posetrack.pose_resample import remsaple_pose_df_fps
from src.lfp.warp.event_anchored_linear import gait_warper
from src.stats.preproc.outliers import filter_df
from src.tabular.text import capitalize_df
from src.io.pkl_io import load_pkl, save_pkl
import plugin.default.cycle_feature_default as cfg

import importlib
importlib.reload(cfg)

#%% Set the project folder
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = project / r"derivatives"
posetrack_dir = deri / r"posetrack"
mne_dir = deri / r"mne"

subjects = [d for d in mne_dir.iterdir() if d.is_dir()]
subjects = sorted(subjects)

paradigms = [r"gait", r"gait_dual"]

#%% Process each subject and paradigm
df_features_concat = []
for paradigm in paradigms:
    for subject_dir in subjects:
        subject = subject_dir.name
        print("Processing subject:", subject)
        
        # Read PoseTrack CSV and mne fif files
        pose_csv = posetrack_dir / subject / paradigm / 'pose-3d-filtered' / 'gait.csv'
        raw_marked_path = mne_dir / subject / paradigm / 'preproc' / f"{subject}_raw_marked_raw.fif"
        raw_path = mne_dir / subject / paradigm / 'preproc' / f"{subject}_bi_raw.fif"

        if any(not p.exists() for p in [pose_csv, raw_marked_path, raw_path]):
            print(f"  Missing files for {subject} {paradigm}, skipping...")
            continue
        
        df_pose, metadata = load_csv_3d(pose_csv)
        raw_marked = mne.io.read_raw_fif(raw_marked_path, preload=True)
        raw = mne.io.read_raw_fif(raw_path, preload=True)

        sr = raw.info['sfreq']

        # Resample PoseTrack dataframe to target FPS

        df_pose_resampled = remsaple_pose_df_fps(
            df_pose,
            fps_new=sr,
            **cfg.resample_cfg,
        )

        #  sync pose dataframe to raw
        df_pose_sync, report = sync_df_to_raw(
            df_pose_resampled,
            posetrack_fps=sr,
            raw=raw_marked,
            **cfg.pose_sync_cfg,
        )
            
        #  Extract gait cycles
        gait_cycle_epochs, _ = gait_warper(raw, **cfg.gait_feature_warp_cfg)

        #  Extract gait cycle features
        df_features = extract_features(
            df_pose_sync,
            gait_cycle_epochs,
            fps=sr,
        )    
        
        feature_dir = fileops.create_subfolder(posetrack_dir / subject / paradigm, 'features')
        save_path = feature_dir / 'gait_cycle_features.pkl'
        save_pkl(df_features, save_path)
        
        if df_features.empty:
            print(f"  No features extracted for {subject} {paradigm}, skipping...")
            continue
        
        df_features['Subject'] = subject
        df_features['Task'] = 'Single' if paradigm == "gait" else "Dual"
        cols_basic = ['Subject', 'Task', 'epoch']
        df_features = df_features[cols_basic + [col for col in cfg.gait_features if col in df_features.columns]]
        df_features_concat.append(df_features)

# %% concatenate features
df_features_concat = pd.concat(df_features_concat, axis=0, ignore_index=True)
df_features_concat = capitalize_df(df_features_concat)

# %% save concatenated features
table_dir = project / 'summary' / 'table'
save_dir = fileops.create_subfolder(table_dir, 'posetrack')
save_path = save_dir / 'gait_cycle_features.pkl'
save_pkl(df_features_concat, save_path)

# %%
