
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
from src.tabular.text import capitalize_df, normalize_side
from src.io.pkl_io import load_pkl, save_pkl

import importlib

#%% Set the project folder
project = Path(os.environ["MYLFP_PROJECT_ROOT"])

deri = project / r"derivatives"
posetrack_dir = deri / r"posetrack"
mne_dir = deri / r"mne"

subjects = [d for d in mne_dir.iterdir() if d.is_dir()]
subjects = sorted(subjects)

paradigms = [r"turn_L_feature", r"turn_R_feature", 
             r"turn_dual_L_feature", r"turn_dual_R_feature"]

#%% Process each subject and paradigm
df_features_concat = []
for subject_dir in subjects:
    subject = subject_dir.name
    print("Processing subject:", subject)    
    for paradigm in paradigms:
        epochs_path = mne_dir / subject / 'stitch' / 'warped' / paradigm / 'warp_labels.pkl'
        if epochs_path.exists():
            print("Processing paradigm:", paradigm) 
            epochs = load_pkl(epochs_path)
        else:
            continue
        
        features_dict = {'Epoch':[], 'Pose':[], 'Task':[], 'Side_pose':[], 'Duration':[],}
        for i, ep in enumerate(epochs['ALL']):
            label = ep.label
            pose, side = label.rsplit('_',1)
            if pose == 'turn':
                pose = 'Turn'
                task = 'Single'
            else:
                pose = 'Turn'
                task = 'Dual'   
            side = normalize_side(side)             
            duration = ep.total_duration_s
            features_dict['Epoch'].append(i)
            features_dict['Pose'].append(pose)
            features_dict['Task'].append(task)
            features_dict['Side_pose'].append(side)
            features_dict['Duration'].append(duration)
            
        df_features = pd.DataFrame(features_dict)
        
        if df_features.empty:
            print(f"  No features extracted for {subject} {paradigm}, skipping...")
            continue
        
        df_features['Subject'] = subject
        cols_basic = ['Subject', 'Pose', 'Side_pose', 'Task', 'Epoch', 'Duration']
        df_features = df_features[cols_basic]
        df_features_concat.append(df_features)

# %% concatenate features
df_features_concat = pd.concat(df_features_concat, axis=0, ignore_index=True)
df_features_concat = capitalize_df(df_features_concat)

# %% save concatenated features
table_dir = project / 'summary' / 'table'
save_dir = fileops.create_subfolder(table_dir, 'posetrack')
save_path = save_dir / 'turn_features.pkl'
save_pkl(df_features_concat, save_path)

# %%
