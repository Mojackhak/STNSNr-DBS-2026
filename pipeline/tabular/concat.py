"""Merge warped tensor outputs across subjects into analysis-ready DataFrames.

This script is intended to be run interactively in IPython/Jupyter using cell
markers (#%%). The goal is to:

- Scan all subjects under a derivatives/mne folder.
- For each subject and each warped paradigm folder (stitch*/warped/<paradigm>/),
  load tensors_warped.pkl.
- Extract a configured subset of tensor features (see plugin.default.concat_default).
- Convert each 4D tensor (epoch, channel, freq, time) into a long-form
  table where each (epoch, channel) row contains a nested 2D DataFrame
  (freq x time).
- Attach lead/contact metadata by merging with cohort lead coordinate tables.
- Export the merged tables as .pkl DataFrames.

Notes
-----
- This script assumes each selected feature tensor is 4D and has corresponding
  axes in tensor['meta']['axes'].
- Paradigm grouping is defined in plugin.default.concat_default.
"""

#%% 0) Imports & interactive setup
import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.utils.paths as paths
import src.utils.fileops as fileops
from src.tabular.text import capitalize_df
from src.tabular.tensor_slices import split_tensor4d_to_nested_df as tensor2df
# import plugin.default.concat_default as cfg
# import plugin.default.concat_posetrack_default as cfg
import plugin.default.concat_turn_default as cfg

from src.io.pkl_io import load_pkl, save_pkl

import importlib
importlib.reload(cfg)

#%% 1) Project paths and cohort-level metadata
#   - Define dataset root paths.
#   - Load cohort lead/contact metadata tables used for merging.
#   - Create output folder for the exported tables.

project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
deri = project / 'derivatives' / 'mne'
subject_dir_list = paths.get_files_in_structure(deri, '*/')
subject_dir_list = sorted(subject_dir_list)

lead_coords_path = project / 'summary' / 'cohort' / 'lead' / 'contact_space_locs.pkl'
lead_coords_paired_path = project / 'summary' / 'cohort' / 'lead' / 'contact_space_locs_paired.pkl'

df_lead_coords = pd.read_pickle(lead_coords_path)
df_lead_coords_paired = pd.read_pickle(lead_coords_paired_path)

summary_dir = project / 'summary'
table_dir = fileops.create_subfolder(summary_dir, 'table')
concat_dir = fileops.create_subfolder(table_dir, 'concat')
raw_dir = fileops.create_subfolder(table_dir, 'raw')

overwrite = True  # whether to overwrite existing raw dataframes

#%% 2) Initialize collectors (nested dict of lists)
#   - Build nested containers to accumulate per-subject DataFrames.
#   - The nested structure is:
#       <group> -> <feature> -> <subfeature> -> list[pd.DataFrame]
#   - Grouping is defined by cfg.epoch_locked and cfg.state_block.

# Used only for key membership; do not append into this template.
feature_dict = cfg.make_features_dict()

# Each call creates a fresh dict with fresh lists (no shared references).
paradigm_dict = {f: cfg.make_features_dict() for f in cfg.paradigm_dict.keys()}

#%% 3) Scan subjects and accumulate per-paradigm tables
#   - Iterate each subject and each warped paradigm.
#   - Load tensors_warped.pkl, extract configured tensors, convert to DataFrames.
#   - Append rows into the correct group container.

for subject_dir in subject_dir_list:
    subject = subject_dir.name
    print("Processing subject:", subject)
    warp_dir_list = paths.get_files_in_structure(subject_dir, 'stitch*/warped/*/')
    for warp_dir in warp_dir_list:
        paradigm = warp_dir.name
        is_paradigm = any([paradigm == val 
                           for key in paradigm_dict.keys() 
                           for val in cfg.paradigm_dict[key]])
        if not is_paradigm:
            continue
        else:
            paradigm_key = [k for k in paradigm_dict.keys() 
                            if any(paradigm == val for val in cfg.paradigm_dict[k])][0]
        tensors_path = warp_dir / 'tensors_warped.pkl'
        if not tensors_path.exists():
            print("  Warning: tensors_warped.pkl not found in", warp_dir)
            continue
        tensors = load_pkl(tensors_path)
        for params in tensors.keys():
            tensor = tensors[params]
            if params in feature_dict.keys():
                subfeature_dict = feature_dict[params]
                for subparam in tensor.keys():
                    if subparam in subfeature_dict.keys():
                        tensor_feature = tensor[subparam]
                        epoch = tensor['meta']['axes']['epoch']
                        epoch = list(zip(epoch, np.arange(len(epoch))))
                        channel = tensor['meta']['axes']['channel']
                        freq = tensor['meta']['axes']['freq']
                        time_axis = cfg.time_axis.get(paradigm, cfg.time_axis['default'])
                        time = tensor['meta']['axes'][time_axis]
                        rows = tensor2df(
                            tensor_feature,
                            epoch,
                            channel,
                            freq,
                            time,
                        )
                        rows['Subject'] = subject
                        rows['Paradigm'] = paradigm
                        paradigm_dict[paradigm_key][params][subparam].append(rows)
                        

#%% 4) Merge lists into DataFrames and attach lead/contact metadata
#   - Merge list[pd.DataFrame] -> pd.DataFrame for each leaf node.
#   - Normalize column naming.
#   - Merge lead coordinates/labels onto each row by (Subject, Contact).

def merge_nested_lists(obj, merge_fn):
    """
    Walk a nested structure (dict/list/other). When encountering a list, merge it.
    """
    if isinstance(obj, dict):
        return {k: merge_nested_lists(v, merge_fn=merge_fn) for k, v in obj.items()}

    if isinstance(obj, list):
        return merge_fn(obj)
    
    return obj
                
def merge_metadata(df_list):
    if len(df_list) == 0:
        return pd.DataFrame()
    else:
        is_paired = isinstance(df_list[0].at[0,'channel'], tuple)
        # normalize df
        df_list = [capitalize_df(df) for df in df_list]
        dfs = pd.concat(df_list, axis=0, ignore_index=True)
        dfs.rename(columns={'Channel':'Contact'}, inplace=True)
        if is_paired:
            dfs = df_lead_coords_paired.merge(dfs, on=['Subject','Contact'], how='right')
        else:
            dfs = df_lead_coords.merge(dfs, on=['Subject','Contact'], how='right')
        return dfs

paradigm_dict_merged = merge_nested_lists(paradigm_dict, merge_metadata)

paradigm_save_path = concat_dir / 'paradigm_dict_merged.pkl'

if overwrite or not paradigm_save_path.exists():
    save_pkl(paradigm_dict_merged, paradigm_save_path)
    
paradigm_dict_merged = load_pkl(paradigm_save_path)  

#%% 5) Save merged DataFrames
# Purpose:
#   - Persist merged outputs into a folder structure:
#       summary/table/raw/<paradigm>/<params>/<subparam>/<params>_<subparam>.pkl

def save_merged_dict(merged_dict):
    for paradigm in merged_dict.keys():
        paradigm_dir = fileops.create_subfolder(raw_dir, paradigm)
        paradigm_dict = merged_dict[paradigm]
        for params in paradigm_dict.keys():
            params_dir = fileops.create_subfolder(paradigm_dir, params)
            params_dict = paradigm_dict[params]
            for subparam in params_dict.keys():
                df_dir = fileops.create_subfolder(params_dir, subparam)
                save_path = df_dir / f"{subparam}-na-raw.pkl"
                if save_path.exists() and not overwrite:
                    print("  merged dataframe already exists. Skipping...", save_path)
                    continue
                df = params_dict[subparam]
                if df.empty:
                    print("  Warning: empty dataframe for", save_path)
                    continue
                normalize_func = cfg.normalize_func_dict.get(
                    paradigm, cfg.normalize_func_dict['default'])
                df = normalize_func(df)
                                    
                save_pkl(df, save_path)

save_merged_dict(paradigm_dict_merged)
            
# %%
