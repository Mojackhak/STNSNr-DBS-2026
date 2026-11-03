"""Merge warped tensor outputs across subjects into analysis-ready DataFrames.

This script is intended to be run interactively in IPython/Jupyter using cell
markers (#%%). The goal is to:

- Add MNI coordinates to each unit based on lead/contact positions.
- Map each unit to brain regions using atlas mapping.
- Merge symptom lateralization information.
- Export the final merged DataFrame as CSV and Pickle files for downstream analyses.
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
import src.anat.mapper as mapper
from src.tabular.text import capitalize_df, normalize_side
from src.tabular.selection import slice_df
from src.stats.preproc.outliers import filter_df
from src.stats.preproc.aggregate import summarize_df
from src.stats.preproc.transform import transform_df
from src.tabular.tensor_slices import split_tensor4d_to_nested_df as tensor2df
import plugin.default.coords_default as cfg
import plugin.default.lead_default as lead_cfg
from src.io.pkl_io import load_pkl, save_pkl


import importlib
importlib.reload(cfg)

#%% 1) Project paths and cohort-level metadata
#   - Define dataset root paths.
#   - Load cohort lead/contact metadata tables used for merging.
#   - Create output folder for the exported tables.

project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
deri = project / 'derivatives' / 'spikeinterface'
subject_dir_list = paths.get_files_in_structure(deri, '*/')
subject_dir_list = sorted(subject_dir_list)

lead_offset_path = project / 'summary' / 'cohort' / 'lead' / 'lead_offset.xlsx'
df_lead_offset = pd.read_excel(lead_offset_path)

summary_dir = project / 'summary'
table_dir = fileops.create_subfolder(summary_dir, 'table')
raw_dir = fileops.create_subfolder(table_dir, 'raw')
mer_dir = fileops.create_subfolder(raw_dir, 'mer')
spike_dir = fileops.create_subfolder(mer_dir, 'spike')

region_list=lead_cfg.region_list
atlas_name=lead_cfg.atlas_name
threshold=lead_cfg.threshold

atlas_path = project / 'config' / 'atlas' / atlas_name


#%% 2) Initialize collectors (nested dict of lists)

sorting_files = paths.get_files_in_structure(deri, "*/surgery/waveform/waveform_properties.csv")
dfs = pd.DataFrame()
for file in sorting_files:
    df = pd.read_csv(file)      
    df['subject'] = file.parts[-4]
    dfs = pd.concat([dfs, df], axis=0)
    
dfs = dfs[["subject"] + list(dfs.columns[:-1])]
df_lead_offset["subject"] = df_lead_offset['NameEn'].map(lambda x: 'sub-' + x)
dfs = dfs.merge(df_lead_offset, on=['subject', 'side'], how='left')
dfs = slice_df(dfs, cfg.label_slice_cfg)
dfs = dfs[cfg.properties_cfg]
dfs = dfs.dropna().reset_index(drop=True)

save_path = spike_dir / 'waveform_properties.csv'
dfs.to_csv(save_path, index=False)
    
#%% 3.1): Get units MNI coords according to the lead contact position.
# run /pipeline/anat/lead_table.m outputs:
# - contact_space_coords.mat within each subject's reconstruction folder.
dfs = pd.read_csv(save_path) # reload the saved CSV

#%% 3.2): Map contact coordinates to brain regions using atlas mapping.
dfs = mapper.get_points_loc_df(
    dfs,
    region_list=region_list,
    atlas_path=atlas_path,
    threshold=threshold,
)

dfs.to_csv(save_path, index=False)

#%% 4): Get the filpped coordinates for L side contacts (flip L to R):
# run /pipeline/anat/lead_flip.m to generate flipped coordinates for left contacts
# (commonly used for symmetric hemisphere analyses by mapping L -> R).

dfs = pd.read_csv(save_path) # reload the saved CSV
dfs = capitalize_df(dfs)
dfs['Side'] = dfs['Side'].map(normalize_side)

#%% 5.1): Add symptom lateralization information (OPTIONAL).
# MAH: more-affected hemisphere; LAH: less-affected hemisphere
# (to be filled in based on clinical data)
subj_meta_file = lead_cfg.subj_meta_file
subj_meta_path = project / 'summary' / 'cohort' / 'subj' / subj_meta_file
severe_brain_side_col = lead_cfg.severe_brain_side_col  # column in subj_meta_file indicating side

subj_meta = pd.read_excel(subj_meta_path)
subj_meta['Subject'] = "Sub-" + subj_meta['NameEn'].astype(str)
subj_meta2merge = subj_meta[['ID', severe_brain_side_col]].copy()
subj_meta2merge[severe_brain_side_col] = subj_meta2merge[severe_brain_side_col].map(normalize_side)

#%% 5.2): Merge lateralization info into the cohort DataFrame.
dfs_merged = dfs.merge(subj_meta2merge, on='ID', how='left')
dfs_merged['Hemi'] = np.where(
    dfs_merged[severe_brain_side_col] == "Bilat", 'Bilat', 
    np.where(
      dfs_merged['Side'] == dfs_merged[severe_brain_side_col], 
      'MAH', 'LAH'))

#%% 6.1): Rearrange and clean up final cohort DataFrame.
# dfs_merged = dfs_merged.drop(columns=['Subject', severe_brain_side_col])
dfs_merged = dfs_merged.drop(columns=[severe_brain_side_col]) # keep Subject for now
cols_basic = [col for col in cfg.cols_order if col in dfs_merged.columns]
dfs_merged = dfs_merged[cols_basic + [col for col in dfs_merged.columns if col not in cols_basic]]

#%% 7.1) Export final merged DataFrame 
# -----------------------------------------------------------------------------
dfs_merged.to_csv(save_path, index=False)
dfs_merged.to_pickle(save_path.with_suffix('.pkl'))

#%% 8.1) Derive folder
derive_dir = summary_dir / 'table' / 'derive'
mer_stats_dir = fileops.create_subfolder(derive_dir, 'mer')
spike_dir = fileops.create_subfolder(mer_stats_dir, 'spike')

#%% 9) Convert wide table to long table
id_cols = list(dfs_merged.columns[:22])
value_cols = list(dfs_merged.columns[22:])
df_melt = dfs_merged.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="Param",
    value_name="Value",
)

# %% 9) Transform and filter Value
df_transformed = df_melt.copy()
df_transformed['Value'] = df_transformed['Value'].map(np.abs)
params = df_transformed['Param'].unique()
for param in params:
    mode = cfg.transform_mode_cfg.get(param.lower(), 'none')
    print(f"Transforming param {param} with mode {mode} ...")
    mask = df_transformed['Param'] == param
    df_subset = df_transformed[mask]
    df_subset_transformed = transform_df(
        df_subset,
        mode=mode,
        **cfg.transform_func_cfg,
    )
    df_transformed.loc[mask, 'Value'] = df_subset_transformed['Value']

df_transformed['Region'] = np.where(
    df_transformed['Region'] == 'Mid', 'SNr', df_transformed['Region']
)
df_transformed = slice_df(df_transformed, cfg.derive_slice_cfg)

# %% 9) Export transformed table
save_spike_path = spike_dir / 'spike_unit_transformed.csv'

df_transformed.to_csv(save_spike_path, index=False)
df_transformed.to_pickle(save_spike_path.with_suffix('.pkl'))

#%%
stats_dir = summary_dir / 'stats'
mer_stats_dir = fileops.create_subfolder(stats_dir, 'mer')
spike_stats_dir = fileops.create_subfolder(mer_stats_dir, 'spike')

# %% 9) Filter outliers 
group_cols = ['Region', 'Hemi', 'Label', 'Param']
filter_outliers_cfg = cfg.filter_outliers_cfg
filter_outliers_cfg['alpha'] = 0.999999
df_filtered, report = filter_df(
    df_transformed,
    group_cols=group_cols,
    **cfg.filter_outliers_cfg
)

print("Dropped %d outliers (%.2f%%)" % 
      (report['overall']['n_removed'], 
       report['overall']['pct_removed']*100,))

#%% 10) Export filtered table

save_spike_path = spike_stats_dir / 'spike_unit_raw.csv'
df_filtered.drop(columns=['Subject'], inplace=True)  # remove Subject column before saving
df_filtered.to_csv(save_spike_path, index=False)
df_filtered.to_pickle(save_spike_path.with_suffix('.pkl'))

# %% 10.1) Export unique unit coordinate table
spike_unit_coords_cols = [
    'ID', 'Region', 'Side', 'Hemi', 'Label',
    'MNI_x', 'MNI_y', 'MNI_z',
    'SNr_in', 'SNr_x', 'SNr_y', 'SNr_z',
    'STN_in', 'STN_x', 'STN_y', 'STN_z',
    'MNI_x_flip', 'MNI_y_flip', 'MNI_z_flip',
    'Depth', 'Offset',
]
spike_unit_coords_unique_cols = [
    'ID', 'Region', 'Side', 'Hemi', 'Label',
    'MNI_x', 'MNI_y', 'MNI_z',
]
spike_unit_coords = (
    df_filtered[spike_unit_coords_cols]
    .drop_duplicates(subset=spike_unit_coords_unique_cols, keep='first')
    .reset_index(drop=True)
)
save_spike_coords_path = spike_stats_dir / 'spike_unit_coords.csv'
spike_unit_coords.to_csv(save_spike_coords_path, index=False)
spike_unit_coords.to_pickle(save_spike_coords_path.with_suffix('.pkl'))

# %% 11) Aggregate table
cols_agg = ['ID', 'Region', 'Label', 'Param']
df_agg = summarize_df(df_filtered, group_cols=cols_agg, **cfg.summarize_df_cfg)

# %% 12) Export aggregated table
save_spike_path = spike_stats_dir / 'spike_subject_raw.csv'
df_agg.to_csv(save_spike_path, index=False)
df_agg.to_pickle(save_spike_path.with_suffix('.pkl'))
 
# %%
