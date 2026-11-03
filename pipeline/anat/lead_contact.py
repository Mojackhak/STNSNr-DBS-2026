#%%
"""
Pipeline: Map Lead-DBS contact coordinates (MNI space) to atlas regions.

Purpose
-------
This script builds a cohort-level table of DBS contact locations by:
1) Loading Lead-DBS exported contact coordinates (MATLAB .mat) in MNI space.
2) Mapping each contact to atlas regions using binary region masks (NIfTI).
3) Aggregating per-subject results into a cohort summary CSV/PKL.
4) (Optional) Pairing contacts within each subject for downstream pairwise analyses.

External prerequisites (MATLAB)
-------------------------------
- Run /pipeline/anat/lead_coords.m:
  * Produces summary/cohort/subject_coords_config.json (if used elsewhere).
  * Produces contact_space_coords.mat for each subject (Lead-DBS reconstruction output).
- Run /pipeline/anat/lead_flip.m:
  * Produces "flipped" coordinates for left hemisphere contacts (L->R), if needed.

Notes
-----
- The atlas mapping is performed by src.anat.mapper (hard-coded Lead-DBS .mat structure).
- This script assumes a BIDS-like project layout with derivatives/leaddbs and config/atlas.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.utils.paths as paths
import src.utils.fileops as fileops
import plugin.default.lead_default as cfg
import src.anat.mapper as mapper
import src.tabular.text as text

import importlib
importlib.reload(cfg)

#%% 1): Finish lead reconstruction in Lead-DBS 

#%% 2): Project-level configuration
# - project: BIDS-style project root
# - region_list / atlas_name / threshold / side_dict: mapping parameters
# - matfilename: Lead-DBS MATLAB export filename
# - save_name: output CSV filename for contact localization
# -----------------------------------------------------------------------------
project = Path(os.environ["MYLFP_PROJECT_ROOT"])
region_list = cfg.region_list
atlas_name = cfg.atlas_name
threshold = cfg.threshold
side_dict = cfg.side_dict

matfilename = '*-reconstruction.mat'
save_name = r"contact_reco_space_locs.csv"

# -----------------------------------------------------------------------------
# Path conventions
# - lead_dir: where Lead-DBS reconstructions live (derivatives/leaddbs)
# - atlas_path: atlas root directory containing hemisphere subfolders and region NIfTI masks
# -----------------------------------------------------------------------------
lead_dir = project / 'derivatives' / 'leaddbs'
atlas_path = project / 'config' / 'atlas' / atlas_name

# -----------------------------------------------------------------------------
# File discovery:
# Find all subject-level Lead-DBS reconstruction MAT files matching:
#   */reconstruction/<matfilename>
# Only keep entries that are files and whose parent folder contains 'reconstruction'.
# -----------------------------------------------------------------------------
search_pattern = Path('*') / 'reconstruction' / matfilename
coordsFiles = paths.get_files_in_structure(lead_dir, search_pattern=search_pattern)
coordsFiles = [file for file in coordsFiles if file.is_file() and
               'reconstruction' in file.parent.name]

#%% 3.1): Map contacts to atlas regions (batch over all subjects/files).
# mapper.get_points_loc_batch:
# - For each coordsFile: loads MNI contact coordinates from Lead-DBS .mat.
# - For each hemisphere (side_dict): checks region membership using NIfTI masks.
# - Writes a per-file CSV (save_name) next to each coordsFile.
# Returns a list of CSV paths (one per input coordsFile).
# -----------------------------------------------------------------------------
contact_loc_files = mapper.get_points_loc_batch(
    coordsFiles=coordsFiles,
    region_list=region_list,
    atlas_path=atlas_path,
    threshold=threshold,
    side_dict=side_dict,
    save_name=save_name,
    coords_mode='reco',
    point_names_mode='single',)

#%% 3.2): Aggregate per-subject contact mapping CSVs into a cohort-level table.
# - Reads each per-subject CSV.
# - Extracts subject identifier from the file path (file.parts[-3]).
# - Inserts "subject" column and concatenates all subjects into one DataFrame.
# -----------------------------------------------------------------------------
dfs = pd.DataFrame()
for file in contact_loc_files:
    print(f"Processing {file} ...")
    df = pd.read_csv(file)
    subj = file.parts[-3]  # Extract subject from file path
    subjs = np.array([subj] * df.shape[0])
    df.insert(0, 'subject', subjs)  # Insert subject column at the beginning
    dfs = pd.concat([dfs, df], ignore_index=True)

# Nomralize the dfs
dfs['side'] = dfs['side'].map(text.normalize_side)
dfs = text.capitalize_df(dfs)

save_path = project / 'summary' / 'cohort' / 'lead' / save_name
os.makedirs(save_path.parent, exist_ok=True)
dfs.to_csv(save_path, index=False)

#%% 4.1): Get the filpped coordinates for L side contacts (flip L to R):
# run /pipeline/anat/lead_flip.m to generate flipped coordinates for left contacts
# (commonly used for symmetric hemisphere analyses by mapping L -> R).

dfs = pd.read_csv(save_path) # reload the saved CSV

#%% 5.1): Add symptom lateralization information (OPTIONAL).
# MAH: more-affected hemisphere; LAH: less-affected hemisphere
# (to be filled in based on clinical data)
subj_meta_file = cfg.subj_meta_file
subj_meta_path = project / 'summary' / 'cohort' / 'subj' / subj_meta_file
severe_brain_side_col = cfg.severe_brain_side_col  # column in subj_meta_file indicating side

subj_meta = pd.read_excel(subj_meta_path)
subj_meta['Subject'] = "Sub-" + subj_meta['NameEn'].astype(str)
subj_meta2merge = subj_meta[['ID', 'Subject', severe_brain_side_col]].copy()
subj_meta2merge[severe_brain_side_col] = subj_meta2merge[severe_brain_side_col].map(text.normalize_side)

#%% 5.2): Merge lateralization info into the cohort DataFrame.
dfs_merged = dfs.merge(subj_meta2merge, on='Subject', how='left')
dfs_merged['Hemi'] = np.where(
    dfs_merged[severe_brain_side_col] == "Bilat", 'Bilat', 
    np.where(
      dfs_merged['Side'] == dfs_merged[severe_brain_side_col], 
      'MAH', 'LAH'))

#%% 6.1): Rearrange and clean up final cohort DataFrame.
# dfs_merged = dfs_merged.drop(columns=['Subject', severe_brain_side_col])
dfs_merged = dfs_merged.drop(columns=[severe_brain_side_col]) # keep Subject for now
cols_basic = [col for col in cfg.cols_basic if col in dfs_merged.columns]
dfs_merged = dfs_merged[cols_basic + [col for col in dfs_merged.columns if col not in cols_basic]]

#%% 7.1) Save cohort-level contact localization CSV/PKL.
# - Writes cohort-level CSV to summary/cohort/lead/<save_name>.
# -----------------------------------------------------------------------------
dfs_merged.to_csv(save_path, index=False)
dfs_merged.to_pickle(save_path.with_suffix('.pkl'))

# %%
