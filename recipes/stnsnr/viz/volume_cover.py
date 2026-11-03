#%%
import os
import pandas as pd
import numpy as np
import re
import nibabel as nib
import pickle
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))
import src.utils.paths as paths
import src.utils.fileops as fileops
import plugin.default.viz_defaults as cfg
from src.io.pkl_io import load_pkl, save_pkl
from src.tabular.selection import slice_df
from src.tabular.reshape import split_rows_by_columns
import matplotlib.pyplot as plt
from src.anat.voxelize import map_property_to_mni

import importlib
importlib.reload(cfg)

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
summary_dir = project / r'summary'
stats_dir = summary_dir / r'stats'
cohort_dir = summary_dir / r'cohort'
paradigm_path = cohort_dir / r'subj'/ r'subj_paradigm.xlsx' 
df_paradigm = pd.read_excel(paradigm_path)
sections = {'rest': {'Sit': 2}, 
            'med': {'On': -2},
            'walk': {'Stand': 3, 'Walk': 4, 'Turn': 5},
            'pain': {'Pain': -1}}

lead_coords_path = cohort_dir / 'lead' / 'contact_space_locs.pkl'
lead_coords_paired_path = cohort_dir / 'lead' / 'contact_space_locs_paired.pkl'

df_lead_coords = pd.read_pickle(lead_coords_path)
df_lead_coords_paired = pd.read_pickle(lead_coords_paired_path)

subset = {'Region': ['SNr', 'STN', ("SNr", "STN")],
          'Side': ['Left', 'Right', ('Left', 'Left'), ('Right', 'Right')]}
df_lead_coords = slice_df(df_lead_coords, subset)
df_lead_coords_paired = slice_df(df_lead_coords_paired, subset)
df_lead_coords_paired["Region0"] = df_lead_coords_paired["Region"]
cols_to_split = [col for col in df_lead_coords_paired.columns if col not in ['ID', 'Subject', 'Region0']]
df_lead_coords_paired = split_rows_by_columns(
    df_lead_coords_paired,
    cols_to_split=cols_to_split,
)
df_lead_dict = {'single': df_lead_coords,
                'double': df_lead_coords_paired}

_, mask_dict, _ = cfg.load_atlas(project) # load atlas

#%%
def plot_volume_cover(section: str, mode: str):

    section_dir = stats_dir / section
    topomap_dir = fileops.create_subfolder(section_dir, 'topomap')
    topomap_dir = fileops.create_subfolder(topomap_dir, 'cover')
    mode_dir = fileops.create_subfolder(topomap_dir, mode)
    for var in sections[section].keys():
        print(f" Processing variable: {var}")
        df_section = df_paradigm.iloc[1:-1, [0,sections[section][var]]].copy()
        df_section = df_section[df_section.iloc[:,1]=='✓'].copy()
        df_section = df_section.dropna().reset_index(drop=True)
        df_merge = df_lead_dict[mode].merge(
            df_section,
            on='ID',
            how='right')
        df_merge = df_merge.iloc[:, :-1].copy()  # drop the paradigm column
        df_merge['Value'] = 1.0  # dummy value for plotting
        df_merge = df_merge.drop(columns=['Subject'], errors='ignore')
        df_merge = df_merge.dropna().reset_index(drop=True)
        save_dir = fileops.create_subfolder(mode_dir, var)
        filebase = 'contact_locs'
        save_path = save_dir / f'{filebase}.pkl'
        save_pkl(df_merge, save_path)
        df_merge.to_csv(save_dir / f'{filebase}.csv', index=False)

        region_list = df_merge['Region'].unique()
                
        for region in region_list:
            df_region = df_merge[df_merge['Region'] == region]
            df_region = df_region.dropna().reset_index(drop=True)
            save_dir_region = fileops.create_subfolder(save_dir, region)
            df_region.to_csv(save_dir_region / f'{filebase}.csv', index=False)
            
            print(f" Processing region: {region}")
            
            save_path = save_dir_region / f"{filebase}.pdf"

            _, img, log = map_property_to_mni(
                mask_dict[region], 
                df_region,
                **cfg.voxelization_cfg
            )
            
            save_pkl(log, save_dir_region / f'{filebase}.pkl')
            nib.save(img, save_dir_region / f'{filebase}.nii.gz')

# %%
for section in sections.keys():
    for mode in ['single','double']:
        print(f"Processing section: {section}, mode: {mode}")
        plot_volume_cover(section, mode)

# %%
