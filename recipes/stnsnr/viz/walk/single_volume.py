#%%
import os
import pandas as pd
import numpy as np
import re
import nibabel as nib
import pickle
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import src.utils.paths as paths
import src.utils.fileops as fileops
import src.viz.visualdf as vdf
import plugin.default.viz_defaults as cfg
from src.io.pkl_io import load_pkl, save_pkl
import matplotlib
import matplotlib.pyplot as plt
from src.anat.voxelize import map_property_to_mni

import importlib
importlib.reload(cfg)

#%%
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
plt.ioff()

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
summary_dir = project / r'summary'
stats_dir = summary_dir / r'stats'
section_dir = stats_dir / r"walk"
params_dir_list = [section_dir / r"tfr", 
                   section_dir / r"params", 
                   section_dir / r"burst"]
wk_dirs = []
for params_dir in params_dir_list:
    wk_dirs.extend(paths.get_files_in_structure(params_dir, search_pattern="*/*-scalar"))

var = 'Pose'
overwrite_nifti = False  # whether to overwrite existing NIfTI heatmaps
img_type = 'norm'

#%% load atlas
bg, mask_dict, mask_plot_dict = cfg.load_atlas(project)
 
#%% read data

for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
   
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='contact_normalized.pkl')[0]
    df = load_pkl(df_path)
    
    #%%
    bd_list = df['Band'].cat.categories.tolist()
    
    #%%
    for bd in bd_list:
        df_band = df[df['Band'] == bd]
        df_band.reset_index(drop=True, inplace=True)
        save_dir_bd = wk_dir / bd
        
        print(f" Processing band: {bd}")
        
        #%%
        region_list = df_band['Region'].cat.categories.tolist()  
              
        #%%
        for region in region_list:
            df_region = df_band[df_band['Region'] == region]
            df_region.reset_index(drop=True, inplace=True)
            save_dir_region = save_dir_bd / region
            
            print(f" Processing region: {region}")
            
            #%% 
            var_list = df_region[var].cat.categories.tolist() 
            
            #%% 
            for var_level in var_list:
                df_var = df_region[df_region[var] == var_level]
                df_var.reset_index(drop=True, inplace=True)
                save_dir_var = save_dir_region / var_level
                
                print(f"  Processing variable level: {var_level}")
                
                filebase = f"{bd}_{region}_{var_level}"
                pdf_path = save_dir_var / f"{filebase}.pdf"
                nii_path = save_dir_var / f"{filebase}.nii.gz"
                pkl_path = save_dir_var / f"{filebase}.pkl"
                
                #%% map to mni
                if not overwrite_nifti and nii_path.exists():
                    print(f"   Reusing existing NIfTI: {nii_path}")
                    img = nib.load(nii_path)
                else:
                    _, img, log = map_property_to_mni(
                        mask_dict[region], 
                        df_var,
                        **cfg.voxelization_cfg
                    )
                    
                    save_pkl(log, pkl_path)
                    nib.save(img, nii_path)
                
                #%% plot nifti images
                fig = cfg.plot_nifti_wrapper(
                    img=img,
                    img_type=img_type,
                    save_path=pdf_path,
                    param_type=param_type,
                    bd=bd,
                    bg=bg
                )

                # %%
                plt.close('all')
