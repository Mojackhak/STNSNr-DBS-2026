#%%
import os
import pandas as pd
import numpy as np
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

import importlib
importlib.reload(cfg)

#%% do not show figures during batch processing
matplotlib.use('Agg')
plt.ioff()

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
summary_dir = project / r'summary'
stats_dir = summary_dir / r'stats'
section_dir = stats_dir / r"gait_phase"
params_dir_list = [section_dir / r"conn", 
                   section_dir / r"gc", 
                   section_dir / r"psi"]
wk_dirs = []
for params_dir in params_dir_list:
    wk_dirs.extend(paths.get_files_in_structure(params_dir, search_pattern="*/*-trace"))

#%% read data

for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
   
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='region_normalized.pkl')[0]
    df = load_pkl(df_path)
    df = cfg.with_connection_region_label(df, param_type)
    
    df = cfg.smooth_trace(df)

    #%%
    bd_list = df['Band'].cat.categories.tolist()

    #%%
    for bd in bd_list:
        df_band = df[df['Band'] == bd]
        df_band.reset_index(drop=True, inplace=True)
        
        save_dir = fileops.create_subfolder(wk_dir, bd)  

        #%% Define wrapper functions for plotting
        
        def plot_series_norm_wrapper(
            x_var: str,
            panel_var: str | None = None,
            facet_var: str | None = None,
        ) -> plt.Figure:
            
            fig = cfg.plot_series_wrapper(
                df=df_band,
                df_type='norm',
                save_dir=save_dir,
                param_type=param_type,
                bd=bd,
                x_var=x_var,
                panel_var=panel_var,
                facet_var=facet_var,
                derive_type='trace',
                section='gait',
                append_region=True,
            )
            
            return fig
        
        #%% 1.1. Region
        x_var = 'Region'
        fig = plot_series_norm_wrapper(
            x_var=x_var,
            )
        
        # %%
        plt.close('all')
