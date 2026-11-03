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
importlib.reload(vdf)

#%% do not show figures during batch processing
matplotlib.use('Agg')
plt.ioff()

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
summary_dir = project / r'summary'
stats_dir = summary_dir / r'stats'
section_dir = stats_dir / r"clinic" / r"pca"
wk_dirs = [section_dir / r"gait", 
           section_dir / r"balance", 
           ]

#%%
def replace_invalid_chars(file_path: Path):
    df = pd.read_csv(file_path)
    df.contrast = df.contrast.str.replace(r'[()]', '', regex=True)
    df.rhs = df.rhs.str.replace(r'[()]', '', regex=True)
    df.group2 = 'STN+SNr'
    df.to_csv(file_path, index=False)
    
#%% read data

for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
   
    df_path = paths.get_files_list(wk_dir, filetypes='.xlsx', contain=r'dist.xlsx')[0]
    df = pd.read_excel(df_path)
        
    #%%
    save_dir_model_list = paths.get_files_in_structure(wk_dir, search_pattern="*_model")
    
    #%%
    for save_dir_model in save_dir_model_list:
        print(f"  Processing model results in: {save_dir_model.stem}")
            
        #%% Define wrapper functions for plotting
        def plot_scalar_raw_wrapper(
            x_var: str,
            panel_var: str | None = None,
            facet_var: str | None = None,
        ) -> plt.Figure:
            
            fig = cfg.plot_scalar_wrapper(
                df=df,
                df_type='raw',
                save_dir=save_dir_model,
                param_type='',
                bd='',
                x_var=x_var,
                panel_var=panel_var,
                facet_var=facet_var,
                x_label=' ',
                y_label='Improvement projection (PC1–PC2)',
                cfg_update = {
                    "xtick_rotation": 45,
                    }
                )
            
            
            return fig
        
        #%% 1.1. Protocol-Phase 
        x_var = 'Protocol'
        panel_var = 'Phase'
        tukey_path = save_dir_model / f'{x_var}-{panel_var}_tukey.csv'
        replace_invalid_chars(tukey_path)        
        fig = plot_scalar_raw_wrapper(
            x_var=x_var,
            panel_var=panel_var,
            )

        #%% 1.2. Phase-Protocol 
        x_var = 'Phase'
        panel_var = 'Protocol'
        tukey_path = save_dir_model / f'{x_var}-{panel_var}_tukey.csv'
        replace_invalid_chars(tukey_path)               
        fig = plot_scalar_raw_wrapper(
            x_var=x_var,
            panel_var=panel_var,
            )
            
        # %%
        plt.close('all')
