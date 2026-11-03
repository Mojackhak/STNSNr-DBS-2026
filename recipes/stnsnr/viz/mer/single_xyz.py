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
import matplotlib
import matplotlib.pyplot as plt
from recipes.stnsnr.viz.mer.bd_2_label_map import bd_2_label_map

import importlib
importlib.reload(cfg)

#%% do not show figures during batch processing
matplotlib.use('Agg')
plt.ioff()

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS
summary_dir = project / r'summary'
stats_dir = summary_dir / r'stats'
section_dir = stats_dir / r"mer"
wk_dirs = [section_dir / r"spike",]

#%% read data

for wk_dir in wk_dirs:
    if not wk_dir.is_dir():
        continue
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
    
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='spike_unit_raw.pkl')[0]
    df = pd.read_pickle(df_path)
    
    #%%
    bd_list = df['Param'].unique().tolist()

    #%%
    for bd in bd_list:
        print(f" Processing scale: {bd}")
        df_band = df[df['Param'] == bd].copy()
        df_band.reset_index(drop=True, inplace=True)
        
        save_dir_bd = wk_dir / bd
        
        df_band = cfg.zscore_by_group(df_band, group="Region")
        df_band.to_excel(save_dir_bd / f'scale_subject_{bd}.xlsx', index=False)
        
        #%%
        save_dir_model_list = paths.get_files_in_structure(save_dir_bd, search_pattern="*_model")
        
        #%%
        for save_dir_model in save_dir_model_list:
            print(f"  Processing model results in: {save_dir_model.stem}")
             
            #%% Define wrapper functions for plotting
            def plot_scalar_fit_wrapper(
                x_var: str,
            ) -> plt.Figure:
                
                curve = pd.read_csv(save_dir_model / f'{x_var}_curve.csv')
                slope = pd.read_csv(save_dir_model / f'{x_var}_slope.csv')  
                config_updated = cfg.update_fit_cfg(1)
                axis = x_var.lower()[-1]
                config_updated['x_label'] = f"MNI-{axis} (z-score)"  
                config_updated['y_label'] = bd_2_label_map.get(bd, bd)
                fig = vdf.plot_single_effect_fit(
                    df=df_band,
                    curve=curve,
                    slope=slope,
                    x_var=x_var,
                    **config_updated
                ) 
                                             
                save_path = save_dir_model / f'{x_var}_fit.pdf'
                cfg.save_fig(fig, save_path)
                
                return fig
            
            #%% 1.1. zMX (β and p-values)
            x_var = 'zMX'
            fig = plot_scalar_fit_wrapper(
                x_var=x_var,
                )

            #%% 2.1. zMY (β and p-values)
            x_var = 'zMY'
            fig = plot_scalar_fit_wrapper(
                x_var=x_var,
                )

            #%% 3.1. zMZ (β and p-values)
            x_var = 'zMZ'
            fig = plot_scalar_fit_wrapper(
                x_var=x_var,
                )     
               
            # %%
            plt.close('all')
