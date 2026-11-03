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
section_dir = stats_dir / r"pain_evoked"
params_dir_list = [section_dir / r"tfr", 
                   section_dir / r"params", 
                   section_dir / r"burst"]
wk_dirs = []
for params_dir in params_dir_list:
    wk_dirs.extend(paths.get_files_in_structure(params_dir, search_pattern="*/*-mean-*"))

#%% read data

for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    print(f"Processing data: {wk_dir_name}")

    #%%
   
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='region_raw.pkl')[0]
    df = load_pkl(df_path)
    
    #%%
    bd_list = df['Band'].cat.categories.tolist()

    #%%
    for bd in bd_list:
        df_band = df[df['Band'] == bd]
        df_band.reset_index(drop=True, inplace=True)
        
        save_dir_bd = wk_dir / bd
        
        #%%
        save_dir_model_list = paths.get_files_in_structure(save_dir_bd, search_pattern="*_model")
        
        #%%
        for save_dir_model in save_dir_model_list:
            print(f"  Processing model results in: {save_dir_model.stem}")
             
            #%% Define wrapper functions for plotting
            def plot_scalar_raw_wrapper(
                x_var: str,
                panel_var: str | None = None,
                facet_var: str | None = None,
                df_plot: pd.DataFrame | None = None,
                panel_levels: list | None = None,
                facet_levels: list | None = None,
                output_filebase: str | None = None,
            ) -> plt.Figure:
                
                fig = cfg.plot_scalar_pain_evoked_wrapper(
                    df=df_band if df_plot is None else df_plot,
                    df_type='raw',
                    save_dir=save_dir_model,
                    wk_dir_name=wk_dir_name,
                    bd=bd,
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                    panel_levels=panel_levels,
                    facet_levels=facet_levels,
                    output_filebase=output_filebase,
                )
                
                return fig
            
            def plot_scalar_fit_wrapper(
                x_var: str,
                panel_var: str | None = None,
                facet_var: str | None = None,
                df_plot: pd.DataFrame | None = None,
                panel_levels: list | None = None,
                facet_levels: list | None = None,
                output_filebase: str | None = None,
            ) -> plt.Figure:
                
                fig = cfg.plot_fit_pain_evoked_wrapper(
                    df=df_band if df_plot is None else df_plot,
                    df_type='norm',
                    save_dir=save_dir_model,
                    wk_dir_name=wk_dir_name,
                    bd=bd,
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                    x_label=r"$\log_{10}\mathrm{Force}\ (g)$",
                    panel_levels=panel_levels,
                    facet_levels=facet_levels,
                    output_filebase=output_filebase,
                )
                
                return fig

            def plot_scalar_raw_with_region_splits(
                x_var: str,
                panel_var: str | None = None,
                facet_var: str | None = None,
            ) -> plt.Figure:
                fig = plot_scalar_raw_wrapper(
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                )
                for split in cfg.iter_split_level_specs(
                    df_band,
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                    split_vars=(cfg.REGION_VAR,),
                ):
                    fig = plot_scalar_raw_wrapper(
                        x_var=x_var,
                        panel_var=panel_var,
                        facet_var=facet_var,
                        df_plot=split["df"],
                        panel_levels=split["panel_levels"],
                        facet_levels=split["facet_levels"],
                        output_filebase=split["output_filebase"],
                    )
                return fig

            def plot_scalar_fit_with_region_splits(
                x_var: str,
                panel_var: str | None = None,
                facet_var: str | None = None,
            ) -> plt.Figure:
                fig = plot_scalar_fit_wrapper(
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                )
                for split in cfg.iter_split_level_specs(
                    df_band,
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                    split_vars=(cfg.REGION_VAR,),
                ):
                    fig = plot_scalar_fit_wrapper(
                        x_var=x_var,
                        panel_var=panel_var,
                        facet_var=facet_var,
                        df_plot=split["df"],
                        panel_levels=split["panel_levels"],
                        facet_levels=split["facet_levels"],
                        output_filebase=split["output_filebase"],
                    )
                return fig

            def plot_scalar_fit_with_phase_region_splits(
                x_var: str,
                panel_var: str | None = None,
                facet_var: str | None = None,
            ) -> plt.Figure:
                fig = plot_scalar_fit_with_region_splits(
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                )
                for split_vars in (
                    (cfg.PHASE_VAR,),
                    (cfg.REGION_VAR, cfg.PHASE_VAR),
                ):
                    for split in cfg.iter_split_level_specs(
                        df_band,
                        x_var=x_var,
                        panel_var=panel_var,
                        facet_var=facet_var,
                        split_vars=split_vars,
                    ):
                        fig = plot_scalar_fit_wrapper(
                            x_var=x_var,
                            panel_var=panel_var,
                            facet_var=facet_var,
                            df_plot=split["df"],
                            panel_levels=split["panel_levels"],
                            facet_levels=split["facet_levels"],
                            output_filebase=split["output_filebase"],
                        )
                return fig
            
            #%% 1.1. Phase-Region 
            x_var = 'Phase'
            panel_var = 'Region'
            fig = plot_scalar_raw_with_region_splits(
                x_var=x_var,
                panel_var=panel_var,
                )

            #%% 2.1. LogForce-Phase-Region(β and p)
            x_var = 'LogForce'
            panel_var = 'Phase'
            facet_var = 'Region'
            fig = plot_scalar_fit_with_phase_region_splits(
                x_var=x_var,
                panel_var=panel_var,
                facet_var=facet_var
                )            
            
            # %%
            plt.close('all')
