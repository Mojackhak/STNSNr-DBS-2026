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
section_dir = stats_dir / r"pain"
params_dir_list = [section_dir / r"tfr", ]
wk_dirs = []
for params_dir in params_dir_list:
    wk_dirs.extend(paths.get_files_in_structure(params_dir, search_pattern="*/*-spectral"))

#%% read data

for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
   
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='region_raw.pkl')[0]
    df = load_pkl(df_path)
    
    df_norm_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='region_normalized.pkl')[0]
    df_norm = load_pkl(df_norm_path)
    
    df = cfg.format_force(df)
    df_norm = cfg.format_force(df_norm)     
    
    #%% create save folder
    save_dir = fileops.create_subfolder(wk_dir, 'All')
        
    #%% Define wrapper functions for plotting
    def plot_series_raw_wrapper(
        x_var: str,
        panel_var: str | None = None,
        facet_var: str | None = None,
        df_plot: pd.DataFrame | None = None,
        panel_levels: list | None = None,
        facet_levels: list | None = None,
        output_filebase: str | None = None,
    ) -> plt.Figure:
        
        fig = cfg.plot_series_wrapper(
            df=df if df_plot is None else df_plot,
            df_type='raw',
            save_dir=save_dir,
            param_type=param_type,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            output_filebase=output_filebase,
        )
        
        return fig
    
    def plot_series_norm_wrapper(
        x_var: str,
        panel_var: str | None = None,
        facet_var: str | None = None,
        df_plot: pd.DataFrame | None = None,
        panel_levels: list | None = None,
        facet_levels: list | None = None,
        output_filebase: str | None = None,
    ) -> plt.Figure:
        
        fig = cfg.plot_series_wrapper(
            df=df_norm if df_plot is None else df_plot,
            df_type='norm',
            save_dir=save_dir,
            param_type=param_type,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            output_filebase=output_filebase,
        )
        
        return fig

    def iter_phase_region_split_specs(
        df_plot: pd.DataFrame,
        x_var: str,
        panel_var: str | None = None,
        facet_var: str | None = None,
    ) -> list[dict]:
        split_var_groups = [(cfg.REGION_VAR,)]
        if x_var != cfg.PHASE_VAR:
            split_var_groups.extend([
                (cfg.PHASE_VAR,),
                (cfg.REGION_VAR, cfg.PHASE_VAR),
            ])

        specs = []
        for split_vars in split_var_groups:
            specs.extend(cfg.iter_split_level_specs(
                df_plot,
                x_var=x_var,
                panel_var=panel_var,
                facet_var=facet_var,
                split_vars=split_vars,
            ))
        return specs

    def plot_series_raw_with_phase_region_splits(
        x_var: str,
        panel_var: str | None = None,
        facet_var: str | None = None,
    ) -> plt.Figure:
        fig = plot_series_raw_wrapper(
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
        )
        for split in iter_phase_region_split_specs(
            df,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
        ):
            fig = plot_series_raw_wrapper(
                x_var=x_var,
                panel_var=panel_var,
                facet_var=facet_var,
                df_plot=split["df"],
                panel_levels=split["panel_levels"],
                facet_levels=split["facet_levels"],
                output_filebase=split["output_filebase"],
            )
        return fig

    def plot_series_norm_with_phase_region_splits(
        x_var: str,
        panel_var: str | None = None,
        facet_var: str | None = None,
    ) -> plt.Figure:
        fig = plot_series_norm_wrapper(
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
        )
        for split in iter_phase_region_split_specs(
            df_norm,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
        ):
            fig = plot_series_norm_wrapper(
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
    fig = plot_series_raw_with_phase_region_splits(
        x_var=x_var,
        panel_var=panel_var,
        )

    #%% 1.2. Phase-Region-Lat
    x_var = 'Phase'
    panel_var = 'Region'
    facet_var = 'Lat'
    fig = plot_series_raw_with_phase_region_splits(
        x_var=x_var,
        panel_var=panel_var,
        facet_var=facet_var
        )
    
    #%% 2.1. Lat-Phase-Region
    x_var = 'Lat'
    panel_var = 'Phase'
    facet_var = 'Region'
    fig = plot_series_norm_with_phase_region_splits(
        x_var=x_var,
        panel_var=panel_var,
        facet_var=facet_var
        )
    
    #%% 3.1. Force-Phase-Region 
    x_var = 'Force'
    panel_var = 'Phase'
    facet_var = 'Region'
    fig = plot_series_norm_with_phase_region_splits(
        x_var=x_var,
        panel_var=panel_var,
        facet_var=facet_var
        )    
    
    # %%
    plt.close('all')
