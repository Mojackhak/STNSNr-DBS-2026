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
section_dir = stats_dir / r"fog_episode"
params_dir_list = [section_dir / r"conn", 
                   section_dir / r"gc", ]
wk_dirs = []
for params_dir in params_dir_list:
    wk_dirs.extend(paths.get_files_in_structure(params_dir, search_pattern="*/*-raw"))

#%% read data

for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
   
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='region_normalized.pkl')[0]
    df = load_pkl(df_path)
    df = cfg.with_connection_region_label(df, param_type)
    
    #%% create save folder
    save_dir = fileops.create_subfolder(wk_dir, 'All')
        
    #%% Define wrapper functions for plotting

    def plot_raw_norm_wrapper(
        panel_var: str | None = None,
        facet_var: str | None = None,
        df_plot: pd.DataFrame | None = None,
        panel_levels: list | None = None,
        facet_levels: list | None = None,
        output_filebase: str | None = None,
    ) -> plt.Figure:
        
        fig = cfg.plot_raw_wrapper(
            df=df if df_plot is None else df_plot,
            df_type='norm',
            save_dir=save_dir,
            param_type=param_type,
            panel_var=panel_var,
            facet_var=facet_var,
            section='fog_episode',
            append_region=True,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            output_filebase=output_filebase,
        )
        
        return fig

    def iter_connection_region_splits(
        df_plot: pd.DataFrame,
        panel_var: str | None = None,
        facet_var: str | None = None,
    ) -> list[dict]:
        panel_var_split, facet_var_split, _ = cfg.append_region_layout(
            None,
            panel_var,
            facet_var,
        )
        df_split = cfg.with_connection_region_label(df_plot, param_type)
        return cfg.iter_split_level_specs(
            df_split,
            x_var=None,
            panel_var=panel_var_split,
            facet_var=facet_var_split,
            split_vars=(cfg.REGION_VAR,),
        )

    def plot_raw_norm_with_region_splits(
        panel_var: str | None = None,
        facet_var: str | None = None,
    ) -> plt.Figure:
        fig = plot_raw_norm_wrapper(
            panel_var=panel_var,
            facet_var=facet_var,
        )
        for split in iter_connection_region_splits(
            df,
            panel_var=panel_var,
            facet_var=facet_var,
        ):
            fig = plot_raw_norm_wrapper(
                panel_var=panel_var,
                facet_var=facet_var,
                df_plot=split["df"],
                panel_levels=split["panel_levels"],
                facet_levels=split["facet_levels"],
                output_filebase=split["output_filebase"],
            )
        return fig
    
    #%% 1.1. Region
    fig = plot_raw_norm_with_region_splits()
    
    # %%
    plt.close('all')
