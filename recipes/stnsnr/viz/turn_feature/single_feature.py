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
from recipes.stnsnr.viz.turn_feature.combined_feature_fit import plot_combined_feature_fit
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
section_list_dir = [stats_dir / r"turn_feature",]
params = ["tfr", "params", "burst"]
params_dir_list = [section_dir / param for section_dir in section_list_dir
                   for param in params]
wk_dirs = []
for params_dir in params_dir_list:
    wk_dirs.extend(paths.get_files_in_structure(params_dir, search_pattern="*/*-scalar"))

#%%
feature_map = {
    "Duration": "turn duration",
}

#%% read data

for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
    
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='region_normalized.pkl')[0]
    df = load_pkl(df_path)

    # 1) Per (Region, Feature, Band, ID): subject mean and within deviation
    g_id = ["Region", "Feature", "Band", "ID"]
    df["feature_mean_id"] = df.groupby(g_id, observed=True)["Value_feature"].transform("mean")
    df["feature_within"] = df["Value_feature"] - df["feature_mean_id"]

    # 2) Per (Region, Band, Feature): grand mean and between component
    g_stratum = ["Region", "Band", "Feature"]
    df["feature_grand_mean"] = (
        df.groupby(g_stratum, observed=True, group_keys=False)
        .apply(
            lambda g: pd.Series(
                np.nanmean(g.loc[~g["ID"].duplicated(), "feature_mean_id"].to_numpy()),
                index=g.index
            ),
            include_groups=False
        )
    )
    df["feature_between"] = df["feature_mean_id"] - df["feature_grand_mean"]

    # 3) Z-score within each (Region, Band, Feature) stratum (R-like)
    df["z_Feature_within"] = df.groupby(g_stratum, observed=True)["feature_within"].transform(cfg.r_scale_like)
    df["z_Feature_between"] = df.groupby(g_stratum, observed=True)["feature_between"].transform(cfg.r_scale_like)        
    df.to_csv(df_path.with_suffix('.csv'), index=False)
    
    #%%
    feature_list = df["Feature"].unique().tolist()
    
    #%%
    for feature in feature_list:
        
        df_feature = df[df['Feature'] == feature]
        
        #%%
        bd_list = df['Band'].cat.categories.tolist()

        #%%
        for bd in bd_list:
            df_band = df_feature[df_feature['Band'] == bd]
            df_band.reset_index(drop=True, inplace=True)
            
            save_dir_bd = wk_dir / bd
            
            #%%
            save_dir_model_list = paths.get_files_in_structure(save_dir_bd, search_pattern="*_model")
            
            #%%
            for save_dir_model in save_dir_model_list:
                print(f"  Processing model results in: {save_dir_model.stem}")
                
                #%% Define wrapper functions for plotting
                def plot_scalar_fit_wrapper(
                    x_var: str,
                    panel_var: str | None = None,
                    x_label: str | None = None,
                    df_plot: pd.DataFrame | None = None,
                    panel_levels: list | None = None,
                    output_filebase: str | None = None,
                ) -> plt.Figure:
                    
                    fig = cfg.plot_fit_wrapper(
                        df=df_band if df_plot is None else df_plot,
                        df_type='norm',
                        save_dir=save_dir_model,
                        param_type=param_type,
                        bd=bd,
                        x_var=x_var,
                        panel_var=panel_var,
                        x_label = x_label,
                        filename=f"{x_var}-{panel_var}_{feature}",
                        panel_levels=panel_levels,
                        output_filebase=output_filebase,
                    )
                    
                    return fig

                def split_output_filebase(
                    split: dict,
                    x_var: str,
                    panel_var: str | None,
                ) -> str:
                    layout_filebase = cfg.build_layout_filebase(x_var, panel_var)
                    filename = f"{x_var}-{panel_var}_{feature}"
                    suffix = split["output_filebase"].removeprefix(layout_filebase)
                    return f"{filename}{suffix}"

                def plot_scalar_fit_with_region_splits(
                    x_var: str,
                    panel_var: str | None = None,
                    x_label: str | None = None,
                ) -> plt.Figure:
                    fig = plot_scalar_fit_wrapper(
                        x_var=x_var,
                        panel_var=panel_var,
                        x_label=x_label,
                    )
                    for split in cfg.iter_split_level_specs(
                        df_band,
                        x_var=x_var,
                        panel_var=panel_var,
                        split_vars=(cfg.REGION_VAR,),
                    ):
                        fig = plot_scalar_fit_wrapper(
                            x_var=x_var,
                            panel_var=panel_var,
                            x_label=x_label,
                            df_plot=split["df"],
                            panel_levels=split["panel_levels"],
                            output_filebase=split_output_filebase(
                                split,
                                x_var,
                                panel_var,
                            ),
                        )
                    return fig

                def plot_region_scalar_wrapper() -> plt.Figure:
                    filebase = f"Region_{feature}"
                    fig = cfg.plot_scalar_wrapper(
                        df=df_band,
                        df_type='norm',
                        save_dir=save_dir_model,
                        param_type=param_type,
                        bd=bd,
                        x_var='Region',
                        output_filebase=filebase,
                        model_filebase=filebase,
                    )
                    
                    return fig
                
                #%% 1.1. z_Feature_within-Region (β and p-values)
                x_var = 'z_Feature_within'
                panel_var = 'Region'
                x_label = f'Within-subject {feature_map[feature]} (z-scored)'
                fig = plot_scalar_fit_with_region_splits(
                    x_var=x_var,
                    panel_var=panel_var,
                    x_label=x_label
                    )

                #%% 2.1. z_Feature_between-Region (β and p-values)
                x_var = 'z_Feature_between'
                panel_var = 'Region'
                x_label = f'Between-subject {feature_map[feature]} (z-scored)'                
                fig = plot_scalar_fit_with_region_splits(
                    x_var=x_var,
                    panel_var=panel_var,
                    x_label=x_label
                    )

                #%% 3.1. Combined z_Feature-Region-Gradient
                fig = plot_combined_feature_fit(
                    df=df_band,
                    df_type='norm',
                    save_dir=save_dir_model,
                    param_type=param_type,
                    bd=bd,
                    feature=feature,
                )

                #%% 4.1. Region
                fig = plot_region_scalar_wrapper()
                
                # %%
                plt.close('all')
