#%%
import os
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.multitest import multipletests

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
section_dir = stats_dir / r"clinic" / r"effect"
wk_dirs = [section_dir / r"3m", 
           section_dir / r"gait", 
           section_dir / r"balance", 
           ]

#%%
pair_selection = [
    'A - B', 'A - C', 'A - D', 'A - E',
    'B - C', 'D - E',
    'B - D', 'C - E',
]

condition_map = {
    "A": "Pre-op",
    "B": "STN (immediate)",
    "C": "STN+SNr (immediate)",
    "D": "STN (3 m)",
    "E": "STN+SNr (3 m)",
}
CONDITION_ORDER = tuple(condition_map.values())
CONDITION_PALETTE = {
    "Pre-op": "#F2000E",
    "STN (immediate)": "#E87000",
    "STN+SNr (immediate)": "#884392",
    "STN (3 m)": "#0E6AAF",
    "STN+SNr (3 m)": "#0CA228",
}
CONDITION_PLOT_CONFIG = {
    "jitter_palette": CONDITION_PALETTE,
    "fill_palette": CONDITION_PALETTE,
    "outline_palette": CONDITION_PALETTE,
}

def observed_condition_order(values: pd.Series) -> list[str]:
    observed = set(values.dropna())
    return [condition for condition in CONDITION_ORDER if condition in observed]

def map_condition_and_p_adjust(stem_path: Path, save_name: str):
    tukey_path = stem_path.parent / f"{stem_path.stem}_tukey.csv"
    df_tukey = pd.read_csv(tukey_path)
    df_tukey = df_tukey[df_tukey['contrast'].isin(pair_selection)]
    df_tukey.reset_index(drop=True, inplace=True)
    p_raw = df_tukey['p_raw'].values
    _, p_across, _, _ = multipletests(p_raw, alpha=0.05, method="fdr_bh")
    df_tukey['p_across'] = p_across
    # map condition names
    df_tukey['lhs'] = df_tukey['lhs'].map(condition_map)
    df_tukey['rhs'] = df_tukey['rhs'].map(condition_map)
    df_tukey['group1'] = df_tukey['group1'].map(condition_map)
    df_tukey['group2'] = df_tukey['group2'].map(condition_map)
    df_tukey['contrast'] = df_tukey['lhs'] + ' - ' + df_tukey['rhs']
    save_path = stem_path.parent / f"{save_name}_tukey.csv"
    df_tukey.to_csv(save_path, index=False)
    
    emm_path = stem_path.parent / f"{stem_path.stem}_emm.csv"
    df_emm = pd.read_csv(emm_path)
    df_emm.rename(columns={'Group': 'Condition'}, inplace=True)
    df_emm['Condition'] = df_emm['Condition'].map(condition_map)
    save_path = stem_path.parent / f"{save_name}_emm.csv"
    df_emm.to_csv(save_path, index=False)
    
#%% read data
for wk_dir in wk_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit('-',1)[0]
    print(f"Processing data: {wk_dir_name}")

    #%%
   
    df_path = paths.get_files_list(wk_dir, filetypes='.pkl', contain='scale_subject.pkl')[0]
    df = load_pkl(df_path)

    #%%
    bd_list = df['Feature'].unique().tolist()

    #%%
    for bd in bd_list:
        df_band = df[df['Feature'] == bd].copy()
        condition_order = observed_condition_order(df_band["Condition"])
        df_band["Condition"] = pd.Categorical(
            df_band["Condition"],
            categories=condition_order,
            ordered=True,
        )
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
            ) -> plt.Figure:
                
                fig = cfg.plot_scalar_wrapper(
                    df=df_band,
                    df_type='raw',
                    save_dir=save_dir_model,
                    param_type='',
                    bd='',
                    x_var=x_var,
                    panel_var=panel_var,
                    facet_var=facet_var,
                    x_label=' ',
                    y_label=bd,
                    cfg_update = {
                        "xtick_rotation": 45,
                        "y_label_offset_mm": 5,
                        **CONDITION_PLOT_CONFIG,
                        "x_levels": condition_order,
                        }
                    )
                
                return fig
            
            #%% 1.1. Condition 
            x_var = 'Condition'
            stem_path = save_dir_model / "Group"
            map_condition_and_p_adjust(stem_path, save_name="Condition")
            fig = plot_scalar_raw_wrapper(
                x_var=x_var,
                )
               
            # %%
            plt.close('all')
