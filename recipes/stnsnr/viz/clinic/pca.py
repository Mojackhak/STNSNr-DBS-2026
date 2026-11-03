#%%
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))

import src.utils.paths as paths
import src.utils.fileops as fileops
import src.viz.radar as radar
from src.stats.pca import pca
import src.viz.pca as pca_viz
import plugin.default.viz_defaults as cfg

import importlib
importlib.reload(pca)
importlib.reload(pca_viz)
importlib.reload(radar)

CONDITION_ORDER = (
    "Pre-op",
    "STN (immediate)",
    "STN+SNr (immediate)",
    "STN (3 m)",
    "STN+SNr (3 m)",
)
CONDITION_PALETTE = {
    "Pre-op": "#F2000E",
    "STN (immediate)": "#E87000",
    "STN+SNr (immediate)": "#884392",
    "STN (3 m)": "#0E6AAF",
    "STN+SNr (3 m)": "#0CA228",
}
CONDITION_LABEL_NORMALIZATION = {
    "STN (3m)": "STN (3 m)",
    "STN+SNr (3m)": "STN+SNr (3 m)",
}

def condition_colors_for(values: pd.Series) -> list[str]:
    observed = set(values.dropna())
    return [CONDITION_PALETTE[label] for label in CONDITION_ORDER if label in observed]

PCA_PLOT_CONFIG = {
    "boxsize": cfg.boxsize,
    "dpi": cfg.dpi,
    "font_family": "Arial",
    "axis_label_fontsize": cfg.font_size,
    "tick_label_fontsize": cfg.tick_label_size,
    "legend_fontsize": cfg.font_size,
    "legend_marker_size": 4,
    "line_width": cfg.line_width,
    "center_edge_width": 0.2,
    "x_label_offset_mm": cfg.x_label_offset_mm,
    "y_label_offset_mm": cfg.y_label_offset_mm,
    "transparent": cfg.transparent,
    "show": False,
}

LOADING_HEATMAP_CONFIG = {
    "dpi": cfg.dpi,
    "font_family": "Arial",
    "cellsize": (6.0, 4.0),
    "cmap": cfg.cm.vik,
    "vmin": -1.0,
    "vmax": 1.0,
    "label_top_height_mm": cfg.strip_top_height_mm,
    "label_top_pad_mm": cfg.strip_pad_mm,
    "label_top_bg_color": cfg.label_top_bg_color,
    "label_top_text_color": cfg.label_text_color,
    "label_top_fontsize": 7,
    "label_top_fontweight": "bold",
    "x_label_offset_mm": 0.0,
    "y_label_offset_mm": 18.0,
    "cbar_label_offset_mm": 8.0,
    "axis_label_fontsize": 7,
    "tick_label_fontsize": 6,
    "colorbar_label": "Loading factor",
    "colorbar_width_mm": 3.0,
    "colorbar_pad_mm": 2.0,
    "colorbar_length_mm": None,
    "transparent": True,
    "show": False,
}

LOADING_HEATMAP_LABELS = {
    "gait": "Gait",
    "balance": "Balance",
}

#%% 1) Project paths and cohort-level metadata

project = Path(os.environ["MYLFP_PROJECT_ROOT"])  # Folder path of project constructed in BIDS

summary_dir = project / 'summary'
clinic_dir = summary_dir / 'stats' / 'clinic'
pca_dir = clinic_dir / "pca"
feature = 'balance'
wk_dir = pca_dir / feature
df_paths = paths.get_files_list(wk_dir, filetypes='.pkl', contain='_wide')

#%%
pca_features_gait = {
    "Pace (m s)": 1,
    'Gait cycle (s)': -1,
    "Double supporting phase (%)": -1,
    "Stride length (cm)": 1,
    "LAS step length (cm)": 1,
    "MAS step length (cm)": 1,
    "LAS foot clearance (cm)": 1,
    "MAS foot clearance (cm)": 1,
    "LAS stride width (cm)": -1,
    "MAS stride width (cm)": -1,
    "LAS swing phase (%)": 1,
    "MAS swing phase (%)": 1,
}

pca_features_balance = {
    'Postural sway velocity during eyes-closed bipedal stance (cms)': -1,
    'Postural sway velocity during eyes-open bipedal stance (cms)': -1,
    'Romberg (%)': -1,
    'Support base area during eyes-closed bipedal stance (cm2)': -1,
    'Support base area during eyes-open bipedal stance (cm2)': -1,
    'Sway area during eyes-closed bipedal stance (cm2)': -1,
    'Sway area during eyes-open bipedal stance (cm2)': -1,
    'Sway path length per unit time during eyes-closed bipedal stance (cm)': -1,
    'Sway path length per unit time during eyes-open bipedal stance (cm)': -1,
    'Total sway path length during eyes-closed bipedal stance (cm)': -1,
    'Total sway path length during eyes-open bipedal stance (cm)': -1
    }

pca_features = {
    'gait': pca_features_gait,
    'balance': pca_features_balance,
    }


def save_loading_factors_heatmap(feature_name: str, label_top: str) -> None:
    feature_dir = pca_dir / feature_name
    feature_df_paths = paths.get_files_list(feature_dir, filetypes=".pkl", contain="_wide")
    if not feature_df_paths:
        raise FileNotFoundError(f"Missing wide PCA input in {feature_dir}")

    feature_df = pd.read_pickle(feature_df_paths[0])
    feature_columns = list(pca_features[feature_name].keys())
    missing_columns = [column for column in feature_columns if column not in feature_df.columns]
    if missing_columns:
        raise ValueError(f"{feature_name}: missing PCA feature columns: {missing_columns}")

    feature_df = feature_df[feature_columns].copy()
    feature_df.dropna(inplace=True)
    feature_fit_result = pca.perform_PCA(feature_df)
    fig = pca_viz.plot_loading_factors_heatmap(
        feature_fit_result["df_loadings"],
        feature_fit_result["explained_variance_ratio"],
        feature_fit_result["df_x_processed"],
        n_components=2,
        label_top=label_top,
        **LOADING_HEATMAP_CONFIG,
    )
    for suffix in ("pdf", "png"):
        save_path = feature_dir / f"loading_factors.{suffix}"
        cfg.save_fig(fig, save_path)
    plt.close(fig)

#%%
df_path = df_paths[0]
df = pd.read_pickle(df_path)
df = df[pca_features[feature].keys()].copy()
df.dropna(inplace=True)
fit_result = pca.perform_PCA(df)

#%% Improvement direction in top-k PC space (unit vector)
pc_improvement_unit_vector = pca.get_improvement_pc_unit_vector(
    fit_result=fit_result,
    feature_directions=pca_features[feature],
    n_components=2,
)

# %% 
loadings = fit_result['df_loadings']
top_features_df_filtered = pca.extract_top_features(
    loadings, n_components='all', top_n='all')

# %%
df_result = fit_result['df_scores'].reset_index()
df_dist = pca.get_signed_dist(
    df_result, match=["ID"], 
    vector=pc_improvement_unit_vector,
    baseline={"Phase": "Pre-op"},
    ncomponents=2)

for col in df_dist.columns:
    first = df_dist.at[0,col]
    if isinstance(first, tuple):
        if len(first) == 1:
            df_dist[col] = df_dist[col].str[0]
            if isinstance(first[0],str):
                df_dist[col].astype(str)

df_dist[["Protocol", "Phase"]] = (
    df_dist["sample_cmp"].str
    .extract(r"^\s*(.*?)\s*\((.*?)\)\s*$")
)
df_dist.rename(columns={"distance": "Value"}, inplace=True)
df_dist = df_dist[["ID", "Protocol", "Phase", "Value"]]
df_dist.loc[df_dist["Phase"] == "3m", "Phase"] = "3 m"
df_dist.loc[df_dist["Phase"] == "immediate", "Phase"] = "Immediate"
df_dist = df_dist.sort_values(
    ["Protocol", "Phase", "ID"],
    ascending=[True, False, True])
        
# %%

save_path = wk_dir / "dist.xlsx"
df_dist.to_excel(save_path, index=False)

# %%
df_result["Phase"] = df_result["Phase"].replace(CONDITION_LABEL_NORMALIZATION)
df_result["Phase"] = pd.Categorical(
    df_result["Phase"],
    categories=CONDITION_ORDER,
    ordered=True
)

save_path = wk_dir / "result.xlsx"
df_result.to_excel(save_path, index=False)

#%%
explained_variance = fit_result['explained_variance_ratio']
condition_pca_plot_config = PCA_PLOT_CONFIG.copy()
condition_pca_plot_config["color_palette"] = condition_colors_for(df_result["Phase"])

fig = pca_viz.visualize_PCA(
    df_result, explained_variance,
    var1="Phase", var2="Phase", error_bar='sem',
    size_sample=5, size_center=5, tight_lim=False, **condition_pca_plot_config)
save_path = wk_dir / "pca.pdf"
cfg.save_fig(fig, save_path)

#%%

fig = pca_viz.visualize_PCA(
    df_result, explained_variance,
    var1="Phase", var2="Phase", error_bar='sem', size_sample=0, 
    size_center=5, tight_lim=True,
    xlim=(-2, 2), ylim=(-0.8,0.8),
    scatter_x=pc_improvement_unit_vector['PC1'],
    scatter_y=pc_improvement_unit_vector['PC2'], **condition_pca_plot_config)
save_path = wk_dir / "pca_zoom.pdf"
cfg.save_fig(fig, save_path)

for loading_feature, label_top in LOADING_HEATMAP_LABELS.items():
    save_loading_factors_heatmap(loading_feature, label_top)

# %%
n = 9 # for radar
features = top_features_df_filtered.loc[:n, 'PC1'].tolist()

# %%
agg_path = wk_dir / f'{feature}_agg.pkl'
df_agg = pd.read_pickle(agg_path)
df_radar = df_agg[df_agg.Feature.isin(features)].copy()

df_radar["Protocol"] = pd.Categorical(
    df_radar["Protocol"],
    categories=["Pre-op", "STN", "STN+SNr",],
    ordered=True
)

df_radar["Phase"] = pd.Categorical(
    df_radar["Phase"],
    categories=["Pre-op", "immediate", "3m",],
    ordered=True
)

df_radar["Protocol_Phase"] = df_radar["Protocol_Phase"].replace(
    CONDITION_LABEL_NORMALIZATION
)
df_radar["Protocol_Phase"] = pd.Categorical(
    df_radar["Protocol_Phase"],
    categories=CONDITION_ORDER,
    ordered=True
)

df_radar = df_radar.reset_index(drop=True)

# %%

group_cols = ["Phase"]
for key, df_sub in df_radar.groupby(group_cols, observed=True):
    fig = radar.plot_radar(
        df_sub, col_cats="Feature", col_value="Value",
        col_group="Protocol", col_p="p", category_scale_to=(0.2, 1),
        radial_tick_format="percent", show_radial_tick_labels=False,
        line_width=1,
        color_palette=cfg.palette, title=key[0])
        
    save_path = wk_dir / f"radar_{key[0]}.pdf"
    cfg.save_fig(fig, save_path)
    
# %%

fig = radar.plot_radar(
    df_radar, col_cats="Feature", col_value="Value",
    col_group="Protocol_Phase", col_p=None, category_scale_to=(0.2, 1),
    radial_tick_format="numeric", show_radial_tick_labels=False,
    line_width=1,
    color_palette=condition_colors_for(df_radar["Protocol_Phase"]), title="All")
    
save_path = wk_dir / f"radar_all.pdf"
cfg.save_fig(fig, save_path)

# %%
