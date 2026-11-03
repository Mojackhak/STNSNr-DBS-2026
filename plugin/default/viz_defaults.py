from __future__ import annotations

from itertools import product
import numpy as np
import pandas as pd
from pathlib import Path
import src.viz.visualdf as vdf
import matplotlib.pyplot as plt
import nibabel as nib
from cmcrameri import cm
from typing import Any, Final, Literal, List


#%% configuration
value_col = "Value"
palette = ['#F2000E', '#0E6AAF', '#0CA228', '#E87000', '#884392']
label_top_bg_color = "#D7E3E0"
label_right_bg_color = "#E3DCCF"
raw_mean_line_color = '#404040'
hline_color = '#404040'
label_text_color = "black"
label_fontweight = "bold"
panel_gap = (3.0, 3.0)
dpi = 600
line_width = 1
font_size = 7
tick_label_size = 6
grid = False
show_top_right_axes = False
transparent = True

ribbon_alpha = 0.30
legend_loc = 'outside_right'

band_shadows = {
    (1, 4): "#E5E5E540",    # delta
    (4, 8): "#CCCCCC40",    # theta
    (8, 13): "#B3B3B340",   # alpha
    (13, 20): "#9A9A9A40",  # beta_low
    (20, 35): "#81818140",  # beta_high
    (35, 80): "#68686840"  # gamma
}

NO_DELTA_FREQUENCY_LIMITS: Final[tuple[int, int]] = (4, 75)
NO_DELTA_SUFFIX: Final[str] = "noDelta"

boxsize_base = (6, 25)
boxsize = (30, 25)
strip_top_height_mm = 4.5   
strip_right_width_mm = 4.5  
strip_pad_mm = 2          
x_label_offset_mm = 5      
y_label_offset_mm = 5      
colorbar_width_mm = 3
colorbar_pad_mm = 2
cbar_label_offset_mm = 5

def save_fig(fig, save_path: str | Path):
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

def type_n_cfg(config_updated: dict, type_n: int) -> dict:
    
    if type_n <= 2: # only top strip
        config_updated.pop('strip_right_width_mm', None)
        config_updated.pop('label_right_bg_color', None)
    
    if type_n <= 1: # no strips and multiple panels
        config_updated.pop('strip_top_height_mm', None)
        config_updated.pop('strip_pad_mm', None)
        config_updated.pop('label_fontsize', None)
        config_updated.pop('label_top_bg_color', None)
        config_updated.pop('label_text_color', None)
        config_updated.pop('label_fontweight', None)
        config_updated.pop('panel_gap', None)
    
    return config_updated

#%% label dict
RAW_YLABEL_TPL: Final[dict[str, str]] = {
    "aperiod-mean": "{prefix}Aperiodic {key} power (dB)",
    "period-mean": "{prefix}Periodic {key} power (dB)",
    "raw-mean": "{prefix}Total {key} power (dB)",
    "params-mean": "{prefix}{key}",
    "burst-rate": r"{prefix}$\operatorname{{arcsinh}}$ {key} burst rate (Hz)",
    # "burst-duration": r"{prefix}$\log_{{10}}$ {key} burst duration (s)",
    "burst-duration": r"{prefix}$\operatorname{{ln}}$ {key} burst duration (s)",
    "burst-occupation": "{prefix}{key} burst occupation",
    # "burst-mean": r"{prefix}$\log_{{10}}$ {key} burst amplitude (V)",
    "burst-mean": r"{prefix}$\operatorname{{ln}}$ {key} burst amplitude (V)",
    "coh-mean": r"{prefix}Fisher z of $\sqrt{{{key}\,\mathrm{{coherence}}}}$",
    "ciplv-mean": r"{prefix}$\operatorname{{logit}}({key}\,\mathrm{{ciPLV}})$",
    "wpli-mean": r"{prefix}$\operatorname{{logit}}({key}\,\mathrm{{wPLI}})$",
    "delta_net_gc-mean": "{prefix}{key} TRGC",
    "psi-mean": "{prefix}{key} PSI",

    "aperiod-transformed": "{prefix}Aperiodic {key} power (dB)",
    "period-transformed": "{prefix}Periodic {key} power (dB)",
    "raw-transformed": "{prefix}Total {key} power (dB)",
    "coh-transformed": r"{prefix}Fisher z of $\sqrt{{{key}\,\mathrm{{coherence}}}}$",
    "ciplv-transformed": r"{prefix}$\operatorname{{logit}}({key}\,\mathrm{{ciPLV}})$",
    "wpli-transformed": r"{prefix}$\operatorname{{logit}}({key}\,\mathrm{{wPLI}})$",
    "delta_net_gc-transformed": "{prefix}{key} TRGC",
  
}

PAIN_EVOKED_YLABEL_TPL: Final[dict[str, str]] = {
    "mean-onset_time": "{metric_label}: onset latency \n relative to ONS/OFS (s)",
    "mean-peak_time": "{metric_label}: peak latency \n relative to ONS/OFS (s)",
    "mean-offset_time": "{metric_label}: offset latency \n relative to ONS/OFS (s)",
    "mean-peak_value": "{metric_label}: peak value \n in response to ONS/OFS (a.u.)",
    "mean-auc": "{metric_label}: response AUC \n in response to ONS/OFS (a.u.)",
    "mean-auc_rel_left": "{metric_label}: BL-corrected response AUC \n in response to ONS/OFS (a.u.)",
    "mean-duration": "{metric_label}: response \n duration following ONS/OFS (s)",
    "mean-mean_slope_rise": "{metric_label}: rise \n slope after ONS/OFS (a.u.)",
    "mean-mean_slope_fall": "{metric_label}: fall \n slope after ONS/OFS (a.u.)",
}
#%% Band dict
BAND_TPL = {'Delta': 'δ', 'Theta': 'θ', 'Alpha': 'α', 'Beta': 'β',
            'Beta_low': 'low-β', 'Beta_high': 'high-β', 'Gamma': 'γ',
            'Exponent': 'Exponent', 'Offset': 'Offset'}

REST_BURST_BAND_ORDER: Final[list[str]] = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta_low",
    "Beta_high",
    "Gamma",
]

#%% Prefix dict
PREFIX_TPL = {'raw': '', 'norm': 'Δ', 'percent': 'Δ%'}

PAIN_EVOKED_PREFIX_TPL = {'coh': 'Coherence', 
                          'period': 'Periodic power',
                          'ciplv': 'ciPLV',
                          'wpli': 'wPLI',
                          'raw': 'Total power',
                          'params': '', }

#%% connection region helpers
REGION_VAR: Final[str] = "Region"
PHASE_VAR: Final[str] = "Phase"
POSE_VAR: Final[str] = "Pose"
TASK_VAR: Final[str] = "Task"
UNDIRECTED_CONNECTION_REGION: Final[str] = "SNr-STN"
DIRECTED_CONNECTION_REGION: Final[str] = "SNr→STN"
UNDIRECTED_CONNECTION_PARAMS: Final[frozenset[str]] = frozenset({"coh", "ciplv", "wpli"})
DIRECTED_CONNECTION_PARAMS: Final[frozenset[str]] = frozenset({"psi", "delta_net_gc"})


def connection_region_label(param_type: str) -> str | None:
    param = param_type.split("-", 1)[0]
    if param in UNDIRECTED_CONNECTION_PARAMS:
        return UNDIRECTED_CONNECTION_REGION
    if param in DIRECTED_CONNECTION_PARAMS:
        return DIRECTED_CONNECTION_REGION
    return None


def with_connection_region_label(
    df: pd.DataFrame,
    param_type: str,
    *,
    region_var: str = REGION_VAR,
) -> pd.DataFrame:
    out = df.copy()
    label = connection_region_label(param_type)
    if label is None or region_var not in out.columns:
        return out

    is_categorical = isinstance(out[region_var].dtype, pd.CategoricalDtype)
    ordered = out[region_var].dtype.ordered if is_categorical else False
    out[region_var] = pd.Categorical(
        [label] * len(out),
        categories=[label],
        ordered=ordered,
    )
    return out


def present_levels(df: pd.DataFrame, var: str) -> list:
    values = df[var]
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [
            level
            for level in values.cat.categories.tolist()
            if (values == level).any()
        ]
    return values.dropna().drop_duplicates().tolist()


def filter_to_level(df: pd.DataFrame, var: str, level, source_name: str) -> pd.DataFrame:
    if var not in df.columns:
        raise KeyError(f"{source_name} is missing required column: {var}")
    return df[df[var].astype(str) == str(level)].reset_index(drop=True)


def filename_level_label(value) -> str:
    label = str(value).replace("\u2192", "-to-")
    out = []
    for char in label:
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        elif char.isspace():
            out.append("_")
        else:
            out.append("-")
    return "".join(out).strip("-_") or "level"


def append_filebase_suffix(filebase: str, suffix: str | None) -> str:
    if suffix is None:
        return filebase
    suffix_label = filename_level_label(suffix)
    return f"{filebase}-{suffix_label}" if suffix_label else filebase


def no_delta_spectral_config() -> dict[str, list[int]]:
    return {"x_limits": list(NO_DELTA_FREQUENCY_LIMITS)}


def no_delta_raw_config() -> dict[str, list[int]]:
    return {"y_limits": list(NO_DELTA_FREQUENCY_LIMITS)}


def iter_split_level_specs(
    df: pd.DataFrame,
    x_var: str,
    panel_var: str | None = None,
    facet_var: str | None = None,
    *,
    split_vars: tuple[str, ...] = (REGION_VAR, TASK_VAR),
) -> list[dict[str, Any]]:
    layout_vars = {panel_var, facet_var}
    active_split_vars = [
        var
        for var in split_vars
        if var != x_var and var in layout_vars and var in df.columns
    ]
    if not active_split_vars:
        return []

    level_lists = [present_levels(df, var) for var in active_split_vars]
    specs = []
    base_filebase = build_layout_filebase(x_var, panel_var, facet_var)

    for level_combo in product(*level_lists):
        df_split = df
        levels_by_var = dict(zip(active_split_vars, level_combo))
        for var, level in levels_by_var.items():
            df_split = filter_to_level(df_split, var, level, "split data")

        if df_split.empty:
            continue

        suffix = "-".join(
            filename_level_label(levels_by_var[var])
            for var in active_split_vars
        )
        specs.append({
            "df": df_split,
            "panel_levels": (
                [levels_by_var[panel_var]]
                if panel_var in levels_by_var
                else None
            ),
            "facet_levels": (
                [levels_by_var[facet_var]]
                if facet_var in levels_by_var
                else None
            ),
            "output_filebase": f"{base_filebase}-{suffix}",
        })

    return specs


def append_region_layout(
    x_var: str | None,
    panel_var: str | None = None,
    facet_var: str | None = None,
    *,
    region_var: str = REGION_VAR,
) -> tuple[str | None, str | None, str | None]:
    if region_var in {x_var, panel_var, facet_var}:
        return panel_var, facet_var, None
    if panel_var is None:
        return region_var, facet_var, region_var
    if facet_var is None:
        return panel_var, region_var, region_var
    return panel_var, facet_var, None


def build_layout_filebase(
    x_var: str | None = None,
    panel_var: str | None = None,
    facet_var: str | None = None,
) -> str:
    parts = [var for var in (x_var, panel_var, facet_var) if var is not None]
    return "-".join(parts) if parts else "All"


def _fallback_filebase_without_var(
    x_var: str | None,
    panel_var: str | None,
    facet_var: str | None,
    fallback_var: str | None,
) -> str | None:
    if fallback_var is None:
        return None
    parts = [var for var in (x_var, panel_var, facet_var) if var is not None and var != fallback_var]
    return "-".join(parts) if parts else "All"


def _read_model_csv(
    save_dir: Path,
    filebase: str,
    suffix: str,
    *,
    param_type: str,
    model_fallback_var: str | None = None,
    model_fallback_filebase: str | None = None,
) -> pd.DataFrame:
    path = save_dir / f"{filebase}_{suffix}.csv"
    if path.exists():
        table = pd.read_csv(path)
        fallback_value = (
            connection_region_label(param_type)
            if model_fallback_var == REGION_VAR
            else None
        )
        if fallback_value is not None and model_fallback_var not in table.columns:
            table[model_fallback_var] = fallback_value
        return table

    fallback_value = (
        connection_region_label(param_type)
        if model_fallback_var == REGION_VAR
        else None
    )
    if model_fallback_var is None or model_fallback_filebase is None or fallback_value is None:
        return pd.read_csv(path)

    fallback_path = save_dir / f"{model_fallback_filebase}_{suffix}.csv"
    if not fallback_path.exists():
        return pd.read_csv(path)

    table = pd.read_csv(fallback_path)
    if model_fallback_var not in table.columns:
        table[model_fallback_var] = fallback_value
    return table

#%% label format function
def _join_label_parts(*parts: str) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def format_label(param_type: str, *, prefix: str = "", key: str = "") -> str:
    try:
        tpl = RAW_YLABEL_TPL[param_type]
    except KeyError as e:
        raise KeyError(f"Unknown param_type: {param_type!r}") from e
    return tpl.format(prefix=prefix, key=key)


def format_pain_evoked_label(param_type: str, *, prefix: str = "", key: str = "") -> str:
    try:
        tpl = PAIN_EVOKED_YLABEL_TPL[param_type]
    except KeyError as e:
        raise KeyError(f"Unknown param_type: {param_type!r}") from e
    metric_label = _join_label_parts(key, prefix)
    return tpl.format(prefix=prefix, key=key, metric_label=metric_label)

#%%  scalar plot config

scalar_cfg = {
    # --- data ---
    
    # --- roles ---
    'value_col': value_col,  

    # --- palettes ---
    'jitter_palette': palette,
    'fill_palette': palette,
    'outline_palette': palette,

    # --- visual style ---
    'jitter_width': 0.20,
    'jitter_alpha': 1,
    'jitter_size': 2,
    'grid': grid,
    'dpi': dpi,
    'seed': 42,
    'show_top_right_axes': show_top_right_axes,
    
    # --- boxplot controls ---
    'show_box': True,
    'box_width': 0.55,
    'fill_alpha': 0.30,
    'whiskers': 'tukey',
    
    # global fallback outline styles
    'box_edge_width': line_width,
    'median_linewidth': line_width,
    'whisker_linewidth': line_width,
    'cap_linewidth': line_width,

    # outliers
    'outlier_marker': "o",
    'outlier_markersize': 0,

    # EMM line
    'emm_line_width': line_width,
    'emm_line_color': "black",
    'emm_line_style': "-",

    # Raw mean line
    'raw_mean_line_width': line_width,
    'raw_mean_line_color': raw_mean_line_color,
    'raw_mean_line_style': "--",

    # Error bar styling
    'error_bar_linewidth': line_width,
    'error_bar_cap': 2.0,
    'error_bar_color': "black",
    
    # --- vertical / horizontal references ---
    'horizontal_lines': [0,], 
    'hline_color': 'black',   
    'hline_style': ':',
    'hline_width': line_width,
    'hline_alpha': 0.3,

    # --- strips (top per column; right per row) — sizes RELATIVE TO PANEL ---
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    
    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,      
    
    # --- text sizes ---
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # --- BOXSIZE-BASED GEOMETRY (deterministic) ---
    'include_global_label_margins': True,

    # absolute gaps between panels (inner-box to inner-box)
    'panel_gap': panel_gap,

    # --- Tukey bracket controls ---
    'show_brackets': True,
    'hide_ns': True,
    'y_start': 0.50,
    'y_end': 0.90,
    'y_step': 0.08,
    'bracket_height_frac': 0.0,
    'bracket_color': "black",
    'bracket_linewidth': line_width,
    'bracket_text_size': font_size,

    # --- transparency ---
    'transparent': transparent,
}


#%% wrapper functions

def update_scalar_cfg(x_var, n, type_n):
    
    config_updated = scalar_cfg.copy()
    boxsize = (boxsize_base[0]*n, boxsize_base[1])
    config_updated['boxsize'] = boxsize
        
    config_updated['jitter_var'] = x_var
    config_updated['fill_var'] = x_var
    config_updated['outline_var'] = x_var
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated
    
def plot_scalar_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    bd: str,
    x_var: str,
    panel_var: str | None = None,
    facet_var: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    cfg_update: dict | None = None,
    fig_format: str = '.pdf',
    model_fallback_var: str | None = None,
    append_region: bool = False,
    panel_levels: list | None = None,
    facet_levels: list | None = None,
    output_filebase: str | None = None,
    model_filebase: str | None = None,
) -> plt.Figure:
    
    if append_region:
        df = with_connection_region_label(df, param_type)
        panel_var, facet_var, added_region_var = append_region_layout(
            x_var,
            panel_var,
            facet_var,
        )
        if model_fallback_var is None:
            model_fallback_var = added_region_var

    n = len(df[x_var].unique())
    type_n = 1

    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{x_var}-{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{x_var}-{panel_var}'
        else:
            filebase = f'{x_var}'
    
    save_dir = Path(save_dir)
    read_filebase = model_filebase or filebase
    model_fallback_filebase = _fallback_filebase_without_var(
        x_var,
        panel_var,
        facet_var,
        model_fallback_var,
    )
    emm = _read_model_csv(
        save_dir,
        read_filebase,
        "emm",
        param_type=param_type,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )
    tuk = _read_model_csv(
        save_dir,
        read_filebase,
        "tukey",
        param_type=param_type,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )   
    
    config_updated = update_scalar_cfg(x_var, n, type_n)
    if x_label is not None:
        config_updated['x_label'] = x_label
    else:
        config_updated['x_label'] = f'{x_var}'
    
    if y_label is not None:
        config_updated['y_label'] = y_label
    else:
        key = BAND_TPL[bd]
        prefix = PREFIX_TPL[df_type]
        y_label = format_label(param_type, prefix=prefix, key=key)
        config_updated['y_label'] = y_label
    
    
    if cfg_update is not None:
        config_updated.update(cfg_update)
    
    if type_n == 1:
        fig = vdf.plot_single_effect_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            panel_var=panel_var,
            panel_levels=panel_levels,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            **config_updated
        )           
    save_filebase = output_filebase or filebase
    save_path = save_dir / f'{save_filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig

def plot_scalar_pain_evoked_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    wk_dir_name: str,
    bd: str,
    x_var: str,
    panel_var: str | None = None,
    facet_var: str | None = None,
    fig_format: str = '.pdf',
    model_fallback_var: str | None = None,
    append_region: bool = False,
    panel_levels: list | None = None,
    facet_levels: list | None = None,
    output_filebase: str | None = None,
) -> plt.Figure:
    
    n = len(df[x_var].unique())
    type_n = 1
    param, param_type = wk_dir_name.split('-',1)
    if append_region:
        df = with_connection_region_label(df, wk_dir_name)
        panel_var, facet_var, added_region_var = append_region_layout(
            x_var,
            panel_var,
            facet_var,
        )
        if model_fallback_var is None:
            model_fallback_var = added_region_var
    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{x_var}-{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{x_var}-{panel_var}'
        else:
            filebase = f'{x_var}'
    
    save_dir = Path(save_dir)
    model_fallback_filebase = _fallback_filebase_without_var(
        x_var,
        panel_var,
        facet_var,
        model_fallback_var,
    )
    emm = _read_model_csv(
        save_dir,
        filebase,
        "emm",
        param_type=wk_dir_name,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )
    tuk = _read_model_csv(
        save_dir,
        filebase,
        "tukey",
        param_type=wk_dir_name,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )   
    
    config_updated = update_scalar_cfg(x_var, n, type_n)
    key = BAND_TPL[bd]
    prefix = PAIN_EVOKED_PREFIX_TPL[param]
    y_label = format_pain_evoked_label(param_type, prefix=prefix, key=key)
    config_updated['y_label'] = y_label
    config_updated['x_label'] = f'{x_var}'   
    
    
    if type_n == 1:
        fig = vdf.plot_single_effect_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            panel_var=panel_var,
            panel_levels=panel_levels,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_scalar(
            df=df,
            emm=emm,
            tuk=tuk,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            **config_updated
        )           
    save_filebase = output_filebase or filebase
    save_path = save_dir / f'{save_filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig

#%%  series plot config

series_cfg = {
    
    'value_col': value_col,  
            
    # figure / appearance
    'line_palette': palette,
    'line_width': line_width,
    'ribbon_alpha': ribbon_alpha,
    'grid': grid,
    'dpi': dpi,

    # horizontal references (apply to every panel)
    'vline_color': "#000000",   
    'hline_color': hline_color,
    'hline_style': "--",
    'hline_width': line_width,
    'hline_alpha': 0.6,
    
    # --- strips (top per column; right per row) — sizes RELATIVE TO PANEL ---
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,    
    
    # Legend controls
    'legend_loc': legend_loc,
    'legend_fontsize': font_size,

    # text sizes
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # legend
    'legend_fontsize': font_size,

    # absolute gaps between panels (inner-box to inner-box)
    'boxsize': boxsize,
    'panel_gap': panel_gap,    # (gap_x, gap_y)

    # transparency
    'transparent': transparent,
}

series_spectral_cfg = {
    'x_log': True,
    'x_limits': [1, 75],
    'x_label': 'Frequency (Hz)',
    'vertical_shadows': band_shadows,  
    **series_cfg,
}

series_trace_cfg = {
    'x_log': False,
    **series_cfg,
}

series_gait_trace_cfg = {
    'x_log': False,
    'x_limits': [0, 100],
    'x_label': 'Gait cycle (%)',
    'vertical_shadows': {
        (0, 5): "#B3B3B340",    
        (13, 23): "#B3B3B340",  
        (45, 55): "#B3B3B340",    
        (63, 73): "#B3B3B340",    
        (95, 100): "#B3B3B340",    
    },
    'vertical_lines': [18, 50, 68],
    **series_cfg,
}

series_pain_trace_cfg = {
    'x_log': False,
    'x_limits': [0, 20],
    'x_label': 'Time (s)',
    'vertical_shadows': {
        (0, 2.5): "#B3B3B340",    # BL
        (3.5, 5.5): "#B3B3B340",  # ONS
        (6.5, 12.5): "#B3B3B340",     # SUS
        (13.5, 15.5): "#B3B3B340",    # OFS
        (16.5, 20): "#B3B3B340",    # PST
    },
    'vertical_lines': [4, 14], 
    **series_cfg,
}

series_fog_trace_cfg = {
    'x_log': False,
    'x_limits': [0, 3],
    'x_label': 'Time (s)',
    'vertical_shadows': {
        (0, 1.0): "#E5E5E540",    
        (1.0, 2.0): "#CCCCCC40",  
        (2.0, 3.0): "#B3B3B340",    
        # (4.5, 5.5): "#B3B3B340",    
        # (6, 7): "#B3B3B340",    
    },
    # 'vertical_lines': [1, 2,],    
    **series_cfg,
}

series_section_trace_cfg = {
    'gait': series_gait_trace_cfg,
    'pain': series_pain_trace_cfg,
    'fog_episode': series_fog_trace_cfg,
    'default': series_trace_cfg,
}
#%% smooth trace function

def smooth_trace(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    def apply_lowess(series, frac=0.05):
        x = np.arange(len(series))
        y = series.values
        smoothed = lowess(y, x, frac=frac, return_sorted=False)
        return pd.Series(smoothed, index=series.index)
    
    frac = 0.05
    df['Value'] = df["Value"].map(lambda s: apply_lowess(s, frac))
    return df

#%% update config functions

def update_series_spectral_cfg(type_n):
    
    config_updated = series_spectral_cfg.copy()
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated

def update_series_trace_cfg(type_n, section):
    
    config_updated = series_section_trace_cfg.get(
        section, series_section_trace_cfg['default']).copy()
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated

DERIVE_TYPE = Literal["spectral", "trace"]

def plot_series_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    x_var: str,
    bd: str | None = None,
    panel_var: str | None = None,
    facet_var: str | None = None,
    derive_type: DERIVE_TYPE = 'spectral',
    section: str | None = None,
    fig_format: str = '.pdf',
    append_region: bool = False,
    panel_levels: list | None = None,
    facet_levels: list | None = None,
    output_filebase: str | None = None,
    output_suffix: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> plt.Figure:
    
    if append_region:
        df = with_connection_region_label(df, param_type)
        panel_var, facet_var, _ = append_region_layout(
            x_var,
            panel_var,
            facet_var,
        )

    type_n = 1
    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{x_var}-{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{x_var}-{panel_var}'
        else:
            filebase = f'{x_var}'
    
    save_dir = Path(save_dir)
    
    if derive_type == 'spectral':
        config_updated = update_series_spectral_cfg(type_n)
    else:  # derive_type == 'trace'
        config_updated = update_series_trace_cfg(type_n, section)
        
    prefix = PREFIX_TPL[df_type]
    
    if bd is None:
        key = ''
    else:
        key = BAND_TPL[bd]
        
    y_label = format_label(param_type, prefix=prefix, key=key)
    config_updated['y_label'] = y_label
    
    if df_type == 'norm':
        config_updated['horizontal_lines'] = [0,]

    if config_overrides:
        config_updated.update(config_overrides)
            
    if type_n == 1:
        fig = vdf.plot_single_effect_series(
            df=df,
            x_var=x_var,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_series(
            df=df,
            x_var=x_var,
            panel_var=panel_var,
            panel_levels=panel_levels,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_series(
            df=df,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            **config_updated
        )           
    
    save_filebase = append_filebase_suffix(output_filebase or filebase, output_suffix)
    save_path = save_dir / f'{save_filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig

#%% categorical summary line plot config

categorical_summary_line_cfg = {
    'value_col': value_col,

    # figure / appearance
    'line_palette': palette,
    'line_width': line_width,
    'line_alpha': 0.95,
    'marker': 'o',
    'marker_size': 2.5,
    'grid': grid,
    'dpi': dpi,
    'show_top_right_axes': show_top_right_axes,

    # Error bar styling
    'error_bar_linewidth': line_width,
    'error_bar_cap': 2.0,
    'error_bar_color': None,

    # reference lines
    'horizontal_lines': None,
    'hline_color': hline_color,
    'hline_style': '--',
    'hline_width': line_width,
    'hline_alpha': 0.6,

    # strips
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    'strip_top_height_mm': strip_top_height_mm,
    'strip_right_width_mm': strip_right_width_mm,
    'strip_pad_mm': strip_pad_mm,
    'x_label_offset_mm': x_label_offset_mm,
    'y_label_offset_mm': y_label_offset_mm,

    # text sizes
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,
    'xtick_rotation': 45.0,

    # legend
    'legend_loc': legend_loc,
    'legend_fontsize': font_size,

    # geometry
    'boxsize': boxsize,
    'panel_gap': panel_gap,
    'include_global_label_margins': True,

    # transparency
    'transparent': transparent,
}


def update_categorical_summary_line_cfg(_n: int, type_n: int) -> dict:
    config_updated = categorical_summary_line_cfg.copy()
    config_updated['boxsize'] = boxsize
    config_updated = type_n_cfg(config_updated, type_n)
    return config_updated


def _summary_line_filebase(
    x_var: str,
    line_var: str | None = None,
    panel_var: str | None = None,
    facet_var: str | None = None,
) -> str:
    parts = [var for var in (x_var, line_var, panel_var, facet_var) if var is not None]
    return "-".join(parts) if parts else "All"


def _format_summary_line_y_label(param_type: str, df_type: str) -> str:
    prefix = PREFIX_TPL[df_type]
    label = format_label(param_type, prefix=prefix, key="")
    return " ".join(label.split())


def plot_categorical_summary_line_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    x_var: str,
    line_var: str | None = None,
    panel_var: str | None = None,
    facet_var: str | None = None,
    x_levels: list | None = None,
    line_levels: list | None = None,
    panel_levels: list | None = None,
    facet_levels: list | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    cfg_update: dict | None = None,
    fig_format: str = '.pdf',
    filebase: str | None = None,
) -> plt.Figure:
    if facet_var is not None and panel_var is None:
        raise ValueError("panel_var must be provided if facet_var is provided.")

    type_n = 1
    if panel_var is not None:
        type_n += 1
    if facet_var is not None:
        type_n += 1

    if x_levels is None:
        n = len(pd.unique(df[x_var].dropna()))
    else:
        n = len(x_levels)

    config_updated = update_categorical_summary_line_cfg(n, type_n)
    config_updated['x_label'] = x_label or x_var
    config_updated['y_label'] = y_label or _format_summary_line_y_label(param_type, df_type)
    if x_var == 'Band':
        config_updated['x_tick_labels'] = BAND_TPL

    if cfg_update is not None:
        config_updated.update(cfg_update)

    fig = vdf.plot_categorical_summary_line(
        df=df,
        x_var=x_var,
        line_var=line_var,
        panel_var=panel_var,
        facet_var=facet_var,
        x_levels=x_levels,
        line_levels=line_levels,
        panel_levels=panel_levels,
        facet_levels=facet_levels,
        **config_updated,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filebase = filebase or _summary_line_filebase(x_var, line_var, panel_var, facet_var)
    save_path = save_dir / f'{filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)

    return fig

#%% correlaion plot config

fit_cfg = {
    'value_col': value_col,  
    # --- labels / scales (figure-level when single_*_label=True) ---
    'color_var': 'ID',    # jitter color

    # --- visual style ---
    'palette': "viridis",
    'jitter_alpha': 1.0,
    'jitter_size': 2,                        # marker size (points)
    'curve_line_width': line_width,
    'ribbon_alpha': ribbon_alpha,                      # solid black emmean line
    'grid': False,
    'dpi': dpi,

    # --- strips (top per column; right per row) — sizes are RELATIVE TO PANEL ---
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,

    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,     
    
    # --- text sizes ---
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # absolute gaps between panels (inner-box to inner-box)
    'boxsize': boxsize,
    'panel_gap': panel_gap,    # (gap_x, gap_y)

    # --- slope label (β and p / q) ---
    'slope_text_fmt': "β = {beta:.3f}\nq = {q:.3f} ({stars})",
    'slope_text_loc': 'lower left',
    'slope_text_box_alpha': 0,

    # # --- axis padding (adds blank edge beyond data) ---
    # 'x_pad_frac': 3,
    # 'y_pad_frac': 2.5,

    # --- transparency ---
    'transparent': transparent,
}

#%% wrapper functions

def r_scale_like(s):
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if sd is None or np.isnan(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd    

def zscore_by_group(df: pd.DataFrame, group: str | List) -> pd.DataFrame:
    cols = ["MNI_x_flip", "MNI_y_flip", "MNI_z_flip"]
    out_cols = ["zMX", "zMY", "zMZ"]    
    df[out_cols] = df.groupby(group, observed=True)[cols].transform(r_scale_like)
    return df

def update_fit_cfg(type_n):
    
    config_updated = fit_cfg.copy()    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated
    
def plot_fit_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    bd: str,
    x_var: str,
    panel_var: str | None = None,
    facet_var: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    filename: str | None = None,
    fig_format: str = '.pdf',
    model_fallback_var: str | None = None,
    model_fallback_filename: str | None = None,
    append_region: bool = False,
    panel_levels: list | None = None,
    facet_levels: list | None = None,
    output_filebase: str | None = None,
    model_filebase: str | None = None,
) -> plt.Figure:
    
    if append_region:
        df = with_connection_region_label(df, param_type)
        original_filename = filename
        panel_var, facet_var, added_region_var = append_region_layout(
            x_var,
            panel_var,
            facet_var,
        )
        if model_fallback_var is None:
            model_fallback_var = added_region_var
        if added_region_var is not None and original_filename is not None and model_fallback_filename is None:
            model_fallback_filename = original_filename
            if original_filename.startswith(x_var):
                filename = (
                    build_layout_filebase(x_var, panel_var, facet_var)
                    + original_filename[len(x_var):]
                )

    type_n = 1

    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{x_var}-{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{x_var}-{panel_var}'
        else:
            filebase = f'{x_var}'
            
    if filename is None:
        filename = filebase
    
    save_dir = Path(save_dir)
    read_filebase = model_filebase or filename
    model_fallback_filebase = (
        model_fallback_filename
        or _fallback_filebase_without_var(
            x_var,
            panel_var,
            facet_var,
            model_fallback_var,
        )
    )
    curve = _read_model_csv(
        save_dir,
        read_filebase,
        "curve",
        param_type=param_type,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )
    slope = _read_model_csv(
        save_dir,
        read_filebase,
        "slope",
        param_type=param_type,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )   
    
    config_updated = update_fit_cfg(type_n)
    key = BAND_TPL[bd]
    prefix = PREFIX_TPL[df_type]
    if y_label is None:
        y_label = format_label(param_type, prefix=prefix, key=key)
    config_updated['y_label'] = y_label
    axis = x_var.lower()[-1]
    if x_label is None:
        config_updated['x_label'] = f"MNI-{axis} (z-scored)"
    else:
        config_updated['x_label'] = x_label
    
    if type_n == 1:
        fig = vdf.plot_single_effect_fit(
            df=df,
            curve=curve,
            slope=slope,
            x_var=x_var,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_fit(
            df=df,
            curve=curve,
            slope=slope,
            x_var=x_var,
            panel_var=panel_var,
            panel_levels=panel_levels,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_fit(
            df=df,
            curve=curve,
            slope=slope,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            **config_updated
        )           
    save_filebase = output_filebase or filename
    save_path = save_dir / f'{save_filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig


def plot_fit_pain_evoked_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    wk_dir_name: str,
    bd: str,
    x_var: str,
    panel_var: str | None = None,
    facet_var: str | None = None,
    x_label: str | None = None,
    fig_format: str = '.pdf',
    model_fallback_var: str | None = None,
    append_region: bool = False,
    panel_levels: list | None = None,
    facet_levels: list | None = None,
    output_filebase: str | None = None,
) -> plt.Figure:
    
    n = len(df[x_var].unique())
    type_n = 1
    
    param, param_type = wk_dir_name.split('-',1)
    if append_region:
        df = with_connection_region_label(df, wk_dir_name)
        panel_var, facet_var, added_region_var = append_region_layout(
            x_var,
            panel_var,
            facet_var,
        )
        if model_fallback_var is None:
            model_fallback_var = added_region_var

    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{x_var}-{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{x_var}-{panel_var}'
        else:
            filebase = f'{x_var}'
    
    save_dir = Path(save_dir)
    model_fallback_filebase = _fallback_filebase_without_var(
        x_var,
        panel_var,
        facet_var,
        model_fallback_var,
    )
    curve = _read_model_csv(
        save_dir,
        filebase,
        "curve",
        param_type=wk_dir_name,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )
    slope = _read_model_csv(
        save_dir,
        filebase,
        "slope",
        param_type=wk_dir_name,
        model_fallback_var=model_fallback_var,
        model_fallback_filebase=model_fallback_filebase,
    )   
    
    config_updated = update_fit_cfg(type_n)
    key = BAND_TPL[bd]
    prefix = PAIN_EVOKED_PREFIX_TPL[param]
    y_label = format_pain_evoked_label(param_type, prefix=prefix, key=key)
    config_updated['y_label'] = y_label
    axis = x_var.lower()[-1]
    if x_label is None:
        config_updated['x_label'] = f"MNI-{axis} (z-scored)"
    else:
        config_updated['x_label'] = x_label
    
    if type_n == 1:
        fig = vdf.plot_single_effect_fit(
            df=df,
            curve=curve,
            slope=slope,
            x_var=x_var,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_fit(
            df=df,
            curve=curve,
            slope=slope,
            x_var=x_var,
            panel_var=panel_var,
            panel_levels=panel_levels,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_fit(
            df=df,
            curve=curve,
            slope=slope,
            x_var=x_var,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            **config_updated
        )           
    save_filebase = output_filebase or filebase
    save_path = save_dir / f'{save_filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig

#%% raw plot config

raw_cfg = {
    
    'value_col': value_col,  
    'mode': 'mean',

    # axis labels (figure-level when single_*_label=True)
    'y_label': 'Frequency (Hz)',

    # axes limits/scales
    'y_limits': [1, 75],
    'x_log': False,
    'y_log': True,
    
    # figure / appearance
    'cmap': cm.vik,
    'vmode': 'sym',
    'grid': grid,
    'dpi': dpi,
    
    # vertical references (apply to every panel)
    'vline_color': "#000000",
    'vline_style': ":",
    'vline_width': line_width,
    'vline_alpha': 0.3,

    # horizontal references (apply to every panel)
    'horizontal_lines': [4, 8, 13, 20, 35],
    'hline_color': "#000000",
    'hline_style': ":",
    'hline_width': line_width,
    'hline_alpha': 0.3,
    
    # --- strips (top per column; right per row) — sizes RELATIVE TO PANEL ---
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,    
    'colorbar_width_mm': colorbar_width_mm,
    'colorbar_pad_mm': colorbar_pad_mm,
    'cbar_label_offset_mm': cbar_label_offset_mm,
    
    # text sizes
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # absolute gaps between panels (inner-box to inner-box)
    'boxsize': boxsize,
    'panel_gap': panel_gap,    # (gap_x, gap_y)

    # transparency
    'transparent': transparent,
}


raw_pain_cfg = {
    'x_limits': [0, 20],
    'x_label': 'Time (s)',
    'vertical_lines': [4, 14],
    **raw_cfg,
}

raw_gait_cfg = {
    'x_limits': [0, 100],
    'x_label': 'Gait cycle (%)',
    'vertical_lines': [18, 50, 68],
    **raw_cfg,
}

raw_fog_cfg = {
    'x_limits': [0, 3],
    'x_label': 'Time (s)',
    'vertical_lines': [1, 2],
    **raw_cfg,
}

raw_section_cfg = {
    'gait': raw_gait_cfg,
    'pain': raw_pain_cfg,
    'fog_episode': raw_fog_cfg,
    'default': raw_cfg,
}

def update_raw_cfg(type_n, section):
    
    config_updated = raw_section_cfg.get(
        section, raw_section_cfg['default']).copy()
    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated


def plot_raw_wrapper(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    panel_var: str | None = None,
    facet_var: str | None = None,
    section: str | None = None,
    fig_format: str = '.pdf',
    append_region: bool = False,
    panel_levels: list | None = None,
    facet_levels: list | None = None,
    output_filebase: str | None = None,
    output_suffix: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> plt.Figure:
    
    if append_region:
        df = with_connection_region_label(df, param_type)
        panel_var, facet_var, _ = append_region_layout(
            None,
            panel_var,
            facet_var,
        )

    type_n = 1
    if facet_var is not None:
        type_n += 1
        if panel_var is None:
            raise ValueError("panel_var must be provided if facet_var is provided.")
        else:
            type_n += 1
            filebase = f'{panel_var}-{facet_var}'
    else:
        if panel_var is not None:
            type_n += 1
            filebase = f'{panel_var}'
        else:
            filebase = 'All'
    
    save_dir = Path(save_dir)

    config_updated = update_raw_cfg(type_n, section)
            
    prefix = PREFIX_TPL[df_type]
        
    colorbar_label = format_label(param_type, prefix=prefix, key='')
    config_updated['colorbar_label'] = colorbar_label

    if config_overrides:
        config_updated.update(config_overrides)
    
            
    if type_n == 1:
        fig = vdf.plot_single_effect_df(
            df=df,
            **config_updated
        ) 
    elif type_n == 2:
        fig = vdf.plot_double_interaction_df(
            df=df,
            panel_var=panel_var,
            panel_levels=panel_levels,
            **config_updated
        )
    else:  # type_n == 3
        fig = vdf.plot_triple_interaction_df(
            df=df,
            panel_var=panel_var,
            facet_var=facet_var,
            panel_levels=panel_levels,
            facet_levels=facet_levels,
            **config_updated
        )           
    
    save_filebase = append_filebase_suffix(output_filebase or filebase, output_suffix)
    save_path = save_dir / f'{save_filebase}_{df_type}{fig_format}'
    save_fig(fig, save_path)
    
    return fig










# #%%####################################################################################
# ########################################################################################
# ################################# Nifty Plot ###########################################
# ########################################################################################
# ########################################################################################
#%% volume plot config

def load_atlas(project: str | Path) -> Any:
    atlas_dir = Path(project) / r'config' / r'atlas'
    SNr_rh_file = atlas_dir / r'SNr_rh_mask.nii.gz'
    STN_rh_file = atlas_dir / r'STN_rh_mask.nii.gz'
    bg_file = atlas_dir / r'7T_100um_Edlow_2019_cropped.nii.gz'

    SNr_rh = nib.load(SNr_rh_file)
    STN_rh = nib.load(STN_rh_file)
    bg = nib.load(bg_file)

    mask_dict = {
        "SNr": SNr_rh,
        "STN": STN_rh,
    }

    mask_plot_dict = {
        "SNr": ([SNr_rh], ['#FF80004D']),
        "STN": ([STN_rh], ['#00C8004D']),
    }
    
    return bg, mask_dict, mask_plot_dict

 
#%% fig config

nifti_cfg = {

    # Binary masks overlay (second layer)
    'mask_threshold': 0.05,
    'mask_default_alpha': 0.3,
    
    # Slicing positions (percent along heatmap bbox on each axis)
    'percent_list': [25, 50, 75],
    'facets': ["Ax", "Cor", "Sag"],

    # Domain & sampling
    'threshold': -np.inf,
    'res_mm': 0.1,

    # Axis labels / visibility
    'x_label': "Slice position (%)",
    'y_label': "Slice plane",
    'show_tick_labels': False,

    # Figure / appearance
    'title': None,
    # 'vmin_heat': vmin_heat,
    # 'vmax_heat': vmax_heat,
    'bg_percentile': (0, 100),
    'dpi': dpi,

    # Deterministic panel geometry (mm)
    'boxsize': boxsize,
    'panel_gap': panel_gap,
    'label_fontsize': font_size,
    'label_top_bg_color': label_top_bg_color,
    'label_right_bg_color': label_right_bg_color,
    'label_text_color': label_text_color,
    'label_fontweight': label_fontweight,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': font_size,
    
    'strip_top_height_mm': strip_top_height_mm,   
    'strip_right_width_mm': strip_right_width_mm,  
    'strip_pad_mm': strip_pad_mm,          
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm, 
    'colorbar_width_mm': 3,
    'colorbar_pad_mm': 2,
    'cbar_label_offset_mm': 9,
    
    'transparent': transparent,

    # --- NEW: figure-level (global) scale bar ---
    'draw_global_scale_bar': True,
    'global_scale_bar_length_mm': 2,
    'global_scale_bar_pos': (0.815, 0.05),  # figure fraction
    'global_scale_bar_color': 'black',
    'global_scale_bar_height_mm': 1.5,
    'global_scale_bar_label': f"2 mm",            # None -> "{actual_length} mm"
    'global_scale_bar_label_fontsize': tick_label_size,
    'global_scale_bar_label_color': 'black',
    'global_scale_bar_label_pad_mm': 1,
}


#%% voxelization config
voxelization_cfg = {
    'x_col': "MNI_x_flip",
    'y_col': "MNI_y_flip",
    'z_col': "MNI_z_flip",
    'value_col': "Value",
    'prob_threshold': 0.05,
    'inflate_mm': 2.4,
    'smooth_mm': 0.5,
    'smooth_threshold': 0.05,
    'use_dbscan': True,
    'dbscan_eps_mm': 1.0,
    'method': 'linear',
    'sigma_mm': 0.5,
    'norm': False,
}

#%% wrapper functions
type_specified_dict = {
    'raw': 
        {'cmap_heat': 'viridis',
         'vmode_heat': 'auto',},
    'norm': 
        {'cmap_heat': cm.vik,
         'vmode_heat': 'sym',},
}

def update_nifti_cfg(type_n):
    
    config_updated = nifti_cfg.copy()    
    config_updated = type_n_cfg(config_updated, type_n)
    
    return config_updated

def plot_nifti_wrapper(
    img: nib.Nifti1Image,
    img_type: str,
    save_path: str | Path,
    param_type: str,
    bd: str,
    bg: nib.Nifti1Image | None = None,
) -> plt.Figure:
    
    config_updated = nifti_cfg.copy() 
    key = BAND_TPL[bd]
    prefix = PREFIX_TPL[img_type]
    colorbar_label = format_label(param_type, prefix=prefix, key=key)
    config_updated['colorbar_label'] = colorbar_label
    config_updated['bg_img'] = bg
    config_updated.update(type_specified_dict[img_type])

    fig = vdf.plot_triple_interaction_nifti(
        heat_img=img,
        **config_updated
    )                       

    save_fig(fig, save_path)
    
    return fig

#%%

prism_boxplot_cfg = {

    # --- palettes ---
    'jitter_palette': palette,
    'fill_palette': palette,
    'outline_palette': palette,

    # --- visual style ---
    'jitter_width': 0.20,
    'jitter_alpha': 1,
    'jitter_size': 2,
    'xtick_rotation': 90,
    'ytick_rotation': 0,
    'grid': grid,
    'dpi': dpi,
    'seed': 42,
    'show_top_right_axes': show_top_right_axes,

    
    # --- boxplot controls ---
    'show_box': True,
    'box_width': 0.55,
    'fill_alpha': 0.30,
    'whiskers': 'tukey',
    
    # global fallback outline styles
    'box_edge_width': line_width,
    'median_linewidth': line_width,
    'whisker_linewidth': line_width,
    'cap_linewidth': line_width,

    # outliers
    'outlier_marker': "o",
    'outlier_markersize': 0,

    # Raw mean line
    'raw_mean_line_width': line_width,
    'raw_mean_line_color': "#000000",
    'raw_mean_line_style': "-",
    
    # --- vertical / horizontal references ---
    'horizontal_lines': None, 
    'hline_color': 'black',   
    'hline_style': ':',
    'hline_width': line_width,
    'hline_alpha': 0.3,

    # --- labels / scales (figure-level when single_*_label=True) ---
    'y_min_zero': True,
    'x_label_offset_mm': x_label_offset_mm,      
    'y_label_offset_mm': y_label_offset_mm,      
    
    # --- text sizes ---
    'title_fontsize': font_size,
    'axis_label_fontsize': font_size,
    'tick_label_fontsize': tick_label_size,

    # --- BOXSIZE-BASED GEOMETRY (deterministic) ---
    'include_global_label_margins': True,

    # --- Tukey bracket controls ---
    'show_brackets': True,
    'hide_ns': True,
    'y_start': 0.50,
    'y_end': 0.90,
    'y_step': 0.08,
    'bracket_height_frac': 0.0,
    'bracket_color': "black",
    'bracket_linewidth': line_width,
    'bracket_text_size': font_size,

    # --- transparency ---
    'transparent': transparent,
}


def plot_prism_boxplot_wrapper(
    save_dir: str | Path,
    fig_format: str = '.pdf',
) -> plt.Figure:
    
    save_dir = Path(save_dir)
    df = pd.read_excel(save_dir / f'df.xlsx')
    tuk = pd.read_excel(save_dir / f'tuk.xlsx')
    n = len(df.columns)
    config_updated = prism_boxplot_cfg.copy()
    boxsize = (boxsize_base[0]*n, boxsize_base[1])
    config_updated['boxsize'] = boxsize
    config_updated['y_label'] = save_dir.name
    fig = vdf.plot_prism_boxplot(
        df=df,
        tuk=tuk,
        **config_updated
    )   
    save_path = save_dir / f'{save_dir.name}{fig_format}'
    save_fig(fig, save_path)
    
    return fig

#%% Paradigm-specific plotting functions
#%%
def format_force(df):
    df["Force"] = df["Force"].astype(int).astype(str)
    df["Force"] = df["Force"].map(lambda x: f'{x} g')
    df["Force"] = pd.Categorical(
        df["Force"],
        categories=["26 g","60 g","100 g","180 g","300 g"],
        ordered=True
    )   
    return df
