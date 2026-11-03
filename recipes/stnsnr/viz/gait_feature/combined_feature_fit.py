from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import plugin.default.viz_defaults as cfg
import src.viz.visualdf as vdf


COMBINED_X_VAR = "z_Feature"
GRADIENT_VAR = "Gradient"
GRADIENT_LEVELS = ["Between", "Within"]
PREDICTOR_BY_GRADIENT = {
    "Between": "z_Feature_between",
    "Within": "z_Feature_within",
}


def _combined_filebase(feature: str) -> str:
    return f"{COMBINED_X_VAR}-Region-{GRADIENT_VAR}_{feature}"


def _model_filebase(predictor: str, feature: str) -> str:
    return f"{predictor}-Region_{feature}"


def _model_fallback_filebase(predictor: str, feature: str) -> str:
    return f"{predictor}_{feature}"


def _display_feature_label(feature: str, feature_label: str | None) -> str:
    label = feature_label or feature.replace("_", " ").lower()
    return f"{label[:1].upper()}{label[1:]}"


def _read_model_table(
    save_dir: Path,
    predictor: str,
    feature: str,
    suffix: str,
    param_type: str,
) -> pd.DataFrame:
    primary = save_dir / f"{_model_filebase(predictor, feature)}_{suffix}.csv"
    fallback = save_dir / f"{_model_fallback_filebase(predictor, feature)}_{suffix}.csv"

    if primary.exists():
        table = pd.read_csv(primary)
    elif fallback.exists():
        table = pd.read_csv(fallback)
    else:
        raise FileNotFoundError(
            f"Missing model table for {predictor} ({suffix}); tried {primary} and {fallback}."
        )

    if "Region" not in table.columns:
        region_label = cfg.connection_region_label(param_type)
        if region_label is None:
            raise KeyError(
                f"{primary} / {fallback} did not contain Region, and no connection label "
                f"is defined for param_type={param_type!r}."
            )
        table["Region"] = region_label

    return table


def _combined_raw(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for gradient in GRADIENT_LEVELS:
        predictor = PREDICTOR_BY_GRADIENT[gradient]
        if predictor not in df.columns:
            raise KeyError(f"Raw dataframe is missing required predictor column: {predictor}")
        piece = df.copy()
        piece[COMBINED_X_VAR] = piece[predictor]
        piece[GRADIENT_VAR] = gradient
        pieces.append(piece)

    out = pd.concat(pieces, ignore_index=True)
    out[GRADIENT_VAR] = pd.Categorical(out[GRADIENT_VAR], categories=GRADIENT_LEVELS, ordered=True)
    return out


def _combined_curve(save_dir: Path, feature: str, param_type: str) -> pd.DataFrame:
    pieces = []
    for gradient in GRADIENT_LEVELS:
        predictor = PREDICTOR_BY_GRADIENT[gradient]
        table = _read_model_table(save_dir, predictor, feature, "curve", param_type)
        if predictor not in table.columns:
            raise KeyError(f"Curve table for {predictor} is missing predictor column: {predictor}")
        table = table.rename(columns={predictor: COMBINED_X_VAR})
        table[GRADIENT_VAR] = gradient
        pieces.append(table)

    out = pd.concat(pieces, ignore_index=True)
    out[GRADIENT_VAR] = pd.Categorical(out[GRADIENT_VAR], categories=GRADIENT_LEVELS, ordered=True)
    return out


def _combined_slope(save_dir: Path, feature: str, param_type: str) -> pd.DataFrame:
    pieces = []
    for gradient in GRADIENT_LEVELS:
        predictor = PREDICTOR_BY_GRADIENT[gradient]
        table = _read_model_table(save_dir, predictor, feature, "slope", param_type)
        table[GRADIENT_VAR] = gradient
        pieces.append(table)

    out = pd.concat(pieces, ignore_index=True)
    out[GRADIENT_VAR] = pd.Categorical(out[GRADIENT_VAR], categories=GRADIENT_LEVELS, ordered=True)
    return out


def plot_combined_feature_fit(
    df: pd.DataFrame,
    df_type: str,
    save_dir: str | Path,
    param_type: str,
    bd: str,
    feature: str,
    feature_label: str | None = None,
    fig_format: str = ".pdf",
) -> plt.Figure:
    save_dir = Path(save_dir)

    raw = _combined_raw(df)
    curve = _combined_curve(save_dir, feature, param_type)
    slope = _combined_slope(save_dir, feature, param_type)

    config_updated = cfg.update_fit_cfg(type_n=3)
    config_updated["x_label"] = f"{_display_feature_label(feature, feature_label)} (z-scored)"
    config_updated["y_label"] = cfg.format_label(
        param_type,
        prefix=cfg.PREFIX_TPL[df_type],
        key=cfg.BAND_TPL[bd],
    )

    fig = vdf.plot_triple_interaction_fit(
        df=raw,
        curve=curve,
        slope=slope,
        x_var=COMBINED_X_VAR,
        panel_var="Region",
        facet_var=GRADIENT_VAR,
        panel_levels=cfg.present_levels(raw, "Region"),
        facet_levels=GRADIENT_LEVELS,
        **config_updated,
    )

    save_path = save_dir / f"{_combined_filebase(feature)}_{df_type}{fig_format}"
    cfg.save_fig(fig, save_path)
    return fig
