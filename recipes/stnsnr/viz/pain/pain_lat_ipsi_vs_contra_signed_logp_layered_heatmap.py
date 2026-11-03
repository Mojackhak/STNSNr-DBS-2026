from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plugin.default.viz_defaults as cfg
import src.viz.visualdf as vdf
from recipes.stnsnr.stats.pain.pain_lat_ipsi_vs_contra_signed_logp import (
    BAND_LABELS,
    BANDS,
    COMPARISON_TEXT,
    EXPECTED_ROW_COUNT,
    GROUPS,
    MODEL_DIR,
    NEGATIVE_DIRECTION_LABEL,
    OUTPUT_DIR,
    PARAMETERS,
    PARAMETER_GROUP_KEY,
    PARAMETER_LABEL_TOP,
    PARAMETER_REGIONS,
    PHASES,
    POSITIVE_DIRECTION_LABEL,
    SIGNED_LOGP_OUTPUT_PATH,
    SOURCE_CONTRAST,
    VALUE_LIMIT,
    GroupSpec,
)


COLORBAR_LABEL = r"Signed $-\log_{10}(p_{\mathrm{Tukey}})$"
COLORBAR_POSITIVE_ENDPOINT_LABEL = POSITIVE_DIRECTION_LABEL
COLORBAR_NEGATIVE_ENDPOINT_LABEL = NEGATIVE_DIRECTION_LABEL
COLORBAR_DIRECTION_Y_OFFSET_MM = 2.5
COLORBAR_DIRECTION_FONTSIZE = 6
COLORBAR_LABEL_OFFSET_MM = 8.0
COLORBAR_PAD_MM = 12.0
CELL_SIZE_MM = (3.0, 3.0)
PARAMETER_CELL_SIZE_MM = CELL_SIZE_MM
GRID_LINE_COLOR = "black"
GRID_LINE_WIDTH = 0.6
GRID_LINE_ALPHA = 0.9
PANEL_ORDER = tuple(PHASES)
SUMMARY_GROUP_ORDER = ("snr_local", "stn_local", "connectivity")
EXCLUDED_HEATMAP_METRIC_KEYS = frozenset({"aperiod"})


@dataclass(frozen=True)
class HeatmapOutputSpec:
    key: str
    label_top: str
    pdf_path: Path
    png_path: Path


HEATMAP_OUTPUTS = {
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR / "pain_lat_snr_local_ipsi_vs_contra_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_lat_snr_local_ipsi_vs_contra_signed_logp_layered_heatmap.png",
    ),
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR / "pain_lat_stn_local_ipsi_vs_contra_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_lat_stn_local_ipsi_vs_contra_signed_logp_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=OUTPUT_DIR / "pain_lat_connectivity_ipsi_vs_contra_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_lat_connectivity_ipsi_vs_contra_signed_logp_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=OUTPUT_DIR / "pain_lat_params_ipsi_vs_contra_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_lat_params_ipsi_vs_contra_signed_logp_layered_heatmap.png",
    ),
}

TRANSPOSED_HEATMAP_OUTPUTS = {
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR
        / "pain_lat_snr_local_ipsi_vs_contra_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "pain_lat_snr_local_ipsi_vs_contra_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR
        / "pain_lat_stn_local_ipsi_vs_contra_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "pain_lat_stn_local_ipsi_vs_contra_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=OUTPUT_DIR
        / "pain_lat_connectivity_ipsi_vs_contra_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "pain_lat_connectivity_ipsi_vs_contra_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=OUTPUT_DIR
        / "pain_lat_params_ipsi_vs_contra_signed_logp_transposed_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "pain_lat_params_ipsi_vs_contra_signed_logp_transposed_layered_heatmap.png",
    ),
}

SUMMARY_HEATMAP_OUTPUT = HeatmapOutputSpec(
    key="summary_freq_by_param",
    label_top="",
    pdf_path=OUTPUT_DIR / "pain_lat_summary_freq_by_param_ipsi_vs_contra_signed_logp_layered_heatmap.pdf",
    png_path=OUTPUT_DIR / "pain_lat_summary_freq_by_param_ipsi_vs_contra_signed_logp_layered_heatmap.png",
)

PARAMETER_BY_REGION_HEATMAP_OUTPUT = HeatmapOutputSpec(
    key="params_by_region",
    label_top="",
    pdf_path=OUTPUT_DIR / "pain_lat_params_by_region_ipsi_vs_contra_signed_logp_heatmap.pdf",
    png_path=OUTPUT_DIR / "pain_lat_params_by_region_ipsi_vs_contra_signed_logp_heatmap.png",
)


def filter_heatmap_group_metrics(group: GroupSpec) -> GroupSpec:
    return GroupSpec(
        key=group.key,
        label_top=group.label_top,
        metrics=tuple(
            metric for metric in group.metrics if metric.key not in EXCLUDED_HEATMAP_METRIC_KEYS
        ),
        tukey_file=group.tukey_file,
        region=group.region,
    )


HEATMAP_GROUPS = tuple(filter_heatmap_group_metrics(group) for group in GROUPS)


def load_signed_logp_table(input_path: str | Path = SIGNED_LOGP_OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP input table: {input_path}. "
            "Run recipes/stnsnr/stats/pain/pain_lat_ipsi_vs_contra_signed_logp.py first."
        )

    table = pd.read_csv(input_path)
    string_columns = [
        "source_csv",
        "group",
        "label_top",
        "phase",
        "region",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "source_model",
        "source_contrast",
        "comparison",
        "positive_direction",
        "negative_direction",
        "estimate_label",
        "star",
    ]
    for column in string_columns:
        if column in table.columns:
            table[column] = table[column].fillna("").astype(str)

    validate_signed_logp_input(table)
    return table


def validate_signed_logp_input(table: pd.DataFrame) -> None:
    if len(table) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} input rows; got {len(table)}.")

    required_columns = [
        "source_csv",
        "group",
        "phase",
        "region",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "source_model",
        "source_contrast",
        "comparison",
        "positive_direction",
        "negative_direction",
        "source_estimate_ipsi_minus_contra",
        "signed_estimate",
        "p_tukey",
        "signed_logp",
        "signed_logp_display",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Signed-logP input is missing columns: {missing_columns}")

    if (table["source_model"] != MODEL_DIR).any():
        raise ValueError(f"Signed-logP input contains rows outside {MODEL_DIR}.")
    if (table["source_contrast"] != SOURCE_CONTRAST).any():
        raise ValueError(f"Source contrast must be {SOURCE_CONTRAST!r}.")
    if tuple(table["phase"].drop_duplicates()) != PANEL_ORDER:
        raise ValueError(f"Panel order must be {list(PANEL_ORDER)}.")
    if (table["comparison"] != COMPARISON_TEXT).any():
        raise ValueError(f"Comparison must be {COMPARISON_TEXT!r}.")
    if (table["positive_direction"] != POSITIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Positive direction must be {POSITIVE_DIRECTION_LABEL!r}.")
    if (table["negative_direction"] != NEGATIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Negative direction must be {NEGATIVE_DIRECTION_LABEL!r}.")

    banded_metric_labels = table.loc[
        table["group"] != PARAMETER_GROUP_KEY,
        "metric_label",
    ].astype(str)
    if not banded_metric_labels.str.startswith("Δ").all():
        raise ValueError(
            "BL-normalized banded metric labels must start with 'Δ'. "
            "Regenerate the intermediate CSV with the stats script."
        )

    parameter_labels = table.loc[
        table["group"] == PARAMETER_GROUP_KEY,
        "parameter_label",
    ].astype(str)
    if not parameter_labels.str.startswith("Δ").all():
        raise ValueError(
            "BL-normalized parameter labels must start with 'Δ'. "
            "Regenerate the intermediate CSV with the stats script."
        )

    source_estimate = pd.to_numeric(table["source_estimate_ipsi_minus_contra"], errors="coerce")
    signed_estimate = pd.to_numeric(table["signed_estimate"], errors="coerce")
    if not np.allclose(source_estimate, signed_estimate, equal_nan=True):
        raise ValueError("signed_estimate must equal source_estimate_ipsi_minus_contra.")

    for column in ["p_tukey", "signed_logp", "signed_logp_display"]:
        invalid_count = int(pd.to_numeric(table[column], errors="coerce").isna().sum())
        if invalid_count:
            raise ValueError(f"Column {column!r} contains {invalid_count} invalid values.")

    display = pd.to_numeric(table["signed_logp_display"], errors="coerce")
    if ((display < -VALUE_LIMIT) | (display > VALUE_LIMIT)).any():
        raise ValueError(f"signed_logp_display must be clipped to [-{VALUE_LIMIT}, {VALUE_LIMIT}].")

    bad_paths = [
        source_csv
        for source_csv in table["source_csv"].astype(str)
        if f"/{MODEL_DIR}/" not in source_csv
    ]
    if bad_paths:
        raise ValueError(f"Found source paths outside {MODEL_DIR}: {bad_paths[:5]}")


def build_group_matrices(
    group: GroupSpec,
    table: pd.DataFrame,
    *,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [
        get_unique_label(
            table,
            label_column="metric_label",
            group=group.key,
            metric=metric.key,
            region=group.region or "",
        )
        for metric in group.metrics
    ]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, band_label in zip(BANDS, column_labels, strict=True):
            row = select_single_row(
                table,
                group=group.key,
                phase=phase,
                metric=metric.key,
                band=band,
                parameter="",
                region=group.region or "",
            )
            value_matrix.loc[row_label, band_label] = display_value(row["signed_logp_display"])
            p_matrix.loc[row_label, band_label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_matrices(
    table: pd.DataFrame,
    *,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = list(PARAMETER_REGIONS)
    column_labels = [
        get_unique_label(
            table,
            label_column="parameter_label",
            group=PARAMETER_GROUP_KEY,
            metric="params",
            parameter=parameter.key,
        )
        for parameter in PARAMETERS
    ]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter, parameter_label in zip(PARAMETERS, column_labels, strict=True):
        for region in PARAMETER_REGIONS:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
                phase=phase,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[region, parameter_label] = display_value(row["signed_logp_display"])
            p_matrix.loc[region, parameter_label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_by_region_matrices(
    table: pd.DataFrame,
    *,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [
        get_unique_label(
            table,
            label_column="parameter_label",
            group=PARAMETER_GROUP_KEY,
            metric="params",
            parameter=parameter.key,
        )
        for parameter in PARAMETERS
    ]
    column_labels = list(PANEL_ORDER)
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter, parameter_label in zip(PARAMETERS, row_labels, strict=True):
        for phase in PANEL_ORDER:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
                phase=phase,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[parameter_label, phase] = display_value(row["signed_logp_display"])
            p_matrix.loc[parameter_label, phase] = float(row["p_tukey"])

    return value_matrix, p_matrix


def get_unique_label(
    table: pd.DataFrame,
    *,
    label_column: str,
    group: str,
    metric: str,
    region: str | None = None,
    parameter: str | None = None,
) -> str:
    mask = (table["group"] == group) & (table["metric"] == metric)
    if region is not None:
        mask &= table["region"] == region
    if parameter is not None:
        mask &= table["parameter"] == parameter

    labels = table.loc[mask, label_column].drop_duplicates().tolist()
    if len(labels) != 1:
        raise ValueError(
            f"Expected exactly one {label_column!r} for group={group!r}, "
            f"metric={metric!r}, region={region!r}, parameter={parameter!r}; "
            f"got {labels}."
        )
    return str(labels[0])


def display_value(value: object) -> float:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(float(numeric_value)):
        return np.nan
    return float(np.clip(float(numeric_value), -VALUE_LIMIT, VALUE_LIMIT))


def select_single_row(
    table: pd.DataFrame,
    *,
    group: str,
    phase: str,
    metric: str,
    band: str,
    parameter: str,
    region: str,
) -> pd.Series:
    mask = (
        (table["group"] == group)
        & (table["phase"] == phase)
        & (table["metric"] == metric)
        & (table["band"] == band)
        & (table["parameter"] == parameter)
        & (table["region"] == region)
    )
    matches = table.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one signed-logP row for "
            f"group={group!r}, phase={phase!r}, metric={metric!r}, "
            f"band={band!r}, parameter={parameter!r}, region={region!r}; "
            f"got {len(matches)}."
        )
    return matches.iloc[0]


def plot_layered_heatmap(
    value_matrices: list[pd.DataFrame],
    p_matrices: list[pd.DataFrame],
    *,
    label_top: str,
    panel_labels: tuple[str, ...] = PANEL_ORDER,
    cellsize_mm: tuple[float, float] = CELL_SIZE_MM,
) -> plt.Figure:
    if len(value_matrices) != len(panel_labels):
        raise ValueError(
            f"Expected {len(panel_labels)} value matrices; got {len(value_matrices)}."
        )
    if len(p_matrices) != len(panel_labels):
        raise ValueError(f"Expected {len(panel_labels)} p-value matrices; got {len(p_matrices)}.")
    if not value_matrices:
        raise ValueError("At least one panel is required.")

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[[value_matrix] for value_matrix in value_matrices],
        df_ps=[[p_matrix] for p_matrix in p_matrices],
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=[label_top],
        label_right=list(panel_labels),
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=cellsize_mm,
        panel_gap_mm=(0.0, cfg.panel_gap[1]),
        label_top_height_mm=cfg.strip_top_height_mm,
        label_right_width_mm=cfg.strip_right_width_mm,
        strip_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_right_bg_color=cfg.label_right_bg_color,
        label_text_color=cfg.label_text_color,
        label_fontsize=7,
        label_fontweight=cfg.label_fontweight,
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=COLORBAR_LABEL_OFFSET_MM,
        grid_lines=None,
        gline_color=GRID_LINE_COLOR,
        gline_width=GRID_LINE_WIDTH,
        gline_alpha=GRID_LINE_ALPHA,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=COLORBAR_PAD_MM,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(fig)
    return fig


def save_figure(fig: plt.Figure, output: HeatmapOutputSpec) -> tuple[Path, Path]:
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def render_group_heatmap(
    group: GroupSpec,
    table: pd.DataFrame,
    *,
    transposed: bool = False,
) -> tuple[Path, Path]:
    output = TRANSPOSED_HEATMAP_OUTPUTS[group.key] if transposed else HEATMAP_OUTPUTS[group.key]
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for phase in PANEL_ORDER:
        value_matrix, p_matrix = build_group_matrices(group, table, phase=phase)
        if transposed:
            value_matrix, p_matrix = transpose_heatmap_matrices(value_matrix, p_matrix)
        value_matrices.append(value_matrix)
        p_matrices.append(p_matrix)

    fig = plot_layered_heatmap(
        value_matrices,
        p_matrices,
        label_top=output.label_top,
        cellsize_mm=CELL_SIZE_MM,
    )
    return save_figure(fig, output)


def render_parameter_heatmap(table: pd.DataFrame, *, transposed: bool = False) -> tuple[Path, Path]:
    output = (
        TRANSPOSED_HEATMAP_OUTPUTS[PARAMETER_GROUP_KEY]
        if transposed
        else HEATMAP_OUTPUTS[PARAMETER_GROUP_KEY]
    )
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for phase in PANEL_ORDER:
        value_matrix, p_matrix = build_parameter_matrices(table, phase=phase)
        if transposed:
            value_matrix, p_matrix = transpose_heatmap_matrices(value_matrix, p_matrix)
        value_matrices.append(value_matrix)
        p_matrices.append(p_matrix)

    fig = plot_layered_heatmap(
        value_matrices,
        p_matrices,
        label_top=output.label_top,
        cellsize_mm=PARAMETER_CELL_SIZE_MM,
    )
    return save_figure(fig, output)


def transpose_heatmap_matrices(
    value_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return value_matrix.T.copy(), p_matrix.T.copy()


def render_summary_freq_by_param_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    group_by_key = {group.key: group for group in HEATMAP_GROUPS}
    label_top = [group_by_key[group_key].label_top for group_key in SUMMARY_GROUP_ORDER]
    value_grid: list[list[pd.DataFrame]] = []
    p_grid: list[list[pd.DataFrame]] = []

    for phase in PANEL_ORDER:
        value_row: list[pd.DataFrame] = []
        p_row: list[pd.DataFrame] = []
        for group_key in SUMMARY_GROUP_ORDER:
            value_matrix, p_matrix = build_group_matrices(
                group_by_key[group_key],
                table,
                phase=phase,
            )
            value_matrix, p_matrix = transpose_heatmap_matrices(value_matrix, p_matrix)
            value_row.append(value_matrix)
            p_row.append(p_matrix)
        value_grid.append(value_row)
        p_grid.append(p_row)

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=p_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=label_top,
        label_right=list(PANEL_ORDER),
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=CELL_SIZE_MM,
        panel_gap_mm=cfg.panel_gap,
        label_top_height_mm=cfg.strip_top_height_mm,
        label_right_width_mm=cfg.strip_right_width_mm,
        strip_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_right_bg_color=cfg.label_right_bg_color,
        label_text_color=cfg.label_text_color,
        label_fontsize=7,
        label_fontweight=cfg.label_fontweight,
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=COLORBAR_LABEL_OFFSET_MM,
        grid_lines=None,
        gline_color=GRID_LINE_COLOR,
        gline_width=GRID_LINE_WIDTH,
        gline_alpha=GRID_LINE_ALPHA,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=COLORBAR_PAD_MM,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(fig)
    return save_figure(fig, SUMMARY_HEATMAP_OUTPUT)


def render_parameter_by_region_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for region in PARAMETER_REGIONS:
        value_matrix, p_matrix = build_parameter_by_region_matrices(table, region=region)
        value_matrices.append(value_matrix)
        p_matrices.append(p_matrix)

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[value_matrices],
        df_ps=[p_matrices],
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=list(PARAMETER_REGIONS),
        label_right=None,
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=PARAMETER_CELL_SIZE_MM,
        panel_gap_mm=cfg.panel_gap,
        label_top_height_mm=cfg.strip_top_height_mm,
        label_right_width_mm=cfg.strip_right_width_mm,
        strip_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_right_bg_color=cfg.label_right_bg_color,
        label_text_color=cfg.label_text_color,
        label_fontsize=7,
        label_fontweight=cfg.label_fontweight,
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=COLORBAR_LABEL_OFFSET_MM,
        grid_lines=None,
        gline_color=GRID_LINE_COLOR,
        gline_width=GRID_LINE_WIDTH,
        gline_alpha=GRID_LINE_ALPHA,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=COLORBAR_PAD_MM,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(fig)
    return save_figure(fig, PARAMETER_BY_REGION_HEATMAP_OUTPUT)


def set_colorbar_ticks(fig: plt.Figure) -> None:
    colorbar_axis = find_colorbar_axis(fig)
    if colorbar_axis is None:
        return
    colorbar_axis.set_yticks([-VALUE_LIMIT, 0.0, VALUE_LIMIT])
    colorbar_axis.set_yticklabels([f"{-VALUE_LIMIT:g}", "0", f"{VALUE_LIMIT:g}"])


def add_colorbar_end_labels(
    fig: plt.Figure,
    *,
    y_offset_mm: float = COLORBAR_DIRECTION_Y_OFFSET_MM,
    fontsize: float = COLORBAR_DIRECTION_FONTSIZE,
) -> None:
    colorbar_axis = find_colorbar_axis(fig)
    if colorbar_axis is None:
        return

    pos = colorbar_axis.get_position()
    fig_height_in = fig.get_size_inches()[1]
    y_offset = (y_offset_mm / 25.4) / fig_height_in
    x_label = (pos.x0 + pos.x1) / 2.0
    fig.text(
        x_label,
        pos.y1 + y_offset,
        COLORBAR_POSITIVE_ENDPOINT_LABEL,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )
    fig.text(
        x_label,
        pos.y0 - y_offset,
        COLORBAR_NEGATIVE_ENDPOINT_LABEL,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="black",
    )


def find_colorbar_axis(fig: plt.Figure) -> plt.Axes | None:
    for ax in fig.axes:
        if ax.get_ylabel() == COLORBAR_LABEL:
            return ax
    return fig.axes[-1] if fig.axes else None


def run_workflow() -> dict[tuple[str, str], tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table(SIGNED_LOGP_OUTPUT_PATH)

    outputs: dict[tuple[str, str], tuple[Path, Path]] = {}
    for group in HEATMAP_GROUPS:
        outputs[(group.key, "original")] = render_group_heatmap(group, table)
        outputs[(group.key, "transposed")] = render_group_heatmap(
            group,
            table,
            transposed=True,
        )
    outputs[(PARAMETER_GROUP_KEY, "original")] = render_parameter_heatmap(table)
    outputs[(PARAMETER_GROUP_KEY, "transposed")] = render_parameter_heatmap(
        table,
        transposed=True,
    )
    outputs[("summary", "freq_by_param")] = render_summary_freq_by_param_heatmap(table)
    outputs[(PARAMETER_GROUP_KEY, "by_region")] = render_parameter_by_region_heatmap(table)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[tuple[str, str], tuple[Path, Path]]) -> None:
    expected_keys = {
        (group_key, "original")
        for group_key in HEATMAP_OUTPUTS
    } | {
        (group_key, "transposed")
        for group_key in TRANSPOSED_HEATMAP_OUTPUTS
    } | {
        ("summary", "freq_by_param"),
        (PARAMETER_GROUP_KEY, "by_region"),
    }
    if set(outputs) != expected_keys:
        raise ValueError(f"Expected output keys {sorted(expected_keys)}; got {sorted(outputs)}.")
    if PANEL_ORDER != PHASES:
        raise ValueError(f"Panel order changed unexpectedly: {PANEL_ORDER}")

    missing = [
        str(path)
        for paths in outputs.values()
        for path in paths
        if not Path(path).exists() or Path(path).stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError(f"Expected non-empty heatmap files were not created: {missing}")

    wrong_parent = [
        str(path)
        for paths in outputs.values()
        for path in paths
        if Path(path).parent != OUTPUT_DIR
    ]
    if wrong_parent:
        raise ValueError(f"Heatmap outputs must be written only under {OUTPUT_DIR}: {wrong_parent}")


if __name__ == "__main__":
    heatmap_outputs = run_workflow()
    for (group_name, variant), paths in heatmap_outputs.items():
        print(f"{group_name}/{variant}: saved heatmap PDF: {paths[0]}")
        print(f"{group_name}/{variant}: saved heatmap PNG: {paths[1]}")
