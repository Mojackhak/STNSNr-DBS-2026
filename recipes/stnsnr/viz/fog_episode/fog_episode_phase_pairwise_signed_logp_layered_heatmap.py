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
from recipes.stnsnr.stats.fog_episode.fog_episode_phase_pairwise_signed_logp import (
    BAND_LABELS,
    BANDS,
    BANDS_WITH_DELTA,
    DISPLAY_COMPARISONS,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_COUNT_WITH_DELTA,
    GROUPS,
    GROUPS_WITH_DELTA,
    MODEL_DIR,
    OUTPUT_DIR,
    PARAMETERS,
    PARAMETER_GROUP_KEY,
    PARAMETER_LABEL_TOP,
    PARAMETER_REGIONS,
    SIGNED_LOGP_OUTPUT_PATH,
    SIGNED_LOGP_WITH_DELTA_OUTPUT_PATH,
    SOURCE_CONTRASTS,
    VALUE_LIMIT,
    GroupSpec,
)


COLORBAR_LABEL = r"Signed (b-a) $-\log_{10}(p_{\mathrm{Tukey}})$ (a vs b)"
CELL_SIZE_MM = (3.0, 3.0)
PARAMETER_CELL_SIZE_MM = CELL_SIZE_MM
GRID_LINE_COLOR = "black"
GRID_LINE_WIDTH = 0.6
GRID_LINE_ALPHA = 0.9
PANEL_ORDER = tuple(DISPLAY_COMPARISONS)
SUMMARY_GROUP_ORDER = ("snr_local", "stn_local", "connectivity")
EXCLUDED_HEATMAP_METRIC_KEYS = frozenset({"aperiod"})


@dataclass(frozen=True)
class HeatmapOutputSpec:
    key: str
    label_top: str
    pdf_path: Path
    png_path: Path


HEATMAP_OUTPUTS = {
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR / "fog_episode_phase_pairwise_stn_local_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "fog_episode_phase_pairwise_stn_local_signed_logp_layered_heatmap.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR / "fog_episode_phase_pairwise_snr_local_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "fog_episode_phase_pairwise_snr_local_signed_logp_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=OUTPUT_DIR / "fog_episode_phase_pairwise_connectivity_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "fog_episode_phase_pairwise_connectivity_signed_logp_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=OUTPUT_DIR / "fog_episode_phase_pairwise_params_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "fog_episode_phase_pairwise_params_signed_logp_layered_heatmap.png",
    ),
}

TRANSPOSED_HEATMAP_OUTPUTS = {
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_stn_local_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_stn_local_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_snr_local_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_snr_local_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_connectivity_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_connectivity_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_params_signed_logp_transposed_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_params_signed_logp_transposed_layered_heatmap.png",
    ),
}

SUMMARY_HEATMAP_OUTPUTS = {
    "freq_by_param": HeatmapOutputSpec(
        key="summary_freq_by_param",
        label_top="",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_freq_by_param_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_freq_by_param_signed_logp_layered_heatmap.png",
    ),
    "param_by_freq": HeatmapOutputSpec(
        key="summary_param_by_freq",
        label_top="",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_param_by_freq_signed_logp_layered_heatmap.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_param_by_freq_signed_logp_layered_heatmap.png",
    ),
}


HEATMAP_OUTPUTS_WITH_DELTA = {
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_stn_local_signed_logp_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_stn_local_signed_logp_layered_heatmap_with_delta.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_snr_local_signed_logp_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_snr_local_signed_logp_layered_heatmap_with_delta.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_connectivity_signed_logp_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_connectivity_signed_logp_layered_heatmap_with_delta.png",
    ),
}

TRANSPOSED_HEATMAP_OUTPUTS_WITH_DELTA = {
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_stn_local_signed_logp_freq_by_param_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_stn_local_signed_logp_freq_by_param_layered_heatmap_with_delta.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_snr_local_signed_logp_freq_by_param_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_snr_local_signed_logp_freq_by_param_layered_heatmap_with_delta.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_connectivity_signed_logp_freq_by_param_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_connectivity_signed_logp_freq_by_param_layered_heatmap_with_delta.png",
    ),
}

SUMMARY_HEATMAP_OUTPUTS_WITH_DELTA = {
    "freq_by_param": HeatmapOutputSpec(
        key="summary_freq_by_param",
        label_top="",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_freq_by_param_signed_logp_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_freq_by_param_signed_logp_layered_heatmap_with_delta.png",
    ),
    "param_by_freq": HeatmapOutputSpec(
        key="summary_param_by_freq",
        label_top="",
        pdf_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_param_by_freq_signed_logp_layered_heatmap_with_delta.pdf",
        png_path=OUTPUT_DIR
        / "fog_episode_phase_pairwise_summary_param_by_freq_signed_logp_layered_heatmap_with_delta.png",
    ),
}


PARAMETER_BY_COMPARISON_OUTPUT = HeatmapOutputSpec(
    key="params_by_comparison",
    label_top="",
    pdf_path=OUTPUT_DIR
    / "fog_episode_phase_pairwise_params_by_comparison_signed_logp_heatmap.pdf",
    png_path=OUTPUT_DIR
    / "fog_episode_phase_pairwise_params_by_comparison_signed_logp_heatmap.png",
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
HEATMAP_GROUPS_WITH_DELTA = tuple(
    filter_heatmap_group_metrics(group) for group in GROUPS_WITH_DELTA
)


def load_signed_logp_table(
    input_path: str | Path = SIGNED_LOGP_OUTPUT_PATH,
    *,
    expected_row_count: int = EXPECTED_ROW_COUNT,
    allow_delta: bool = False,
) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP input table: {input_path}. "
            "Run recipes/stnsnr/stats/fog_episode/fog_episode_phase_pairwise_signed_logp.py first."
        )

    table = pd.read_csv(input_path)
    string_columns = [
        "group",
        "label_top",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "source_phase_first",
        "source_phase_second",
        "plot_phase_first",
        "plot_phase_second",
        "source_model",
        "source_contrast",
        "comparison",
        "estimate_label",
        "source_csv",
        "star",
    ]
    for column in string_columns:
        if column in table.columns:
            table[column] = table[column].fillna("").astype(str)

    validate_signed_logp_input(
        table,
        expected_row_count=expected_row_count,
        allow_delta=allow_delta,
    )
    return table


def validate_signed_logp_input(
    table: pd.DataFrame,
    *,
    expected_row_count: int = EXPECTED_ROW_COUNT,
    allow_delta: bool = False,
) -> None:
    if len(table) != expected_row_count:
        raise ValueError(f"Expected {expected_row_count} input rows; got {len(table)}.")

    required_columns = [
        "group",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "source_phase_first",
        "source_phase_second",
        "plot_phase_first",
        "plot_phase_second",
        "source_model",
        "source_contrast",
        "comparison",
        "estimate_label",
        "source_estimate_first_minus_second",
        "estimate_second_minus_first",
        "p_tukey",
        "signed_logp",
        "source_csv",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Signed-logP input is missing columns: {missing_columns}")

    if (table["source_model"] != MODEL_DIR).any():
        raise ValueError(f"Signed-logP input contains rows outside {MODEL_DIR}.")
    if set(table["source_contrast"].unique()) != set(SOURCE_CONTRASTS):
        raise ValueError(f"Signed-logP input source contrasts must match {list(SOURCE_CONTRASTS)}.")
    if tuple(table["comparison"].drop_duplicates()) != PANEL_ORDER:
        raise ValueError(f"Panel order must be {list(PANEL_ORDER)}.")
    has_delta = (table["band"] == "Delta").any()
    if not allow_delta and has_delta:
        raise ValueError("Delta rows are excluded from this workflow.")
    if allow_delta and not has_delta:
        raise ValueError("The with-Delta input must include Delta rows.")

    source_estimate = pd.to_numeric(table["source_estimate_first_minus_second"], errors="coerce")
    signed_estimate = pd.to_numeric(table["estimate_second_minus_first"], errors="coerce")
    if not np.allclose(signed_estimate, -source_estimate, equal_nan=True):
        raise ValueError("estimate_second_minus_first must equal -source_estimate_first_minus_second.")

    for column in ["p_tukey", "signed_logp"]:
        invalid_count = int(pd.to_numeric(table[column], errors="coerce").isna().sum())
        if invalid_count:
            raise ValueError(f"Column {column!r} contains {invalid_count} invalid values.")

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
    comparison: str,
    bands: tuple[str, ...] = BANDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [metric.label for metric in group.metrics]
    column_labels = [BAND_LABELS[band] for band in bands]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, band_label in zip(bands, column_labels, strict=True):
            if band not in metric.bands:
                continue
            row = select_single_row(
                table,
                group=group.key,
                comparison=comparison,
                metric=metric.key,
                band=band,
                parameter="",
                region=group.region or "",
            )
            value_matrix.loc[row_label, band_label] = clip_signed_logp(row["signed_logp"])
            p_matrix.loc[row_label, band_label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_matrices(
    table: pd.DataFrame,
    *,
    comparison: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = list(PARAMETER_REGIONS)
    column_labels = [parameter.label for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter in PARAMETERS:
        for region in PARAMETER_REGIONS:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
                comparison=comparison,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[region, parameter.label] = clip_signed_logp(row["signed_logp"])
            p_matrix.loc[region, parameter.label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_by_comparison_matrices(
    table: pd.DataFrame,
    *,
    comparison: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [parameter.label for parameter in PARAMETERS]
    column_labels = list(PARAMETER_REGIONS)
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter in PARAMETERS:
        for region in PARAMETER_REGIONS:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
                comparison=comparison,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[parameter.label, region] = clip_signed_logp(row["signed_logp"])
            p_matrix.loc[parameter.label, region] = float(row["p_tukey"])

    return value_matrix, p_matrix


def clip_signed_logp(value: object) -> float:
    numeric_value = float(value)
    return float(np.clip(numeric_value, -VALUE_LIMIT, VALUE_LIMIT))


def select_single_row(
    table: pd.DataFrame,
    *,
    group: str,
    comparison: str,
    metric: str,
    band: str,
    parameter: str,
    region: str,
) -> pd.Series:
    mask = (
        (table["group"] == group)
        & (table["comparison"] == comparison)
        & (table["metric"] == metric)
        & (table["band"] == band)
        & (table["parameter"] == parameter)
        & (table["region"] == region)
    )
    matches = table.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one signed-logP row for "
            f"group={group!r}, comparison={comparison!r}, metric={metric!r}, "
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

    return vdf.plot_well_heatmap_grid_df(
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
        cbar_label_offset_mm=cfg.cbar_label_offset_mm,
        grid_lines=None,
        gline_color=GRID_LINE_COLOR,
        gline_width=GRID_LINE_WIDTH,
        gline_alpha=GRID_LINE_ALPHA,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=cfg.colorbar_pad_mm,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )


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
    outputs: dict[str, HeatmapOutputSpec] = HEATMAP_OUTPUTS,
    transposed_outputs: dict[str, HeatmapOutputSpec] = TRANSPOSED_HEATMAP_OUTPUTS,
    bands: tuple[str, ...] = BANDS,
) -> tuple[Path, Path]:
    output = transposed_outputs[group.key] if transposed else outputs[group.key]
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for comparison in PANEL_ORDER:
        value_matrix, p_matrix = build_group_matrices(
            group,
            table,
            comparison=comparison,
            bands=bands,
        )
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
    for comparison in PANEL_ORDER:
        value_matrix, p_matrix = build_parameter_matrices(table, comparison=comparison)
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


def render_parameter_by_comparison_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for comparison in PANEL_ORDER:
        value_matrix, p_matrix = build_parameter_by_comparison_matrices(
            table,
            comparison=comparison,
        )
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
        label_top=list(PANEL_ORDER),
        label_right=None,
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=PARAMETER_CELL_SIZE_MM,
        panel_gap_mm=(12.0, cfg.panel_gap[1]),
        label_top_height_mm=cfg.strip_top_height_mm,
        label_right_width_mm=cfg.strip_right_width_mm,
        strip_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_right_bg_color=cfg.label_right_bg_color,
        label_text_color=cfg.label_text_color,
        label_fontsize=5,
        label_fontweight=cfg.label_fontweight,
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=cfg.cbar_label_offset_mm,
        grid_lines=None,
        gline_color=GRID_LINE_COLOR,
        gline_width=GRID_LINE_WIDTH,
        gline_alpha=GRID_LINE_ALPHA,
        axis_label_fontsize=6,
        tick_label_fontsize=5,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=cfg.colorbar_pad_mm,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=6,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    return save_figure(fig, PARAMETER_BY_COMPARISON_OUTPUT)


def transpose_heatmap_matrices(
    value_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return value_matrix.T.copy(), p_matrix.T.copy()


def render_summary_freq_by_param_heatmap(
    table: pd.DataFrame,
    *,
    groups: tuple[GroupSpec, ...] = HEATMAP_GROUPS,
    outputs: dict[str, HeatmapOutputSpec] = SUMMARY_HEATMAP_OUTPUTS,
    bands: tuple[str, ...] = BANDS,
) -> tuple[Path, Path]:
    output = outputs["freq_by_param"]
    group_by_key = {group.key: group for group in groups}
    label_top = [group_by_key[group_key].label_top for group_key in SUMMARY_GROUP_ORDER]
    value_grid: list[list[pd.DataFrame]] = []
    p_grid: list[list[pd.DataFrame]] = []

    for comparison in PANEL_ORDER:
        value_row: list[pd.DataFrame] = []
        p_row: list[pd.DataFrame] = []
        for group_key in SUMMARY_GROUP_ORDER:
            value_matrix, p_matrix = build_group_matrices(
                group_by_key[group_key],
                table,
                comparison=comparison,
                bands=bands,
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
        cbar_label_offset_mm=cfg.cbar_label_offset_mm,
        grid_lines=None,
        gline_color=GRID_LINE_COLOR,
        gline_width=GRID_LINE_WIDTH,
        gline_alpha=GRID_LINE_ALPHA,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=cfg.colorbar_pad_mm,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    return save_figure(fig, output)


def render_summary_param_by_freq_heatmap(
    table: pd.DataFrame,
    *,
    groups: tuple[GroupSpec, ...] = HEATMAP_GROUPS,
    outputs: dict[str, HeatmapOutputSpec] = SUMMARY_HEATMAP_OUTPUTS,
    bands: tuple[str, ...] = BANDS,
) -> tuple[Path, Path]:
    output = outputs["param_by_freq"]
    group_by_key = {group.key: group for group in groups}
    label_right: list[str] = []
    value_grid: list[list[pd.DataFrame]] = []
    p_grid: list[list[pd.DataFrame]] = []

    for group_key in SUMMARY_GROUP_ORDER:
        group = group_by_key[group_key]
        value_row: list[pd.DataFrame] = []
        p_row: list[pd.DataFrame] = []
        for comparison in PANEL_ORDER:
            value_matrix, p_matrix = build_group_matrices(
                group,
                table,
                comparison=comparison,
                bands=bands,
            )
            value_row.append(value_matrix)
            p_row.append(p_matrix)
        value_grid.append(value_row)
        p_grid.append(p_row)
        label_right.append(group.label_top)

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=p_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=list(PANEL_ORDER),
        label_right=label_right,
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
        cbar_label_offset_mm=cfg.cbar_label_offset_mm,
        grid_lines=None,
        gline_color=GRID_LINE_COLOR,
        gline_width=GRID_LINE_WIDTH,
        gline_alpha=GRID_LINE_ALPHA,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=cfg.colorbar_pad_mm,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    return save_figure(fig, output)


def run_workflow() -> dict[tuple[str, str], tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table(SIGNED_LOGP_OUTPUT_PATH)
    table_with_delta = load_signed_logp_table(
        SIGNED_LOGP_WITH_DELTA_OUTPUT_PATH,
        expected_row_count=EXPECTED_ROW_COUNT_WITH_DELTA,
        allow_delta=True,
    )

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
    outputs[(PARAMETER_GROUP_KEY, "by_comparison")] = render_parameter_by_comparison_heatmap(table)
    outputs[("summary", "freq_by_param")] = render_summary_freq_by_param_heatmap(table)
    outputs[("summary", "param_by_freq")] = render_summary_param_by_freq_heatmap(table)

    for group in HEATMAP_GROUPS_WITH_DELTA:
        outputs[(group.key, "with_delta_original")] = render_group_heatmap(
            group,
            table_with_delta,
            outputs=HEATMAP_OUTPUTS_WITH_DELTA,
            transposed_outputs=TRANSPOSED_HEATMAP_OUTPUTS_WITH_DELTA,
            bands=BANDS_WITH_DELTA,
        )
        outputs[(group.key, "with_delta_transposed")] = render_group_heatmap(
            group,
            table_with_delta,
            transposed=True,
            outputs=HEATMAP_OUTPUTS_WITH_DELTA,
            transposed_outputs=TRANSPOSED_HEATMAP_OUTPUTS_WITH_DELTA,
            bands=BANDS_WITH_DELTA,
        )
    outputs[("summary", "with_delta_freq_by_param")] = render_summary_freq_by_param_heatmap(
        table_with_delta,
        groups=HEATMAP_GROUPS_WITH_DELTA,
        outputs=SUMMARY_HEATMAP_OUTPUTS_WITH_DELTA,
        bands=BANDS_WITH_DELTA,
    )
    outputs[("summary", "with_delta_param_by_freq")] = render_summary_param_by_freq_heatmap(
        table_with_delta,
        groups=HEATMAP_GROUPS_WITH_DELTA,
        outputs=SUMMARY_HEATMAP_OUTPUTS_WITH_DELTA,
        bands=BANDS_WITH_DELTA,
    )

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
        (group_key, "with_delta_original")
        for group_key in HEATMAP_OUTPUTS_WITH_DELTA
    } | {
        (group_key, "with_delta_transposed")
        for group_key in TRANSPOSED_HEATMAP_OUTPUTS_WITH_DELTA
    } | {
        ("summary", "freq_by_param"),
        ("summary", "param_by_freq"),
        (PARAMETER_GROUP_KEY, "by_comparison"),
        ("summary", "with_delta_freq_by_param"),
        ("summary", "with_delta_param_by_freq"),
    }
    if set(outputs) != expected_keys:
        raise ValueError(f"Expected output keys {sorted(expected_keys)}; got {sorted(outputs)}.")
    if PANEL_ORDER != DISPLAY_COMPARISONS:
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
