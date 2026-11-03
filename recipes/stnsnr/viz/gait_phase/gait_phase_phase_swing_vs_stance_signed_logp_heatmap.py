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
from recipes.stnsnr.stats.gait_phase.gait_phase_phase_swing_vs_stance_signed_logp import (
    BAND_LABELS,
    BANDS,
    COMPARISON_TEXT,
    EXPECTED_ROW_COUNT,
    GROUPS,
    LAT_ORDER,
    MODEL_DIR,
    NEGATIVE_DIRECTION_LABEL,
    OUTPUT_DIR,
    PARAMETERS,
    PARAMETER_GROUP_KEY,
    PARAMETER_LABEL,
    PARAMETER_REGIONS,
    POSITIVE_DIRECTION_LABEL,
    SIGNED_ESTIMATE_COLUMN,
    SIGNED_LOGP_OUTPUT_PATH,
    SOURCE_CONTRAST,
    SOURCE_ESTIMATE_COLUMN,
    SUMMARY_GROUP_ORDER,
    VALUE_LIMIT,
    GroupSpec,
)


COLORBAR_LABEL = r"Signed $-\log_{10}(p_{\mathrm{Tukey}})$"
COLORBAR_DIRECTION_Y_OFFSET_MM = 2.5
COLORBAR_DIRECTION_FONTSIZE = 6
COLORBAR_LABEL_OFFSET_MM = 12.0
CELL_SIZE_MM = (3.0, 3.0)
GRID_LINE_COLOR = "black"
GRID_LINE_WIDTH = 0.6
GRID_LINE_ALPHA = 0.9


@dataclass(frozen=True)
class HeatmapOutputSpec:
    key: str
    label_right: str
    pdf_path: Path
    png_path: Path


HEATMAP_OUTPUTS = {
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_right="STN",
        pdf_path=OUTPUT_DIR / "gait_phase_phase_stn_local_swing_vs_stance_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "gait_phase_phase_stn_local_swing_vs_stance_signed_logp_heatmap.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_right="SNr",
        pdf_path=OUTPUT_DIR / "gait_phase_phase_snr_local_swing_vs_stance_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "gait_phase_phase_snr_local_swing_vs_stance_signed_logp_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_right="SNr-/→STN",
        pdf_path=OUTPUT_DIR / "gait_phase_phase_connectivity_swing_vs_stance_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "gait_phase_phase_connectivity_swing_vs_stance_signed_logp_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_right=PARAMETER_LABEL,
        pdf_path=OUTPUT_DIR / "gait_phase_phase_params_swing_vs_stance_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "gait_phase_phase_params_swing_vs_stance_signed_logp_heatmap.png",
    ),
    "params_by_region": HeatmapOutputSpec(
        key="params_by_region",
        label_right=PARAMETER_LABEL,
        pdf_path=OUTPUT_DIR / "gait_phase_phase_params_by_region_swing_vs_stance_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "gait_phase_phase_params_by_region_swing_vs_stance_signed_logp_heatmap.png",
    ),
    "params_transposed": HeatmapOutputSpec(
        key="params_transposed",
        label_right=PARAMETER_LABEL,
        pdf_path=OUTPUT_DIR / "gait_phase_phase_params_transposed_swing_vs_stance_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "gait_phase_phase_params_transposed_swing_vs_stance_signed_logp_heatmap.png",
    ),
    "summary": HeatmapOutputSpec(
        key="summary",
        label_right="",
        pdf_path=OUTPUT_DIR / "gait_phase_phase_summary_swing_vs_stance_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "gait_phase_phase_summary_swing_vs_stance_signed_logp_heatmap.png",
    ),
}


def load_signed_logp_table(input_path: str | Path = SIGNED_LOGP_OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP input table: {input_path}. "
            "Run recipes/stnsnr/stats/gait_phase/gait_phase_phase_swing_vs_stance_signed_logp.py first."
        )

    table = pd.read_csv(input_path)
    string_columns = [
        "source_csv",
        "group",
        "group_label",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "lat",
        "source_model",
        "source_contrast",
        "comparison",
        "positive_direction",
        "negative_direction",
        "source_phase_first",
        "source_phase_second",
        "plot_phase_first",
        "plot_phase_second",
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
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "lat",
        "source_model",
        "source_contrast",
        "comparison",
        "positive_direction",
        "negative_direction",
        SOURCE_ESTIMATE_COLUMN,
        SIGNED_ESTIMATE_COLUMN,
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
        raise ValueError(f"Signed-logP input contains rows outside {SOURCE_CONTRAST!r}.")
    if (table["comparison"] != COMPARISON_TEXT).any():
        raise ValueError(f"Signed-logP input contains rows outside {COMPARISON_TEXT!r}.")
    if (table["positive_direction"] != POSITIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Positive direction must be {POSITIVE_DIRECTION_LABEL!r}.")
    if (table["negative_direction"] != NEGATIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Negative direction must be {NEGATIVE_DIRECTION_LABEL!r}.")
    if "Delta" in set(table["band"].astype(str)):
        raise ValueError("Delta band must be excluded from this workflow.")
    if set(table["lat"].unique()) != set(LAT_ORDER):
        raise ValueError(f"Lat values must match {list(LAT_ORDER)}.")

    source_estimate = pd.to_numeric(table[SOURCE_ESTIMATE_COLUMN], errors="coerce")
    signed_estimate = pd.to_numeric(table[SIGNED_ESTIMATE_COLUMN], errors="coerce")
    if not np.allclose(signed_estimate, -source_estimate, equal_nan=True):
        raise ValueError(f"{SIGNED_ESTIMATE_COLUMN} must equal -{SOURCE_ESTIMATE_COLUMN}.")

    for column in ["p_tukey", "signed_logp", "signed_logp_display"]:
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


def display_value(value: object) -> float:
    return float(np.clip(float(value), -VALUE_LIMIT, VALUE_LIMIT))


def select_single_row(
    table: pd.DataFrame,
    *,
    group: str,
    lat: str,
    metric: str,
    band: str,
    parameter: str,
    region: str,
) -> pd.Series:
    mask = (
        (table["group"] == group)
        & (table["lat"] == lat)
        & (table["metric"] == metric)
        & (table["band"] == band)
        & (table["parameter"] == parameter)
        & (table["region"] == region)
    )
    matches = table.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one signed-logP row for "
            f"group={group!r}, lat={lat!r}, metric={metric!r}, band={band!r}, "
            f"parameter={parameter!r}, region={region!r}; got {len(matches)}."
        )
    return matches.iloc[0]


def build_group_matrix(
    group: GroupSpec,
    table: pd.DataFrame,
    *,
    lat: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [metric.label for metric in group.metrics]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, column_label in zip(BANDS, column_labels, strict=True):
            row = select_single_row(
                table,
                group=group.key,
                lat=lat,
                metric=metric.key,
                band=band,
                parameter="",
                region=group.region or "",
            )
            value_matrix.loc[row_label, column_label] = display_value(row["signed_logp_display"])
            p_matrix.loc[row_label, column_label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_matrix(
    table: pd.DataFrame,
    *,
    lat: str,
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
                lat=lat,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[region, parameter.label] = display_value(row["signed_logp_display"])
            p_matrix.loc[region, parameter.label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_by_region_matrix(
    table: pd.DataFrame,
    *,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = list(LAT_ORDER)
    column_labels = [parameter.label for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for lat in LAT_ORDER:
        for parameter in PARAMETERS:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
                lat=lat,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[lat, parameter.label] = display_value(row["signed_logp_display"])
            p_matrix.loc[lat, parameter.label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_transposed_matrix(
    table: pd.DataFrame,
    *,
    lat: str,
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
                lat=lat,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[parameter.label, region] = display_value(row["signed_logp_display"])
            p_matrix.loc[parameter.label, region] = float(row["p_tukey"])

    return value_matrix, p_matrix


def plot_heatmap_grid(
    value_grid: list[list[pd.DataFrame]],
    p_grid: list[list[pd.DataFrame]],
    *,
    label_right: list[str],
    label_top: list[str] | None = None,
) -> plt.Figure:
    top_labels = list(LAT_ORDER) if label_top is None else label_top
    fig = vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=p_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=top_labels,
        label_right=label_right,
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        dpi=cfg.dpi,
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
    add_colorbar_end_labels(fig)
    return fig


def find_colorbar_axis(fig: plt.Figure) -> plt.Axes | None:
    for ax in fig.axes:
        if ax.get_ylabel() == COLORBAR_LABEL:
            return ax
    return fig.axes[-1] if fig.axes else None


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
        POSITIVE_DIRECTION_LABEL,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )
    fig.text(
        x_label,
        pos.y0 - y_offset,
        NEGATIVE_DIRECTION_LABEL,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="black",
    )


def save_figure(fig: plt.Figure, output: HeatmapOutputSpec) -> tuple[Path, Path]:
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def render_group_heatmap(group: GroupSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS[group.key]
    value_row: list[pd.DataFrame] = []
    p_row: list[pd.DataFrame] = []
    for lat in LAT_ORDER:
        value_matrix, p_matrix = build_group_matrix(group, table, lat=lat)
        value_row.append(value_matrix)
        p_row.append(p_matrix)

    fig = plot_heatmap_grid([value_row], [p_row], label_right=[output.label_right])
    return save_figure(fig, output)


def render_parameter_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS[PARAMETER_GROUP_KEY]
    value_row: list[pd.DataFrame] = []
    p_row: list[pd.DataFrame] = []
    for lat in LAT_ORDER:
        value_matrix, p_matrix = build_parameter_matrix(table, lat=lat)
        value_row.append(value_matrix)
        p_row.append(p_matrix)

    fig = plot_heatmap_grid([value_row], [p_row], label_right=[output.label_right])
    return save_figure(fig, output)


def render_parameter_by_region_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS["params_by_region"]
    value_row: list[pd.DataFrame] = []
    p_row: list[pd.DataFrame] = []
    for region in PARAMETER_REGIONS:
        value_matrix, p_matrix = build_parameter_by_region_matrix(table, region=region)
        value_row.append(value_matrix)
        p_row.append(p_matrix)

    fig = plot_heatmap_grid(
        [value_row],
        [p_row],
        label_top=list(PARAMETER_REGIONS),
        label_right=[output.label_right],
    )
    return save_figure(fig, output)


def render_parameter_transposed_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS["params_transposed"]
    value_row: list[pd.DataFrame] = []
    p_row: list[pd.DataFrame] = []
    for lat in LAT_ORDER:
        value_matrix, p_matrix = build_parameter_transposed_matrix(table, lat=lat)
        value_row.append(value_matrix)
        p_row.append(p_matrix)

    fig = plot_heatmap_grid([value_row], [p_row], label_right=[output.label_right])
    return save_figure(fig, output)


def render_summary_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS["summary"]
    group_by_key = {group.key: group for group in GROUPS}
    value_grid: list[list[pd.DataFrame]] = []
    p_grid: list[list[pd.DataFrame]] = []
    label_right: list[str] = []

    for group_key in SUMMARY_GROUP_ORDER:
        group = group_by_key[group_key]
        value_row: list[pd.DataFrame] = []
        p_row: list[pd.DataFrame] = []
        for lat in LAT_ORDER:
            value_matrix, p_matrix = build_group_matrix(group, table, lat=lat)
            value_row.append(value_matrix)
            p_row.append(p_matrix)
        value_grid.append(value_row)
        p_grid.append(p_row)
        label_right.append(group.label)

    fig = plot_heatmap_grid(value_grid, p_grid, label_right=label_right)
    return save_figure(fig, output)


def run_workflow() -> dict[str, tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table()

    outputs: dict[str, tuple[Path, Path]] = {}
    for group in GROUPS:
        outputs[group.key] = render_group_heatmap(group, table)
    outputs[PARAMETER_GROUP_KEY] = render_parameter_heatmap(table)
    outputs["params_by_region"] = render_parameter_by_region_heatmap(table)
    outputs["params_transposed"] = render_parameter_transposed_heatmap(table)
    outputs["summary"] = render_summary_heatmap(table)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[str, tuple[Path, Path]]) -> None:
    expected_keys = {group.key for group in GROUPS} | {
        PARAMETER_GROUP_KEY,
        "params_by_region",
        "params_transposed",
        "summary",
    }
    if set(outputs) != expected_keys:
        raise ValueError(f"Expected output keys {sorted(expected_keys)}; got {sorted(outputs)}.")
    if LAT_ORDER != ("Ipsi", "Contra"):
        raise ValueError(f"Top strip order changed unexpectedly: {LAT_ORDER}.")
    if SUMMARY_GROUP_ORDER != ("snr_local", "stn_local", "connectivity"):
        raise ValueError(f"Summary group order changed unexpectedly: {SUMMARY_GROUP_ORDER}.")
    if PARAMETER_REGIONS != ("SNr", "STN"):
        raise ValueError(f"Parameter top strip order changed unexpectedly: {PARAMETER_REGIONS}.")

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
    for heatmap_name, paths in heatmap_outputs.items():
        print(f"{heatmap_name}: saved heatmap PDF: {paths[0]}")
        print(f"{heatmap_name}: saved heatmap PNG: {paths[1]}")
