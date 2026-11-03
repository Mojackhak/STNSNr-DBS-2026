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
from recipes.stnsnr.stats.pain.pain_logforce_standardized_slope import (
    BAND_LABELS,
    BANDS,
    CONNECTIVITY_METRICS,
    CONNECTIVITY_REGION,
    LOCAL_METRICS,
    MODEL_DIR,
    OUTPUT_DIR,
    OUTPUT_PATH,
    PARAMETERS,
    PARAMETER_GROUP_KEY,
    PARAMETER_LABEL_TOP,
    PHASES,
    REGIONS,
    VALUE_LIMIT,
    MetricSpec,
)


COLORBAR_LABEL = r"Standardized slope of ΔLFP per centered $\log_{10}(\mathrm{Force})$"
CELL_SIZE_MM = (3.0, 3.0)
PARAMETER_CELL_SIZE_MM = CELL_SIZE_MM
PANEL_ORDER = tuple(PHASES)
SUMMARY_GROUP_ORDER = ("snr_local", "stn_local", "connectivity")


@dataclass(frozen=True)
class HeatmapOutputSpec:
    key: str
    label_top: str
    pdf_path: Path
    png_path: Path


@dataclass(frozen=True)
class HeatmapGroupSpec:
    key: str
    label_top: str
    metrics: tuple[MetricSpec, ...]
    region: str


HEATMAP_GROUPS = (
    HeatmapGroupSpec("snr_local", "SNr", LOCAL_METRICS, "SNr"),
    HeatmapGroupSpec("stn_local", "STN", LOCAL_METRICS, "STN"),
    HeatmapGroupSpec("connectivity", CONNECTIVITY_REGION, CONNECTIVITY_METRICS, CONNECTIVITY_REGION),
)

HEATMAP_OUTPUTS = {
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR / "pain_logforce_snr_local_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_snr_local_standardized_slope_layered_heatmap.png",
    ),
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR / "pain_logforce_stn_local_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_stn_local_standardized_slope_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top=CONNECTIVITY_REGION,
        pdf_path=OUTPUT_DIR / "pain_logforce_connectivity_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_connectivity_standardized_slope_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=OUTPUT_DIR / "pain_logforce_params_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_params_standardized_slope_layered_heatmap.png",
    ),
}

TRANSPOSED_HEATMAP_OUTPUTS = {
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR / "pain_logforce_snr_local_standardized_slope_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_snr_local_standardized_slope_freq_by_param_layered_heatmap.png",
    ),
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=OUTPUT_DIR / "pain_logforce_stn_local_standardized_slope_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_stn_local_standardized_slope_freq_by_param_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top=CONNECTIVITY_REGION,
        pdf_path=OUTPUT_DIR / "pain_logforce_connectivity_standardized_slope_freq_by_param_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_connectivity_standardized_slope_freq_by_param_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=OUTPUT_DIR / "pain_logforce_params_standardized_slope_transposed_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "pain_logforce_params_standardized_slope_transposed_layered_heatmap.png",
    ),
}

SUMMARY_HEATMAP_OUTPUT = HeatmapOutputSpec(
    key="summary_freq_by_param",
    label_top="",
    pdf_path=OUTPUT_DIR / "pain_logforce_summary_freq_by_param_standardized_slope_layered_heatmap.pdf",
    png_path=OUTPUT_DIR / "pain_logforce_summary_freq_by_param_standardized_slope_layered_heatmap.png",
)

PHASE_BY_REGION_PARAMETER_OUTPUT = HeatmapOutputSpec(
    key="params_phase_by_region",
    label_top="",
    pdf_path=OUTPUT_DIR / "pain_logforce_params_phase_by_region_standardized_slope_layered_heatmap.pdf",
    png_path=OUTPUT_DIR / "pain_logforce_params_phase_by_region_standardized_slope_layered_heatmap.png",
)


def load_standardized_slope_table(input_path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing standardized slope table: {input_path}. "
            "Run recipes/stnsnr/stats/pain/pain_logforce_standardized_slope.py first."
        )

    table = pd.read_csv(input_path)
    string_columns = [
        "feature_group",
        "group",
        "label_top",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "phase",
        "predictor",
        "star",
        "source_slope_csv",
        "source_sd_csv",
        "sd_source",
    ]
    for column in string_columns:
        if column in table.columns:
            table[column] = table[column].fillna("").astype(str)

    validate_standardized_slope_input(table)
    return table


def validate_standardized_slope_input(table: pd.DataFrame) -> None:
    expected_local_rows = len(LOCAL_METRICS) * len(BANDS) * len(REGIONS) * len(PANEL_ORDER)
    expected_connectivity_rows = len(CONNECTIVITY_METRICS) * len(BANDS) * len(PANEL_ORDER)
    expected_parameter_rows = len(PARAMETERS) * len(REGIONS) * len(PANEL_ORDER)
    expected_rows = expected_local_rows + expected_connectivity_rows + expected_parameter_rows
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} input rows; got {len(table)}.")

    required_columns = [
        "feature_group",
        "group",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "phase",
        "q",
        "standardized_slope",
        "standardized_slope_display",
        "source_slope_csv",
        "source_sd_csv",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Standardized slope input is missing columns: {missing_columns}")

    if set(table["phase"].drop_duplicates()) != set(PANEL_ORDER):
        raise ValueError(f"Panel phases must match {list(PANEL_ORDER)}.")

    display = pd.to_numeric(table["standardized_slope_display"], errors="coerce")
    if display.isna().any():
        raise ValueError("standardized_slope_display contains invalid values.")
    if ((display < -VALUE_LIMIT) | (display > VALUE_LIMIT)).any():
        raise ValueError(f"standardized_slope_display must be clipped to [-{VALUE_LIMIT}, {VALUE_LIMIT}].")

    for column in ["q", "standardized_slope"]:
        invalid_count = int(pd.to_numeric(table[column], errors="coerce").isna().sum())
        if invalid_count:
            raise ValueError(f"Column {column!r} contains {invalid_count} invalid values.")

    bad_paths = [
        source_csv
        for source_csv in table["source_slope_csv"].astype(str)
        if f"/{MODEL_DIR}/" not in source_csv
    ]
    if bad_paths:
        raise ValueError(f"Found source paths outside {MODEL_DIR}: {bad_paths[:5]}")

    bad_sd_paths = [
        source_csv
        for source_csv in table["source_sd_csv"].astype(str)
        if not source_csv.endswith("_region_normalized.csv")
    ]
    if bad_sd_paths:
        raise ValueError(f"Found non-region_normalized SD sources: {bad_sd_paths[:5]}")


def pain_change_label(label: object) -> str:
    text = str(label)
    return text if text.startswith("Δ") else f"Δ{text}"


def build_group_matrices(
    group: HeatmapGroupSpec,
    table: pd.DataFrame,
    *,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [
        pain_change_label(
            get_unique_label(
                table,
                label_column="metric_label",
                group=group.key,
                metric=metric.key,
                region=group.region,
            )
        )
        for metric in group.metrics
    ]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, band_label in zip(BANDS, column_labels, strict=True):
            row = select_single_row(
                table,
                group=group.key,
                phase=phase,
                metric=metric.key,
                band=band,
                parameter="",
                region=group.region,
            )
            value_matrix.loc[row_label, band_label] = display_value(row["standardized_slope_display"])
            q_matrix.loc[row_label, band_label] = float(row["q"])

    return value_matrix, q_matrix


def build_parameter_matrices(
    table: pd.DataFrame,
    *,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = list(REGIONS)
    column_labels = [
        pain_change_label(
            get_unique_label(
                table,
                label_column="parameter_label",
                group=PARAMETER_GROUP_KEY,
                metric="params",
                parameter=parameter.key,
            )
        )
        for parameter in PARAMETERS
    ]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter, parameter_label in zip(PARAMETERS, column_labels, strict=True):
        for region in REGIONS:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
                phase=phase,
                metric="params",
                band=parameter.key,
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[region, parameter_label] = display_value(row["standardized_slope_display"])
            q_matrix.loc[region, parameter_label] = float(row["q"])

    return value_matrix, q_matrix


def build_parameter_phase_by_region_matrices(
    table: pd.DataFrame,
    *,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [
        pain_change_label(
            get_unique_label(
                table,
                label_column="parameter_label",
                group=PARAMETER_GROUP_KEY,
                metric="params",
                parameter=parameter.key,
            )
        )
        for parameter in PARAMETERS
    ]
    column_labels = list(PANEL_ORDER)
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter, row_label in zip(PARAMETERS, row_labels, strict=True):
        for phase in PANEL_ORDER:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
                phase=phase,
                metric="params",
                band=parameter.key,
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[row_label, phase] = display_value(row["standardized_slope_display"])
            q_matrix.loc[row_label, phase] = float(row["q"])

    return value_matrix, q_matrix


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
            f"metric={metric!r}, region={region!r}, parameter={parameter!r}; got {labels}."
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
            "Expected exactly one standardized slope row for "
            f"group={group!r}, phase={phase!r}, metric={metric!r}, "
            f"band={band!r}, parameter={parameter!r}, region={region!r}; got {len(matches)}."
        )
    return matches.iloc[0]


def plot_layered_heatmap(
    value_matrices: list[pd.DataFrame],
    q_matrices: list[pd.DataFrame],
    *,
    label_top: str,
    panel_labels: tuple[str, ...] = PANEL_ORDER,
    cellsize_mm: tuple[float, float] = CELL_SIZE_MM,
) -> plt.Figure:
    if len(value_matrices) != len(panel_labels):
        raise ValueError(f"Expected {len(panel_labels)} value matrices; got {len(value_matrices)}.")
    if len(q_matrices) != len(panel_labels):
        raise ValueError(f"Expected {len(panel_labels)} q-value matrices; got {len(q_matrices)}.")
    if not value_matrices:
        raise ValueError("At least one panel is required.")

    return vdf.plot_well_heatmap_grid_df(
        df_values=[[value_matrix] for value_matrix in value_matrices],
        df_ps=[[q_matrix] for q_matrix in q_matrices],
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
        gline_width=0.6,
        gline_alpha=0.9,
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


def transpose_heatmap_matrices(
    value_matrix: pd.DataFrame,
    q_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return value_matrix.T.copy(), q_matrix.T.copy()


def save_figure(fig: plt.Figure, output: HeatmapOutputSpec) -> tuple[Path, Path]:
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def render_group_heatmap(
    group: HeatmapGroupSpec,
    table: pd.DataFrame,
    *,
    transposed: bool = False,
) -> tuple[Path, Path]:
    output = TRANSPOSED_HEATMAP_OUTPUTS[group.key] if transposed else HEATMAP_OUTPUTS[group.key]
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for phase in PANEL_ORDER:
        value_matrix, q_matrix = build_group_matrices(group, table, phase=phase)
        if transposed:
            value_matrix, q_matrix = transpose_heatmap_matrices(value_matrix, q_matrix)
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_layered_heatmap(
        value_matrices,
        q_matrices,
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
    q_matrices: list[pd.DataFrame] = []
    for phase in PANEL_ORDER:
        value_matrix, q_matrix = build_parameter_matrices(table, phase=phase)
        if transposed:
            value_matrix, q_matrix = transpose_heatmap_matrices(value_matrix, q_matrix)
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_layered_heatmap(
        value_matrices,
        q_matrices,
        label_top=output.label_top,
        cellsize_mm=PARAMETER_CELL_SIZE_MM,
    )
    return save_figure(fig, output)


def render_summary_freq_by_param_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    group_by_key = {group.key: group for group in HEATMAP_GROUPS}
    label_top = [group_by_key[group_key].label_top for group_key in SUMMARY_GROUP_ORDER]
    value_grid: list[list[pd.DataFrame]] = []
    q_grid: list[list[pd.DataFrame]] = []

    for phase in PANEL_ORDER:
        value_row: list[pd.DataFrame] = []
        q_row: list[pd.DataFrame] = []
        for group_key in SUMMARY_GROUP_ORDER:
            value_matrix, q_matrix = build_group_matrices(
                group_by_key[group_key],
                table,
                phase=phase,
            )
            value_matrix, q_matrix = transpose_heatmap_matrices(value_matrix, q_matrix)
            value_row.append(value_matrix)
            q_row.append(q_matrix)
        value_grid.append(value_row)
        q_grid.append(q_row)

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=q_grid,
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
        gline_width=0.6,
        gline_alpha=0.9,
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
    return save_figure(fig, SUMMARY_HEATMAP_OUTPUT)


def render_parameter_phase_by_region_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    value_row: list[pd.DataFrame] = []
    q_row: list[pd.DataFrame] = []
    for region in REGIONS:
        value_matrix, q_matrix = build_parameter_phase_by_region_matrices(table, region=region)
        value_row.append(value_matrix)
        q_row.append(q_matrix)

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[value_row],
        df_ps=[q_row],
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=list(REGIONS),
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
        cbar_label_offset_mm=cfg.cbar_label_offset_mm,
        grid_lines=None,
        gline_width=0.6,
        gline_alpha=0.9,
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
        show_x_ticklabels="all",
        show_y_ticklabels="left",
        transparent=True,
    )
    return save_figure(fig, PHASE_BY_REGION_PARAMETER_OUTPUT)


def run_workflow() -> dict[tuple[str, str], tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_standardized_slope_table(OUTPUT_PATH)

    outputs: dict[tuple[str, str], tuple[Path, Path]] = {}
    for group in HEATMAP_GROUPS:
        outputs[(group.key, "original")] = render_group_heatmap(group, table)
        outputs[(group.key, "transposed")] = render_group_heatmap(group, table, transposed=True)
    outputs[(PARAMETER_GROUP_KEY, "original")] = render_parameter_heatmap(table)
    outputs[(PARAMETER_GROUP_KEY, "transposed")] = render_parameter_heatmap(table, transposed=True)
    outputs[("summary", "freq_by_param")] = render_summary_freq_by_param_heatmap(table)
    outputs[(PARAMETER_GROUP_KEY, "phase_by_region")] = render_parameter_phase_by_region_heatmap(table)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[tuple[str, str], tuple[Path, Path]]) -> None:
    expected_keys = (
        {(group.key, "original") for group in HEATMAP_GROUPS}
        | {(group.key, "transposed") for group in HEATMAP_GROUPS}
        | {
            (PARAMETER_GROUP_KEY, "original"),
            (PARAMETER_GROUP_KEY, "transposed"),
            (PARAMETER_GROUP_KEY, "phase_by_region"),
            ("summary", "freq_by_param"),
        }
    )
    if set(outputs) != expected_keys:
        raise ValueError(f"Expected output keys {sorted(expected_keys)}; got {sorted(outputs)}.")

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
