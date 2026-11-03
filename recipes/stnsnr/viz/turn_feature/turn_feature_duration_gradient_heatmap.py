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
from recipes.stnsnr.stats.turn_feature.turn_feature_duration_standardized_slope import (
    BAND_LABELS,
    BANDS,
    CONNECTIVITY_METRICS,
    GRADIENTS,
    LOCAL_METRICS,
    OUTPUT_DIR,
    OUTPUT_PATH,
    PARAMETER_REGIONS,
    PARAMETERS,
    VALUE_LIMIT,
    validate_standardized_slope_table,
)


COLORBAR_LABEL = "Standardized slope of ΔLFP (Turn - Stand) per z-scored turn duration"
CELL_SIZE_MM = (3.0, 3.0)


@dataclass(frozen=True)
class FrequencyHeatmapSpec:
    key: str
    heatmap_group: str
    label_top: str
    metrics: tuple
    pdf_path: Path
    png_path: Path


@dataclass(frozen=True)
class ParameterHeatmapSpec:
    key: str
    heatmap_group: str
    pdf_path: Path
    png_path: Path


@dataclass(frozen=True)
class FrequencySummaryHeatmapSpec:
    key: str
    pdf_path: Path
    png_path: Path


FREQUENCY_HEATMAPS = (
    FrequencyHeatmapSpec(
        key="stn_local",
        heatmap_group="stn_local",
        label_top="STN",
        metrics=LOCAL_METRICS,
        pdf_path=OUTPUT_DIR / "turn_feature_stn_local_duration_standardized_slope_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_feature_stn_local_duration_standardized_slope_heatmap.png",
    ),
    FrequencyHeatmapSpec(
        key="snr_local",
        heatmap_group="snr_local",
        label_top="SNr",
        metrics=LOCAL_METRICS,
        pdf_path=OUTPUT_DIR / "turn_feature_snr_local_duration_standardized_slope_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_feature_snr_local_duration_standardized_slope_heatmap.png",
    ),
    FrequencyHeatmapSpec(
        key="connectivity",
        heatmap_group="connectivity",
        label_top="SNr-/→STN",
        metrics=CONNECTIVITY_METRICS,
        pdf_path=OUTPUT_DIR / "turn_feature_connectivity_duration_standardized_slope_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_feature_connectivity_duration_standardized_slope_heatmap.png",
    ),
)

PARAMETER_HEATMAPS = (
    ParameterHeatmapSpec(
        key="params",
        heatmap_group="params",
        pdf_path=OUTPUT_DIR / "turn_feature_params_duration_standardized_slope_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_feature_params_duration_standardized_slope_heatmap.png",
    ),
)

FREQUENCY_SUMMARY_HEATMAPS = (
    FrequencySummaryHeatmapSpec(
        key="frequency_summary",
        pdf_path=OUTPUT_DIR / "turn_feature_frequency_summary_duration_standardized_slope_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_feature_frequency_summary_duration_standardized_slope_heatmap.png",
    ),
)
FREQUENCY_SUMMARY_ROWS = (
    FREQUENCY_HEATMAPS[1],
    FREQUENCY_HEATMAPS[0],
    FREQUENCY_HEATMAPS[2],
)


def load_standardized_slope_table(path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing standardized slope table: {input_path}. "
            "Run recipes/stnsnr/stats/turn_feature/turn_feature_duration_standardized_slope.py first."
        )

    table = pd.read_csv(input_path)
    required = {
        "feature_group",
        "heatmap_group",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "gradient",
        "gradient_label",
        "q",
        "standardized_slope",
        "standardized_slope_display",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")

    validate_standardized_slope_table(table)
    return table


def stand_change_label(label: object) -> str:
    text = str(label)
    return text if text.startswith("Δ") else f"Δ{text}"


def build_frequency_matrices(
    table: pd.DataFrame,
    spec: FrequencyHeatmapSpec,
    *,
    gradient: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_order = [metric.key for metric in spec.metrics]
    row_labels = [stand_change_label(metric.label) for metric in spec.metrics]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    label_by_metric = dict(zip(metric_order, row_labels, strict=True))

    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    subset = table[
        table["feature_group"].eq("frequency_metric")
        & table["heatmap_group"].eq(spec.heatmap_group)
        & table["gradient"].eq(gradient)
    ]
    for _, row in subset.iterrows():
        metric = str(row["metric"])
        band = str(row["band"])
        if metric not in label_by_metric or band not in BANDS:
            continue
        row_label = label_by_metric[metric]
        column_label = BAND_LABELS[band]
        value_matrix.loc[row_label, column_label] = row["standardized_slope_display"]
        q_matrix.loc[row_label, column_label] = row["q"]

    return value_matrix, q_matrix


def build_parameter_matrices(
    table: pd.DataFrame,
    spec: ParameterHeatmapSpec,
    *,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [gradient.label for gradient in GRADIENTS]
    column_labels = [stand_change_label(parameter.label) for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    label_by_parameter = {parameter.key: stand_change_label(parameter.label) for parameter in PARAMETERS}
    label_by_gradient = {gradient.key: gradient.label for gradient in GRADIENTS}

    subset = table[
        table["feature_group"].eq("parameter")
        & table["heatmap_group"].eq(spec.heatmap_group)
        & table["region"].eq(region)
    ]
    for _, row in subset.iterrows():
        gradient = str(row["gradient"])
        parameter = str(row["parameter"])
        if gradient not in label_by_gradient or parameter not in label_by_parameter:
            continue
        value_matrix.loc[label_by_gradient[gradient], label_by_parameter[parameter]] = row["standardized_slope_display"]
        q_matrix.loc[label_by_gradient[gradient], label_by_parameter[parameter]] = row["q"]

    return value_matrix, q_matrix


def plot_heatmap_grid_df(
    value_grid: list[list[pd.DataFrame]],
    q_grid: list[list[pd.DataFrame]],
    *,
    label_top: list[str],
    label_right: list[str] | None = None,
    label_fontsize: float = 7.0,
) -> plt.Figure:
    return vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=q_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=label_top,
        label_right=label_right,
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=CELL_SIZE_MM,
        panel_gap_mm=cfg.panel_gap,
        label_top_height_mm=cfg.strip_top_height_mm,
        strip_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_right_bg_color=cfg.label_right_bg_color,
        label_text_color=cfg.label_text_color,
        label_fontsize=label_fontsize,
        label_fontweight=cfg.label_fontweight,
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=8.0,
        grid_lines=None,
        gline_width=0.6,
        gline_alpha=0.9,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_ticks=[-VALUE_LIMIT, 0.0, VALUE_LIMIT],
        colorbar_width_mm=3.0,
        colorbar_pad_mm=2.0,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )


def plot_heatmap_grid(
    value_matrices: list[pd.DataFrame],
    q_matrices: list[pd.DataFrame],
    *,
    label_top: list[str],
    label_fontsize: float = 7.0,
) -> plt.Figure:
    return plot_heatmap_grid_df(
        [value_matrices],
        [q_matrices],
        label_top=label_top,
        label_fontsize=label_fontsize,
    )


def frequency_top_labels(spec: FrequencyHeatmapSpec) -> list[str]:
    return [
        rf"{spec.label_top}$_{{\mathrm{{{gradient.key}}}}}$"
        for gradient in GRADIENTS
    ]


def render_frequency_heatmap(spec: FrequencyHeatmapSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for gradient in GRADIENTS:
        value_matrix, q_matrix = build_frequency_matrices(table, spec, gradient=gradient.key)
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_heatmap_grid(value_matrices, q_matrices, label_top=frequency_top_labels(spec))
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_frequency_summary_heatmap(spec: FrequencySummaryHeatmapSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    value_grid: list[list[pd.DataFrame]] = []
    q_grid: list[list[pd.DataFrame]] = []
    for heatmap in FREQUENCY_SUMMARY_ROWS:
        value_row: list[pd.DataFrame] = []
        q_row: list[pd.DataFrame] = []
        for gradient in GRADIENTS:
            value_matrix, q_matrix = build_frequency_matrices(table, heatmap, gradient=gradient.key)
            value_row.append(value_matrix)
            q_row.append(q_matrix)
        value_grid.append(value_row)
        q_grid.append(q_row)

    fig = plot_heatmap_grid_df(
        value_grid,
        q_grid,
        label_top=[gradient.label for gradient in GRADIENTS],
        label_right=[heatmap.label_top for heatmap in FREQUENCY_SUMMARY_ROWS],
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_parameter_heatmap(spec: ParameterHeatmapSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for region in PARAMETER_REGIONS:
        value_matrix, q_matrix = build_parameter_matrices(table, spec, region=region)
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_heatmap_grid(
        value_matrices,
        q_matrices,
        label_top=list(PARAMETER_REGIONS),
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def run_workflow() -> dict[str, tuple[Path, Path]]:
    table = load_standardized_slope_table()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[Path, Path]] = {}
    for heatmap in FREQUENCY_HEATMAPS:
        outputs[heatmap.key] = render_frequency_heatmap(heatmap, table)
    for heatmap in FREQUENCY_SUMMARY_HEATMAPS:
        outputs[heatmap.key] = render_frequency_summary_heatmap(heatmap, table)
    for heatmap in PARAMETER_HEATMAPS:
        outputs[heatmap.key] = render_parameter_heatmap(heatmap, table)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[str, tuple[Path, Path]]) -> None:
    expected_keys = (
        {heatmap.key for heatmap in FREQUENCY_HEATMAPS}
        | {heatmap.key for heatmap in FREQUENCY_SUMMARY_HEATMAPS}
        | {heatmap.key for heatmap in PARAMETER_HEATMAPS}
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


if __name__ == "__main__":
    heatmap_outputs = run_workflow()
    for heatmap_name, paths in heatmap_outputs.items():
        print(f"{heatmap_name}: saved heatmap PDF: {paths[0]}")
        print(f"{heatmap_name}: saved heatmap PNG: {paths[1]}")
