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
from recipes.stnsnr.stats.gait_feature.gait_feature_gradient_standardized_slope import (
    BAND_LABELS,
    BANDS,
    CONNECTIVITY_METRICS,
    DATASETS,
    GAIT_FEATURES,
    GRADIENTS,
    PARAMETER_REGIONS,
    PARAMETERS,
    VALUE_LIMIT,
    DatasetSpec,
    GaitFeatureSpec,
    MetricSpec,
    validate_standardized_slope_table,
)


CELL_SIZE_MM = (3.0, 3.0)
COLORBAR_TICKS = [-0.6, -0.3, 0.0, 0.3, 0.6]


@dataclass(frozen=True)
class FrequencyHeatmapSpec:
    key: str
    heatmap_group: str
    label_top: str
    metrics: tuple[MetricSpec, ...]
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


def output_base(dataset: DatasetSpec, feature: GaitFeatureSpec, heatmap_key: str) -> Path:
    return dataset.output_dir / f"{dataset.key}_{feature.slug}_{heatmap_key}_gradient_standardized_slope_heatmap"


def frequency_heatmaps(dataset: DatasetSpec, feature: GaitFeatureSpec) -> tuple[FrequencyHeatmapSpec, ...]:
    return (
        FrequencyHeatmapSpec(
            key="stn_local",
            heatmap_group="stn_local",
            label_top="STN",
            metrics=dataset.local_metrics,
            pdf_path=output_base(dataset, feature, "stn_local").with_suffix(".pdf"),
            png_path=output_base(dataset, feature, "stn_local").with_suffix(".png"),
        ),
        FrequencyHeatmapSpec(
            key="snr_local",
            heatmap_group="snr_local",
            label_top="SNr",
            metrics=dataset.local_metrics,
            pdf_path=output_base(dataset, feature, "snr_local").with_suffix(".pdf"),
            png_path=output_base(dataset, feature, "snr_local").with_suffix(".png"),
        ),
        FrequencyHeatmapSpec(
            key="connectivity",
            heatmap_group="connectivity",
            label_top="SNr-/→STN",
            metrics=CONNECTIVITY_METRICS,
            pdf_path=output_base(dataset, feature, "connectivity").with_suffix(".pdf"),
            png_path=output_base(dataset, feature, "connectivity").with_suffix(".png"),
        ),
    )


def parameter_heatmap(dataset: DatasetSpec, feature: GaitFeatureSpec) -> ParameterHeatmapSpec:
    return ParameterHeatmapSpec(
        key="params",
        heatmap_group="params",
        pdf_path=output_base(dataset, feature, "params").with_suffix(".pdf"),
        png_path=output_base(dataset, feature, "params").with_suffix(".png"),
    )


def parameter_by_gradient_heatmap(dataset: DatasetSpec, feature: GaitFeatureSpec) -> ParameterHeatmapSpec:
    return ParameterHeatmapSpec(
        key="params_by_gradient",
        heatmap_group="params",
        pdf_path=output_base(dataset, feature, "params_by_gradient").with_suffix(".pdf"),
        png_path=output_base(dataset, feature, "params_by_gradient").with_suffix(".png"),
    )


def parameter_by_region_heatmap(dataset: DatasetSpec, feature: GaitFeatureSpec) -> ParameterHeatmapSpec:
    return ParameterHeatmapSpec(
        key="params_by_region",
        heatmap_group="params",
        pdf_path=output_base(dataset, feature, "params_by_region").with_suffix(".pdf"),
        png_path=output_base(dataset, feature, "params_by_region").with_suffix(".png"),
    )


def frequency_summary_heatmap(dataset: DatasetSpec, feature: GaitFeatureSpec) -> FrequencySummaryHeatmapSpec:
    return FrequencySummaryHeatmapSpec(
        key="frequency_summary",
        pdf_path=output_base(dataset, feature, "frequency_summary").with_suffix(".pdf"),
        png_path=output_base(dataset, feature, "frequency_summary").with_suffix(".png"),
    )


def frequency_summary_transposed_heatmap(dataset: DatasetSpec, feature: GaitFeatureSpec) -> FrequencySummaryHeatmapSpec:
    return FrequencySummaryHeatmapSpec(
        key="frequency_summary_transposed",
        pdf_path=output_base(dataset, feature, "frequency_summary_transposed").with_suffix(".pdf"),
        png_path=output_base(dataset, feature, "frequency_summary_transposed").with_suffix(".png"),
    )


def load_standardized_slope_table(dataset: DatasetSpec) -> pd.DataFrame:
    if not dataset.output_path.exists():
        raise FileNotFoundError(
            f"Missing standardized slope table: {dataset.output_path}. "
            "Run recipes/stnsnr/stats/gait_feature/gait_feature_gradient_standardized_slope.py first."
        )

    table = pd.read_csv(dataset.output_path)
    required = {
        "dataset",
        "feature_group",
        "heatmap_group",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "feature",
        "feature_label",
        "gradient",
        "gradient_label",
        "q",
        "standardized_slope",
        "standardized_slope_display",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{dataset.output_path} is missing columns: {sorted(missing)}")

    validate_standardized_slope_table(table, dataset)
    return table


def stand_change_label(label: object) -> str:
    text = str(label)
    return text if text.startswith("Δ") else f"Δ{text}"


def colorbar_label(feature: GaitFeatureSpec) -> str:
    return f"Standardized slope of ΔLFP (Walk - Stand) per z-scored {feature.label}"


def build_frequency_matrices(
    table: pd.DataFrame,
    spec: FrequencyHeatmapSpec,
    *,
    feature: GaitFeatureSpec,
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
        & table["feature"].eq(feature.key)
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
    feature: GaitFeatureSpec,
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
        & table["feature"].eq(feature.key)
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


def build_parameter_by_gradient_matrices(
    table: pd.DataFrame,
    spec: ParameterHeatmapSpec,
    *,
    feature: GaitFeatureSpec,
    gradient: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = list(PARAMETER_REGIONS)
    column_labels = [stand_change_label(parameter.label) for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    label_by_parameter = {parameter.key: stand_change_label(parameter.label) for parameter in PARAMETERS}

    subset = table[
        table["feature_group"].eq("parameter")
        & table["heatmap_group"].eq(spec.heatmap_group)
        & table["feature"].eq(feature.key)
        & table["gradient"].eq(gradient)
    ]
    for _, row in subset.iterrows():
        region = str(row["region"])
        parameter = str(row["parameter"])
        if region not in PARAMETER_REGIONS or parameter not in label_by_parameter:
            continue
        value_matrix.loc[region, label_by_parameter[parameter]] = row["standardized_slope_display"]
        q_matrix.loc[region, label_by_parameter[parameter]] = row["q"]

    return value_matrix, q_matrix


def build_parameter_by_region_matrices(
    table: pd.DataFrame,
    spec: ParameterHeatmapSpec,
    *,
    feature: GaitFeatureSpec,
    gradient: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [stand_change_label(parameter.label) for parameter in PARAMETERS]
    column_labels = list(PARAMETER_REGIONS)
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    label_by_parameter = {parameter.key: stand_change_label(parameter.label) for parameter in PARAMETERS}

    subset = table[
        table["feature_group"].eq("parameter")
        & table["heatmap_group"].eq(spec.heatmap_group)
        & table["feature"].eq(feature.key)
        & table["gradient"].eq(gradient)
    ]
    for _, row in subset.iterrows():
        region = str(row["region"])
        parameter = str(row["parameter"])
        if region not in PARAMETER_REGIONS or parameter not in label_by_parameter:
            continue
        value_matrix.loc[label_by_parameter[parameter], region] = row["standardized_slope_display"]
        q_matrix.loc[label_by_parameter[parameter], region] = row["q"]

    return value_matrix, q_matrix


def plot_heatmap_grid_df(
    value_grid: list[list[pd.DataFrame]],
    q_grid: list[list[pd.DataFrame]],
    *,
    feature: GaitFeatureSpec,
    label_top: list[str],
    label_right: list[str] | None = None,
    label_fontsize: float = 7.0,
    colorbar_length_mm: float | None = None,
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
        colorbar_label=colorbar_label(feature),
        colorbar_ticks=[COLORBAR_TICKS[0], 0.0, COLORBAR_TICKS[-1]] if value_grid[0][0].shape[0] == 2 else COLORBAR_TICKS,
        colorbar_width_mm=3.0,
        colorbar_pad_mm=2.0,
        colorbar_length_mm=colorbar_length_mm,
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
    feature: GaitFeatureSpec,
    label_top: list[str],
    label_fontsize: float = 7.0,
    colorbar_length_mm: float | None = None,
) -> plt.Figure:
    return plot_heatmap_grid_df(
        [value_matrices],
        [q_matrices],
        feature=feature,
        label_top=label_top,
        label_fontsize=label_fontsize,
        colorbar_length_mm=colorbar_length_mm,
    )


def frequency_top_labels(spec: FrequencyHeatmapSpec) -> list[str]:
    return [
        rf"{spec.label_top}$_{{\mathrm{{{gradient.key}}}}}$"
        for gradient in GRADIENTS
    ]


def render_frequency_heatmap(
    spec: FrequencyHeatmapSpec,
    table: pd.DataFrame,
    *,
    feature: GaitFeatureSpec,
) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for gradient in GRADIENTS:
        value_matrix, q_matrix = build_frequency_matrices(
            table,
            spec,
            feature=feature,
            gradient=gradient.key,
        )
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_heatmap_grid(
        value_matrices,
        q_matrices,
        feature=feature,
        label_top=frequency_top_labels(spec),
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_frequency_summary_heatmap(
    spec: FrequencySummaryHeatmapSpec,
    frequency_specs: tuple[FrequencyHeatmapSpec, ...],
    table: pd.DataFrame,
    *,
    feature: GaitFeatureSpec,
) -> tuple[Path, Path]:
    rows = (frequency_specs[1], frequency_specs[0], frequency_specs[2])
    value_grid: list[list[pd.DataFrame]] = []
    q_grid: list[list[pd.DataFrame]] = []
    for heatmap in rows:
        value_row: list[pd.DataFrame] = []
        q_row: list[pd.DataFrame] = []
        for gradient in GRADIENTS:
            value_matrix, q_matrix = build_frequency_matrices(
                table,
                heatmap,
                feature=feature,
                gradient=gradient.key,
            )
            value_row.append(value_matrix)
            q_row.append(q_matrix)
        value_grid.append(value_row)
        q_grid.append(q_row)

    fig = plot_heatmap_grid_df(
        value_grid,
        q_grid,
        feature=feature,
        label_top=[gradient.label for gradient in GRADIENTS],
        label_right=[heatmap.label_top for heatmap in rows],
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_frequency_summary_transposed_heatmap(
    spec: FrequencySummaryHeatmapSpec,
    frequency_specs: tuple[FrequencyHeatmapSpec, ...],
    table: pd.DataFrame,
    *,
    feature: GaitFeatureSpec,
) -> tuple[Path, Path]:
    columns = (frequency_specs[1], frequency_specs[0], frequency_specs[2])
    value_grid: list[list[pd.DataFrame]] = []
    q_grid: list[list[pd.DataFrame]] = []
    for gradient in GRADIENTS:
        value_row: list[pd.DataFrame] = []
        q_row: list[pd.DataFrame] = []
        for heatmap in columns:
            value_matrix, q_matrix = build_frequency_matrices(
                table,
                heatmap,
                feature=feature,
                gradient=gradient.key,
            )
            value_row.append(value_matrix.T)
            q_row.append(q_matrix.T)
        value_grid.append(value_row)
        q_grid.append(q_row)

    fig = plot_heatmap_grid_df(
        value_grid,
        q_grid,
        feature=feature,
        label_top=[heatmap.label_top for heatmap in columns],
        label_right=[gradient.label for gradient in GRADIENTS],
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_parameter_heatmap(
    spec: ParameterHeatmapSpec,
    table: pd.DataFrame,
    *,
    feature: GaitFeatureSpec,
) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for region in PARAMETER_REGIONS:
        value_matrix, q_matrix = build_parameter_matrices(table, spec, feature=feature, region=region)
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_heatmap_grid(
        value_matrices,
        q_matrices,
        feature=feature,
        label_top=list(PARAMETER_REGIONS),
        colorbar_length_mm=18.0,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_parameter_by_gradient_heatmap(
    spec: ParameterHeatmapSpec,
    table: pd.DataFrame,
    *,
    feature: GaitFeatureSpec,
) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for gradient in GRADIENTS:
        value_matrix, q_matrix = build_parameter_by_gradient_matrices(
            table,
            spec,
            feature=feature,
            gradient=gradient.key,
        )
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_heatmap_grid(
        value_matrices,
        q_matrices,
        feature=feature,
        label_top=[gradient.label for gradient in GRADIENTS],
        label_fontsize=5.0,
        colorbar_length_mm=18.0,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_parameter_by_region_heatmap(
    spec: ParameterHeatmapSpec,
    table: pd.DataFrame,
    *,
    feature: GaitFeatureSpec,
) -> tuple[Path, Path]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for gradient in GRADIENTS:
        value_matrix, q_matrix = build_parameter_by_region_matrices(
            table,
            spec,
            feature=feature,
            gradient=gradient.key,
        )
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)

    fig = plot_heatmap_grid(
        value_matrices,
        q_matrices,
        feature=feature,
        label_top=[gradient.label for gradient in GRADIENTS],
        label_fontsize=5.0,
        colorbar_length_mm=18.0,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def run_workflow() -> dict[tuple[str, str, str], tuple[Path, Path]]:
    outputs: dict[tuple[str, str, str], tuple[Path, Path]] = {}
    for dataset in DATASETS:
        table = load_standardized_slope_table(dataset)
        dataset.output_dir.mkdir(parents=True, exist_ok=True)
        for feature in GAIT_FEATURES:
            frequency_specs = frequency_heatmaps(dataset, feature)
            for heatmap in frequency_specs:
                outputs[(dataset.key, feature.key, heatmap.key)] = render_frequency_heatmap(
                    heatmap,
                    table,
                    feature=feature,
                )

            summary_spec = frequency_summary_heatmap(dataset, feature)
            outputs[(dataset.key, feature.key, summary_spec.key)] = render_frequency_summary_heatmap(
                summary_spec,
                frequency_specs,
                table,
                feature=feature,
            )

            transposed_summary_spec = frequency_summary_transposed_heatmap(dataset, feature)
            outputs[(dataset.key, feature.key, transposed_summary_spec.key)] = (
                render_frequency_summary_transposed_heatmap(
                    transposed_summary_spec,
                    frequency_specs,
                    table,
                    feature=feature,
                )
            )

            params_spec = parameter_heatmap(dataset, feature)
            outputs[(dataset.key, feature.key, params_spec.key)] = render_parameter_heatmap(
                params_spec,
                table,
                feature=feature,
            )

            params_by_gradient_spec = parameter_by_gradient_heatmap(dataset, feature)
            outputs[(dataset.key, feature.key, params_by_gradient_spec.key)] = render_parameter_by_gradient_heatmap(
                params_by_gradient_spec,
                table,
                feature=feature,
            )

            params_by_region_spec = parameter_by_region_heatmap(dataset, feature)
            outputs[(dataset.key, feature.key, params_by_region_spec.key)] = render_parameter_by_region_heatmap(
                params_by_region_spec,
                table,
                feature=feature,
            )

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[tuple[str, str, str], tuple[Path, Path]]) -> None:
    expected_keys = {
        (dataset.key, feature.key, heatmap_key)
        for dataset in DATASETS
        for feature in GAIT_FEATURES
        for heatmap_key in (
            "stn_local",
            "snr_local",
            "connectivity",
            "frequency_summary",
            "frequency_summary_transposed",
            "params",
            "params_by_gradient",
            "params_by_region",
        )
    }
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
    for (dataset_key, feature_key, heatmap_name), paths in heatmap_outputs.items():
        print(f"{dataset_key}/{feature_key}/{heatmap_name}: saved heatmap PDF: {paths[0]}")
        print(f"{dataset_key}/{feature_key}/{heatmap_name}: saved heatmap PNG: {paths[1]}")
