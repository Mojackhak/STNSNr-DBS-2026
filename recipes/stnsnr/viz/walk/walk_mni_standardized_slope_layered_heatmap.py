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
from recipes.stnsnr.stats.walk.walk_mni_standardized_slope import (
    BAND_LABELS,
    BANDS,
    COORDINATES,
    METRICS,
    OUTPUT_DIR,
    OUTPUT_PATH,
    PARAMETERS,
    POSES,
    VALUE_LIMIT,
)


COLORBAR_LABEL = "Standardized slope"
CELL_SIZE_MM = (3.0, 3.0)
PANEL_ORDER = tuple(POSES)


@dataclass(frozen=True)
class HeatmapSpec:
    key: str
    region: str
    coordinate: str
    label_top: str
    pdf_path: Path
    png_path: Path


@dataclass(frozen=True)
class ParameterHeatmapSpec:
    key: str
    region: str
    label_top: str
    pdf_path: Path
    png_path: Path


@dataclass(frozen=True)
class CombinedHeatmapSpec:
    key: str
    region: str
    pdf_path: Path
    png_path: Path


@dataclass(frozen=True)
class CombinedParameterHeatmapSpec:
    key: str
    pdf_path: Path
    png_path: Path


HEATMAPS = (
    HeatmapSpec(
        key="stn_mni_x",
        region="STN",
        coordinate="x",
        label_top="zSTN-x",
        pdf_path=OUTPUT_DIR / "walk_stn_mni_x_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_stn_mni_x_standardized_slope_layered_heatmap.png",
    ),
    HeatmapSpec(
        key="stn_mni_y",
        region="STN",
        coordinate="y",
        label_top="zSTN-y",
        pdf_path=OUTPUT_DIR / "walk_stn_mni_y_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_stn_mni_y_standardized_slope_layered_heatmap.png",
    ),
    HeatmapSpec(
        key="stn_mni_z",
        region="STN",
        coordinate="z",
        label_top="zSTN-z",
        pdf_path=OUTPUT_DIR / "walk_stn_mni_z_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_stn_mni_z_standardized_slope_layered_heatmap.png",
    ),
    HeatmapSpec(
        key="snr_mni_x",
        region="SNr",
        coordinate="x",
        label_top="zSNr-x",
        pdf_path=OUTPUT_DIR / "walk_snr_mni_x_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_snr_mni_x_standardized_slope_layered_heatmap.png",
    ),
    HeatmapSpec(
        key="snr_mni_y",
        region="SNr",
        coordinate="y",
        label_top="zSNr-y",
        pdf_path=OUTPUT_DIR / "walk_snr_mni_y_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_snr_mni_y_standardized_slope_layered_heatmap.png",
    ),
    HeatmapSpec(
        key="snr_mni_z",
        region="SNr",
        coordinate="z",
        label_top="zSNr-z",
        pdf_path=OUTPUT_DIR / "walk_snr_mni_z_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_snr_mni_z_standardized_slope_layered_heatmap.png",
    ),
)

PARAMETER_HEATMAPS = (
    ParameterHeatmapSpec(
        key="stn_params_mni",
        region="STN",
        label_top="STN",
        pdf_path=OUTPUT_DIR / "walk_stn_params_mni_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_stn_params_mni_standardized_slope_layered_heatmap.png",
    ),
    ParameterHeatmapSpec(
        key="snr_params_mni",
        region="SNr",
        label_top="SNr",
        pdf_path=OUTPUT_DIR / "walk_snr_params_mni_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_snr_params_mni_standardized_slope_layered_heatmap.png",
    ),
)

COMBINED_HEATMAPS = (
    CombinedHeatmapSpec(
        key="stn_mni_xyz",
        region="STN",
        pdf_path=OUTPUT_DIR / "walk_stn_mni_xyz_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_stn_mni_xyz_standardized_slope_layered_heatmap.png",
    ),
    CombinedHeatmapSpec(
        key="snr_mni_xyz",
        region="SNr",
        pdf_path=OUTPUT_DIR / "walk_snr_mni_xyz_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_snr_mni_xyz_standardized_slope_layered_heatmap.png",
    ),
)

COMBINED_PARAMETER_HEATMAPS = (
    CombinedParameterHeatmapSpec(
        key="stn_snr_params_mni",
        pdf_path=OUTPUT_DIR / "walk_stn_snr_params_mni_standardized_slope_layered_heatmap.pdf",
        png_path=OUTPUT_DIR / "walk_stn_snr_params_mni_standardized_slope_layered_heatmap.png",
    ),
)


def load_standardized_slope_table(path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing standardized slope table: {input_path}. "
            "Run recipes/stnsnr/stats/walk/walk_mni_standardized_slope.py first."
        )

    table = pd.read_csv(input_path)
    required = {
        "feature_group",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "pose",
        "coordinate",
        "coordinate_label",
        "q",
        "standardized_slope",
        "standardized_slope_display",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
    return table


def walk_change_label(label: object) -> str:
    text = str(label)
    return text if text.startswith("Δ") else f"Δ{text}"


def coordinate_display_label(label: object) -> str:
    text = str(label)
    return text.replace("(scored)", "(z-scored)")


def build_heatmap_matrices_by_pose(
    table: pd.DataFrame,
    spec: HeatmapSpec,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for pose in PANEL_ORDER:
        value_matrix, q_matrix = build_heatmap_matrices(table, spec, pose=pose)
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)
    return value_matrices, q_matrices


def build_heatmap_matrices(
    table: pd.DataFrame,
    spec: HeatmapSpec,
    *,
    pose: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_order = [metric.key for metric in METRICS]
    label_by_metric = (
        table.drop_duplicates("metric")
        .set_index("metric")["metric_label"]
        .reindex(metric_order)
        .to_dict()
    )
    row_labels = [walk_change_label(label_by_metric[metric]) for metric in metric_order]
    column_labels = [BAND_LABELS[band] for band in BANDS]

    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    subset = table[
        table["feature_group"].eq("frequency_metric")
        & table["region"].eq(spec.region)
        & table["coordinate"].eq(spec.coordinate)
        & table["pose"].eq(pose)
    ]
    for _, row in subset.iterrows():
        metric = str(row["metric"])
        band = str(row["band"])
        if metric not in metric_order or band not in BANDS:
            continue
        row_label = walk_change_label(label_by_metric[metric])
        column_label = BAND_LABELS[band]
        value_matrix.loc[row_label, column_label] = row["standardized_slope_display"]
        q_matrix.loc[row_label, column_label] = row["q"]

    return value_matrix, q_matrix


def build_parameter_heatmap_matrices_by_pose(
    table: pd.DataFrame,
    spec: ParameterHeatmapSpec,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    value_matrices: list[pd.DataFrame] = []
    q_matrices: list[pd.DataFrame] = []
    for pose in PANEL_ORDER:
        value_matrix, q_matrix = build_parameter_heatmap_matrices(table, spec, pose=pose)
        value_matrices.append(value_matrix)
        q_matrices.append(q_matrix)
    return value_matrices, q_matrices


def build_parameter_heatmap_matrices(
    table: pd.DataFrame,
    spec: ParameterHeatmapSpec,
    *,
    pose: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [coordinate_display_label(coordinate.label) for coordinate in COORDINATES]
    column_labels = [walk_change_label(parameter.label) for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    coordinate_by_key = {item.key: item for item in COORDINATES}
    parameter_by_key = {item.key: item for item in PARAMETERS}
    subset = table[
        table["feature_group"].eq("parameter")
        & table["region"].eq(spec.region)
        & table["pose"].eq(pose)
    ]
    for _, row in subset.iterrows():
        coordinate = str(row["coordinate"])
        parameter = str(row["parameter"])
        if coordinate not in coordinate_by_key or parameter not in parameter_by_key:
            continue
        coordinate_label = coordinate_display_label(coordinate_by_key[coordinate].label)
        parameter_label = walk_change_label(parameter_by_key[parameter].label)
        value_matrix.loc[coordinate_label, parameter_label] = row["standardized_slope_display"]
        q_matrix.loc[coordinate_label, parameter_label] = row["q"]

    return value_matrix, q_matrix


def plot_layered_heatmap(
    value_matrices: list[pd.DataFrame],
    q_matrices: list[pd.DataFrame],
    *,
    label_top: str,
    cellsize_mm: tuple[float, float] = CELL_SIZE_MM,
) -> plt.Figure:
    if len(value_matrices) != len(PANEL_ORDER):
        raise ValueError(f"Expected {len(PANEL_ORDER)} value matrices; got {len(value_matrices)}.")
    if len(q_matrices) != len(PANEL_ORDER):
        raise ValueError(f"Expected {len(PANEL_ORDER)} q-value matrices; got {len(q_matrices)}.")
    if not value_matrices:
        raise ValueError("At least one pose panel is required.")

    return plot_heatmap_grid(
        df_values=[[value_matrix] for value_matrix in value_matrices],
        df_ps=[[q_matrix] for q_matrix in q_matrices],
        label_top=[label_top],
        label_right=list(PANEL_ORDER),
        show_y_ticklabels="all",
        panel_gap_mm=(0.0, cfg.panel_gap[1]),
        cellsize_mm=cellsize_mm,
    )


def plot_heatmap_grid(
    *,
    df_values: list[list[pd.DataFrame]],
    df_ps: list[list[pd.DataFrame]],
    label_top: list[str],
    label_right: list[str],
    show_y_ticklabels: str,
    panel_gap_mm: tuple[float, float],
    cellsize_mm: tuple[float, float] = CELL_SIZE_MM,
) -> plt.Figure:
    return vdf.plot_well_heatmap_grid_df(
        df_values=df_values,
        df_ps=df_ps,
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
        cellsize=cellsize_mm,
        panel_gap_mm=panel_gap_mm,
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
        show_y_ticklabels=show_y_ticklabels,
        transparent=True,
    )


def validate_input_table(table: pd.DataFrame) -> None:
    expected_frequency_rows = 2 * 3 * len(PANEL_ORDER) * len(METRICS) * len(BANDS)
    expected_parameter_rows = 2 * 3 * len(PANEL_ORDER) * len(PARAMETERS)
    expected_rows = expected_frequency_rows + expected_parameter_rows
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(table)}")

    frequency_rows = table[table["feature_group"].eq("frequency_metric")]
    if len(frequency_rows) != expected_frequency_rows:
        raise ValueError(f"Expected {expected_frequency_rows} frequency rows, got {len(frequency_rows)}")

    parameter_rows = table[table["feature_group"].eq("parameter")]
    if len(parameter_rows) != expected_parameter_rows:
        raise ValueError(f"Expected {expected_parameter_rows} parameter rows, got {len(parameter_rows)}")

    if set(table["pose"].dropna()) != set(PANEL_ORDER):
        raise ValueError(f"Unexpected pose values: {sorted(table['pose'].dropna().unique())}")

    if table["value_sd"].isna().any() or (table["value_sd"] <= 0).any():
        bad = table[table["value_sd"].isna() | (table["value_sd"] <= 0)]
        raise ValueError(f"Found missing or non-positive value_sd rows: {bad.head().to_dict('records')}")

    if table["standardized_slope"].isna().any():
        bad = table[table["standardized_slope"].isna()]
        raise ValueError(f"Found missing standardized_slope rows: {bad.head().to_dict('records')}")


def render_heatmap(spec: HeatmapSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    value_matrices, q_matrices = build_heatmap_matrices_by_pose(table, spec)
    fig = plot_layered_heatmap(
        value_matrices,
        q_matrices,
        label_top=spec.label_top,
        cellsize_mm=CELL_SIZE_MM,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_combined_heatmap(spec: CombinedHeatmapSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    spec_by_coordinate = {
        heatmap.coordinate: heatmap
        for heatmap in HEATMAPS
        if heatmap.region == spec.region
    }
    missing_coordinates = [coordinate.key for coordinate in COORDINATES if coordinate.key not in spec_by_coordinate]
    if missing_coordinates:
        raise ValueError(f"Missing heatmap specs for {spec.region} coordinates: {missing_coordinates}")

    df_values: list[list[pd.DataFrame]] = []
    df_ps: list[list[pd.DataFrame]] = []
    for pose in PANEL_ORDER:
        value_row: list[pd.DataFrame] = []
        q_row: list[pd.DataFrame] = []
        for coordinate in COORDINATES:
            coordinate_spec = spec_by_coordinate[coordinate.key]
            value_matrix, q_matrix = build_heatmap_matrices(table, coordinate_spec, pose=pose)
            value_row.append(value_matrix)
            q_row.append(q_matrix)
        df_values.append(value_row)
        df_ps.append(q_row)

    fig = plot_heatmap_grid(
        df_values=df_values,
        df_ps=df_ps,
        label_top=[spec_by_coordinate[coordinate.key].label_top for coordinate in COORDINATES],
        label_right=list(PANEL_ORDER),
        show_y_ticklabels="left",
        panel_gap_mm=(cfg.panel_gap[1], cfg.panel_gap[1]),
        cellsize_mm=CELL_SIZE_MM,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_parameter_heatmap(spec: ParameterHeatmapSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    value_matrices, q_matrices = build_parameter_heatmap_matrices_by_pose(table, spec)
    fig = plot_layered_heatmap(
        value_matrices,
        q_matrices,
        label_top=spec.label_top,
        cellsize_mm=CELL_SIZE_MM,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def render_combined_parameter_heatmap(
    spec: CombinedParameterHeatmapSpec,
    table: pd.DataFrame,
) -> tuple[Path, Path]:
    df_values: list[list[pd.DataFrame]] = []
    df_ps: list[list[pd.DataFrame]] = []
    for pose in PANEL_ORDER:
        value_row: list[pd.DataFrame] = []
        q_row: list[pd.DataFrame] = []
        for parameter_spec in PARAMETER_HEATMAPS:
            value_matrix, q_matrix = build_parameter_heatmap_matrices(table, parameter_spec, pose=pose)
            value_row.append(value_matrix)
            q_row.append(q_matrix)
        df_values.append(value_row)
        df_ps.append(q_row)

    fig = plot_heatmap_grid(
        df_values=df_values,
        df_ps=df_ps,
        label_top=[parameter_spec.label_top for parameter_spec in PARAMETER_HEATMAPS],
        label_right=list(PANEL_ORDER),
        show_y_ticklabels="left",
        panel_gap_mm=(cfg.panel_gap[1], cfg.panel_gap[1]),
        cellsize_mm=CELL_SIZE_MM,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return spec.pdf_path, spec.png_path


def run_workflow() -> dict[str, tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_standardized_slope_table()
    validate_input_table(table)

    outputs: dict[str, tuple[Path, Path]] = {}
    for heatmap in HEATMAPS:
        outputs[heatmap.key] = render_heatmap(heatmap, table)
    for heatmap in COMBINED_HEATMAPS:
        outputs[heatmap.key] = render_combined_heatmap(heatmap, table)
    for heatmap in PARAMETER_HEATMAPS:
        outputs[heatmap.key] = render_parameter_heatmap(heatmap, table)
    for heatmap in COMBINED_PARAMETER_HEATMAPS:
        outputs[heatmap.key] = render_combined_parameter_heatmap(heatmap, table)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[str, tuple[Path, Path]]) -> None:
    expected_keys = (
        {heatmap.key for heatmap in HEATMAPS}
        | {heatmap.key for heatmap in COMBINED_HEATMAPS}
        | {heatmap.key for heatmap in PARAMETER_HEATMAPS}
        | {heatmap.key for heatmap in COMBINED_PARAMETER_HEATMAPS}
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
