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
from recipes.stnsnr.stats.walk.walk_task_dual_vs_single_signed_logp import (
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
    POSES,
    POSITIVE_DIRECTION_LABEL,
    SIGNED_LOGP_OUTPUT_PATH,
    SOURCE_CONTRAST,
    VALUE_LIMIT,
    GroupSpec,
)


COLORBAR_LABEL = r"Signed $-\log_{10}(p_{\mathrm{Tukey}})$"
COLORBAR_DIRECTION_X_OFFSET_MM = 1.0
COLORBAR_DIRECTION_Y_OFFSET_MM = 2.5
COLORBAR_DIRECTION_FONTSIZE = 6
CELL_SIZE_MM = (3.0, 3.0)
PARAMETER_CELL_SIZE_MM = CELL_SIZE_MM
GRID_LINE_COLOR = "black"
GRID_LINE_WIDTH = 0.6
GRID_LINE_ALPHA = 0.9
PANEL_ORDER = tuple(POSES)
HEATMAP_OUTPUT_DIR = OUTPUT_DIR / "task_dual_vs_single"


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
        pdf_path=HEATMAP_OUTPUT_DIR / "walk_task_stn_local_dual_vs_single_signed_logp_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR / "walk_task_stn_local_dual_vs_single_signed_logp_layered_heatmap.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=HEATMAP_OUTPUT_DIR / "walk_task_snr_local_dual_vs_single_signed_logp_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR / "walk_task_snr_local_dual_vs_single_signed_logp_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=HEATMAP_OUTPUT_DIR / "walk_task_connectivity_dual_vs_single_signed_logp_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR / "walk_task_connectivity_dual_vs_single_signed_logp_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=HEATMAP_OUTPUT_DIR / "walk_task_params_dual_vs_single_signed_logp_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR / "walk_task_params_dual_vs_single_signed_logp_layered_heatmap.png",
    ),
}


TRANSPOSED_HEATMAP_OUTPUTS = {
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        label_top="STN",
        pdf_path=HEATMAP_OUTPUT_DIR
        / "walk_task_stn_local_dual_vs_single_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR
        / "walk_task_stn_local_dual_vs_single_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=HEATMAP_OUTPUT_DIR
        / "walk_task_snr_local_dual_vs_single_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR
        / "walk_task_snr_local_dual_vs_single_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=HEATMAP_OUTPUT_DIR
        / "walk_task_connectivity_dual_vs_single_signed_logp_freq_by_param_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR
        / "walk_task_connectivity_dual_vs_single_signed_logp_freq_by_param_layered_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=HEATMAP_OUTPUT_DIR
        / "walk_task_params_dual_vs_single_signed_logp_transposed_layered_heatmap.pdf",
        png_path=HEATMAP_OUTPUT_DIR
        / "walk_task_params_dual_vs_single_signed_logp_transposed_layered_heatmap.png",
    ),
}


def load_signed_logp_table(input_path: str | Path = SIGNED_LOGP_OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP input table: {input_path}. "
            "Run recipes/stnsnr/stats/walk/walk_task_dual_vs_single_signed_logp.py first."
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
        "pose",
        "comparison",
        "positive_direction",
        "negative_direction",
        "source_task_first",
        "source_task_second",
        "plot_task_first",
        "plot_task_second",
        "source_model",
        "source_contrast",
        "source_csv",
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
        "group",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
        "pose",
        "comparison",
        "positive_direction",
        "negative_direction",
        "source_model",
        "source_contrast",
        "p_tukey",
        "signed_logp",
        "source_csv",
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
    if set(table["pose"].unique()) != set(PANEL_ORDER):
        raise ValueError(f"Signed-logP input poses must match {list(PANEL_ORDER)}.")

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
    pose: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [metric.label for metric in group.metrics]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, band_label in zip(BANDS, column_labels, strict=True):
            row = select_single_row(
                table,
                group=group.key,
                pose=pose,
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
    pose: str,
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
                pose=pose,
                metric="params",
                band="",
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[region, parameter.label] = clip_signed_logp(row["signed_logp"])
            p_matrix.loc[region, parameter.label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def clip_signed_logp(value: object) -> float:
    numeric_value = float(value)
    return float(np.clip(numeric_value, -VALUE_LIMIT, VALUE_LIMIT))


def select_single_row(
    table: pd.DataFrame,
    *,
    group: str,
    pose: str,
    metric: str,
    band: str,
    parameter: str,
    region: str,
) -> pd.Series:
    mask = (
        (table["group"] == group)
        & (table["pose"] == pose)
        & (table["metric"] == metric)
        & (table["band"] == band)
        & (table["parameter"] == parameter)
        & (table["region"] == region)
    )
    matches = table.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one signed-logP row for "
            f"group={group!r}, pose={pose!r}, metric={metric!r}, "
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
        dpi=cfg.dpi,
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
        show_y_ticklabels="all",
        transparent=True,
    )
    add_colorbar_end_labels(fig, fig.axes[-1])

    return fig


def add_colorbar_end_labels(
    fig: plt.Figure,
    colorbar_axis: plt.Axes,
    *,
    x_offset_mm: float = COLORBAR_DIRECTION_X_OFFSET_MM,
    y_offset_mm: float = COLORBAR_DIRECTION_Y_OFFSET_MM,
    fontsize: float = COLORBAR_DIRECTION_FONTSIZE,
) -> None:
    pos = colorbar_axis.get_position()
    fig_width_in, fig_height_in = fig.get_size_inches()
    x_offset = (x_offset_mm / 25.4) / fig_width_in
    y_offset = (y_offset_mm / 25.4) / fig_height_in
    x_label = pos.x1 + x_offset
    fig.text(
        x_label,
        pos.y1 + y_offset,
        POSITIVE_DIRECTION_LABEL,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )
    fig.text(
        x_label,
        pos.y0 - y_offset,
        NEGATIVE_DIRECTION_LABEL,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="black",
    )


def render_group_heatmap(
    group: GroupSpec,
    table: pd.DataFrame,
    *,
    transposed: bool = False,
) -> tuple[Path, Path]:
    output = TRANSPOSED_HEATMAP_OUTPUTS[group.key] if transposed else HEATMAP_OUTPUTS[group.key]
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for pose in PANEL_ORDER:
        value_matrix, p_matrix = build_group_matrices(group, table, pose=pose)
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
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def render_parameter_heatmap(table: pd.DataFrame, *, transposed: bool = False) -> tuple[Path, Path]:
    output = (
        TRANSPOSED_HEATMAP_OUTPUTS[PARAMETER_GROUP_KEY]
        if transposed
        else HEATMAP_OUTPUTS[PARAMETER_GROUP_KEY]
    )
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for pose in PANEL_ORDER:
        value_matrix, p_matrix = build_parameter_matrices(table, pose=pose)
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
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def transpose_heatmap_matrices(
    value_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return value_matrix.T.copy(), p_matrix.T.copy()


def run_workflow() -> dict[tuple[str, str], tuple[Path, Path]]:
    HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table(SIGNED_LOGP_OUTPUT_PATH)

    outputs: dict[tuple[str, str], tuple[Path, Path]] = {}
    for group in GROUPS:
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

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[tuple[str, str], tuple[Path, Path]]) -> None:
    expected_keys = {
        (group_key, "original")
        for group_key in HEATMAP_OUTPUTS
    } | {
        (group_key, "transposed")
        for group_key in TRANSPOSED_HEATMAP_OUTPUTS
    }
    if set(outputs) != expected_keys:
        raise ValueError(f"Expected output keys {sorted(expected_keys)}; got {sorted(outputs)}.")
    if PANEL_ORDER != POSES:
        raise ValueError(f"Panel order changed unexpectedly: {PANEL_ORDER}")

    missing = [
        str(path)
        for paths in outputs.values()
        for path in paths
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Expected heatmap files were not created: {missing}")

    empty_pngs = [
        str(paths[1])
        for paths in outputs.values()
        if not Path(paths[1]).exists() or Path(paths[1]).stat().st_size == 0
    ]
    if empty_pngs:
        raise ValueError(f"Expected non-empty PNG heatmaps; found empty files: {empty_pngs}")


if __name__ == "__main__":
    heatmap_outputs = run_workflow()
    for (group_name, orientation), paths in heatmap_outputs.items():
        print(f"{group_name} {orientation}: saved heatmap PDF: {paths[0]}")
        print(f"{group_name} {orientation}: saved heatmap PNG: {paths[1]}")
