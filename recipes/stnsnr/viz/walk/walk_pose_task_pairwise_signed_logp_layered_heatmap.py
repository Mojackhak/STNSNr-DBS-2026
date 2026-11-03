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
from recipes.stnsnr.stats.walk.walk_pose_task_pairwise_signed_logp import (
    BAND_LABELS,
    BANDS,
    DISPLAY_COMPARISONS,
    EXPECTED_ROW_COUNT,
    GROUPS,
    MODEL_DIR,
    OUTPUT_DIR,
    PARAMETERS,
    PARAMETER_GROUP_KEY,
    PARAMETER_LABEL_TOP,
    PARAMETER_REGIONS,
    SIGNED_LOGP_OUTPUT_PATH,
    SOURCE_CONTRASTS,
    TASKS,
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
HEATMAP_OUTPUT_DIR = OUTPUT_DIR / "pose_pairwise"


@dataclass(frozen=True)
class HeatmapOutputSpec:
    key: str
    task: str
    label_top: str
    pdf_path: Path
    png_path: Path


def task_filename_label(task: str) -> str:
    return task.lower()


def build_heatmap_outputs() -> dict[tuple[str, str], HeatmapOutputSpec]:
    outputs: dict[tuple[str, str], HeatmapOutputSpec] = {}
    for group in GROUPS:
        for task in TASKS:
            basename = (
                f"walk_pose_task_pairwise_{group.key}_{task_filename_label(task)}"
                "_signed_logp_layered_heatmap"
            )
            outputs[(group.key, task)] = HeatmapOutputSpec(
                key=group.key,
                task=task,
                label_top=f"{group.label_top} | {task}",
                pdf_path=HEATMAP_OUTPUT_DIR / f"{basename}.pdf",
                png_path=HEATMAP_OUTPUT_DIR / f"{basename}.png",
            )

    for task in TASKS:
        basename = (
            f"walk_pose_task_pairwise_params_{task_filename_label(task)}"
            "_signed_logp_layered_heatmap"
        )
        outputs[(PARAMETER_GROUP_KEY, task)] = HeatmapOutputSpec(
            key=PARAMETER_GROUP_KEY,
            task=task,
            label_top=f"{PARAMETER_LABEL_TOP} | {task}",
            pdf_path=HEATMAP_OUTPUT_DIR / f"{basename}.pdf",
            png_path=HEATMAP_OUTPUT_DIR / f"{basename}.png",
        )
    return outputs


HEATMAP_OUTPUTS = build_heatmap_outputs()


def build_transposed_heatmap_outputs() -> dict[tuple[str, str], HeatmapOutputSpec]:
    outputs: dict[tuple[str, str], HeatmapOutputSpec] = {}
    for group in GROUPS:
        for task in TASKS:
            basename = (
                f"walk_pose_task_pairwise_{group.key}_{task_filename_label(task)}"
                "_signed_logp_freq_by_param_layered_heatmap"
            )
            outputs[(group.key, task)] = HeatmapOutputSpec(
                key=group.key,
                task=task,
                label_top=f"{group.label_top} | {task}",
                pdf_path=HEATMAP_OUTPUT_DIR / f"{basename}.pdf",
                png_path=HEATMAP_OUTPUT_DIR / f"{basename}.png",
            )

    for task in TASKS:
        basename = (
            f"walk_pose_task_pairwise_params_{task_filename_label(task)}"
            "_signed_logp_transposed_layered_heatmap"
        )
        outputs[(PARAMETER_GROUP_KEY, task)] = HeatmapOutputSpec(
            key=PARAMETER_GROUP_KEY,
            task=task,
            label_top=f"{PARAMETER_LABEL_TOP} | {task}",
            pdf_path=HEATMAP_OUTPUT_DIR / f"{basename}.pdf",
            png_path=HEATMAP_OUTPUT_DIR / f"{basename}.png",
        )
    return outputs


TRANSPOSED_HEATMAP_OUTPUTS = build_transposed_heatmap_outputs()


def load_signed_logp_table(input_path: str | Path = SIGNED_LOGP_OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP input table: {input_path}. "
            "Run recipes/stnsnr/stats/walk/walk_pose_task_pairwise_signed_logp.py first."
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
        "task",
        "source_pose_first",
        "source_pose_second",
        "plot_pose_first",
        "plot_pose_second",
        "source_model",
        "source_contrast",
        "comparison",
        "estimate_label",
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
        "task",
        "source_model",
        "source_contrast",
        "comparison",
        "p_tukey",
        "signed_logp",
        "source_csv",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Signed-logP input is missing columns: {missing_columns}")

    if (table["source_model"] != MODEL_DIR).any():
        raise ValueError(f"Signed-logP input contains rows outside {MODEL_DIR}.")
    if (~table["task"].isin(TASKS)).any():
        bad_tasks = table.loc[~table["task"].isin(TASKS), "task"].unique().tolist()
        raise ValueError(f"Signed-logP input contains unexpected tasks: {bad_tasks}")
    if (~table["source_contrast"].isin(SOURCE_CONTRASTS)).any():
        bad_contrasts = table.loc[
            ~table["source_contrast"].isin(SOURCE_CONTRASTS),
            "source_contrast",
        ].unique().tolist()
        raise ValueError(f"Signed-logP input contains unexpected source contrasts: {bad_contrasts}")
    if set(table["comparison"].unique()) != set(PANEL_ORDER):
        raise ValueError(
            "Signed-logP input comparisons do not match the expected panel order: "
            f"{list(PANEL_ORDER)}"
        )

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
    task: str,
    comparison: str,
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
                task=task,
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
    task: str,
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
                task=task,
                comparison=comparison,
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
    task: str,
    comparison: str,
    metric: str,
    band: str,
    parameter: str,
    region: str,
) -> pd.Series:
    mask = (
        (table["group"] == group)
        & (table["task"] == task)
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
            f"group={group!r}, task={task!r}, comparison={comparison!r}, "
            f"metric={metric!r}, band={band!r}, parameter={parameter!r}, "
            f"region={region!r}; got {len(matches)}."
        )
    return matches.iloc[0]


def plot_layered_heatmap(
    value_matrices: list[pd.DataFrame],
    p_matrices: list[pd.DataFrame],
    *,
    label_top: str,
    panel_labels: tuple[str, ...] = PANEL_ORDER,
    cellsize_mm: tuple[float, float] = CELL_SIZE_MM,
    parameter_layout: bool = False,
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
        label_right_width_mm=22.0 if parameter_layout else cfg.strip_right_width_mm,
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
    if parameter_layout:
        for text in fig.findobj(match=matplotlib.text.Text):
            if text.get_text() in panel_labels:
                text.set_rotation(0)
    return fig


def render_group_heatmap(
    group: GroupSpec,
    table: pd.DataFrame,
    *,
    task: str,
    transposed: bool = False,
) -> tuple[Path, Path]:
    output = (
        TRANSPOSED_HEATMAP_OUTPUTS[(group.key, task)]
        if transposed
        else HEATMAP_OUTPUTS[(group.key, task)]
    )
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for comparison in PANEL_ORDER:
        value_matrix, p_matrix = build_group_matrices(group, table, task=task, comparison=comparison)
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


def render_parameter_heatmap(
    table: pd.DataFrame,
    *,
    task: str,
    transposed: bool = False,
) -> tuple[Path, Path]:
    output = (
        TRANSPOSED_HEATMAP_OUTPUTS[(PARAMETER_GROUP_KEY, task)]
        if transposed
        else HEATMAP_OUTPUTS[(PARAMETER_GROUP_KEY, task)]
    )
    value_matrices: list[pd.DataFrame] = []
    p_matrices: list[pd.DataFrame] = []
    for comparison in PANEL_ORDER:
        value_matrix, p_matrix = build_parameter_matrices(table, task=task, comparison=comparison)
        if transposed:
            value_matrix, p_matrix = transpose_heatmap_matrices(value_matrix, p_matrix)
        value_matrices.append(value_matrix)
        p_matrices.append(p_matrix)

    fig = plot_layered_heatmap(
        value_matrices,
        p_matrices,
        label_top=output.label_top,
        cellsize_mm=PARAMETER_CELL_SIZE_MM,
        parameter_layout=True,
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


def run_workflow() -> dict[tuple[str, str, str], tuple[Path, Path]]:
    HEATMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table(SIGNED_LOGP_OUTPUT_PATH)

    outputs: dict[tuple[str, str, str], tuple[Path, Path]] = {}
    for group in GROUPS:
        for task in TASKS:
            outputs[(group.key, task, "original")] = render_group_heatmap(group, table, task=task)
            outputs[(group.key, task, "transposed")] = render_group_heatmap(
                group,
                table,
                task=task,
                transposed=True,
            )
    for task in TASKS:
        outputs[(PARAMETER_GROUP_KEY, task, "original")] = render_parameter_heatmap(table, task=task)
        outputs[(PARAMETER_GROUP_KEY, task, "transposed")] = render_parameter_heatmap(
            table,
            task=task,
            transposed=True,
        )

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[tuple[str, str, str], tuple[Path, Path]]) -> None:
    expected_keys = {
        (group_key, task, "original")
        for group_key, task in HEATMAP_OUTPUTS
    } | {
        (group_key, task, "transposed")
        for group_key, task in TRANSPOSED_HEATMAP_OUTPUTS
    }
    if set(outputs) != expected_keys:
        raise ValueError(f"Expected output keys {sorted(expected_keys)}; got {sorted(outputs)}.")
    if PANEL_ORDER != DISPLAY_COMPARISONS:
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
    for (group_name, task_name, orientation), paths in heatmap_outputs.items():
        print(f"{group_name} {task_name} {orientation}: saved heatmap PDF: {paths[0]}")
        print(f"{group_name} {task_name} {orientation}: saved heatmap PNG: {paths[1]}")
