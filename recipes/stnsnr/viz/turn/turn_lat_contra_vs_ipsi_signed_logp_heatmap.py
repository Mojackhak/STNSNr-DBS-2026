from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
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
from recipes.stnsnr.stats.turn.turn_lat_contra_vs_ipsi_signed_logp import (
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
    POSITIVE_DIRECTION_LABEL,
    SIGNED_LOGP_OUTPUT_PATH,
    SOURCE_CONTRAST,
    VALUE_LIMIT,
    GroupSpec,
)


COLORBAR_LABEL = r"Signed $-\log_{10}(p_{\mathrm{Tukey}})$"
COLORBAR_POSITIVE_ENDPOINT_LABEL = "Ipsi > Contra"
COLORBAR_NEGATIVE_ENDPOINT_LABEL = "Ipsi < Contra"
COLORBAR_DIRECTION_Y_OFFSET_MM = 2.5
COLORBAR_DIRECTION_FONTSIZE = 6
COLORBAR_LABEL_OFFSET_MM = 8.0
COLORBAR_PAD_MM = 12.0
CELL_SIZE_MM = (3.0, 3.0)
GRID_LINE_COLOR = "black"
GRID_LINE_WIDTH = 0.6
GRID_LINE_ALPHA = 0.9


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
        pdf_path=OUTPUT_DIR / "turn_lat_stn_local_contra_vs_ipsi_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_lat_stn_local_contra_vs_ipsi_signed_logp_heatmap.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        label_top="SNr",
        pdf_path=OUTPUT_DIR / "turn_lat_snr_local_contra_vs_ipsi_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_lat_snr_local_contra_vs_ipsi_signed_logp_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        pdf_path=OUTPUT_DIR / "turn_lat_connectivity_contra_vs_ipsi_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_lat_connectivity_contra_vs_ipsi_signed_logp_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        label_top=PARAMETER_LABEL_TOP,
        pdf_path=OUTPUT_DIR / "turn_lat_params_contra_vs_ipsi_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "turn_lat_params_contra_vs_ipsi_signed_logp_heatmap.png",
    ),
}


def load_signed_logp_table(input_path: str | Path = SIGNED_LOGP_OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP input table: {input_path}. "
            "Run recipes/stnsnr/stats/turn/turn_lat_contra_vs_ipsi_signed_logp.py first."
        )

    table = pd.read_csv(input_path)
    string_columns = [
        "source_csv",
        "group",
        "label_top",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
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
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "region",
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
    if (table["comparison"] != COMPARISON_TEXT).any():
        raise ValueError(f"Comparison must be {COMPARISON_TEXT!r}.")
    if (table["positive_direction"] != POSITIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Positive direction must be {POSITIVE_DIRECTION_LABEL!r}.")
    if (table["negative_direction"] != NEGATIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Negative direction must be {NEGATIVE_DIRECTION_LABEL!r}.")

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


def display_value(value: object) -> float:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if not math.isfinite(float(numeric_value)):
        return math.nan
    return float(np.clip(float(numeric_value), -VALUE_LIMIT, VALUE_LIMIT))


def find_signed_logp_row(
    table: pd.DataFrame,
    *,
    group: str,
    metric: str,
    band: str,
    parameter: str,
    region: str,
) -> pd.Series:
    mask = (
        (table["group"] == group)
        & (table["metric"] == metric)
        & (table["band"] == band)
        & (table["parameter"] == parameter)
        & (table["region"] == region)
    )
    matches = table.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one signed-logP row for "
            f"group={group!r}, metric={metric!r}, band={band!r}, "
            f"parameter={parameter!r}, region={region!r}; got {len(matches)}."
        )
    return matches.iloc[0]


def build_group_matrices(group: GroupSpec, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [metric.label for metric in group.metrics]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, band_label in zip(BANDS, column_labels, strict=True):
            row = find_signed_logp_row(
                table,
                group=group.key,
                metric=metric.key,
                band=band,
                parameter="",
                region=group.region or "",
            )
            value_matrix.loc[row_label, band_label] = display_value(row["signed_logp_display"])
            p_matrix.loc[row_label, band_label] = pd.to_numeric(row["p_tukey"], errors="coerce")

    return value_matrix, p_matrix


def build_parameter_matrices(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = list(PARAMETER_REGIONS)
    column_labels = [parameter.label for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter, column_label in zip(PARAMETERS, column_labels, strict=True):
        for region in PARAMETER_REGIONS:
            row = find_signed_logp_row(
                table,
                group=PARAMETER_GROUP_KEY,
                metric="params",
                band=parameter.key,
                parameter=parameter.key,
                region=region,
            )
            value_matrix.loc[region, column_label] = display_value(row["signed_logp_display"])
            p_matrix.loc[region, column_label] = pd.to_numeric(row["p_tukey"], errors="coerce")

    return value_matrix, p_matrix


def plot_heatmap(
    value_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    *,
    label_top: str,
) -> plt.Figure:
    fig = vdf.plot_well_heatmap_df(
        df_value=value_matrix,
        df_p=p_matrix,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=label_top,
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=CELL_SIZE_MM,
        label_top_height_mm=cfg.strip_top_height_mm,
        label_top_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_top_text_color=cfg.label_text_color,
        label_top_fontsize=7,
        label_top_fontweight=cfg.label_fontweight,
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
        colorbar_width_mm=cfg.colorbar_width_mm,
        colorbar_pad_mm=COLORBAR_PAD_MM,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(fig)
    return fig


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


def find_heatmap_axis(fig: plt.Figure, colorbar_axis: plt.Axes) -> plt.Axes | None:
    candidates = []
    for ax in fig.axes:
        if ax is colorbar_axis:
            continue
        if ax.images or ax.collections:
            candidates.append(ax)
    return candidates[0] if candidates else None


def render_group_heatmap(group: GroupSpec, table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS[group.key]
    value_matrix, p_matrix = build_group_matrices(group, table)
    fig = plot_heatmap(value_matrix, p_matrix, label_top=output.label_top)
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def render_parameter_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS[PARAMETER_GROUP_KEY]
    value_matrix, p_matrix = build_parameter_matrices(table)
    fig = plot_heatmap(value_matrix, p_matrix, label_top=output.label_top)
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def run_workflow() -> dict[str, tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table()

    outputs: dict[str, tuple[Path, Path]] = {}
    for group in GROUPS:
        outputs[group.key] = render_group_heatmap(group, table)
    outputs[PARAMETER_GROUP_KEY] = render_parameter_heatmap(table)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[str, tuple[Path, Path]]) -> None:
    expected_keys = {group.key for group in GROUPS} | {PARAMETER_GROUP_KEY}
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
    for group_name, paths in heatmap_outputs.items():
        print(f"{group_name}: saved heatmap PDF: {paths[0]}")
        print(f"{group_name}: saved heatmap PNG: {paths[1]}")
