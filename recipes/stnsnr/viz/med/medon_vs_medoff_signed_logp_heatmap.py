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
from recipes.stnsnr.stats.med.medon_vs_medoff_signed_logp import (
    BAND_LABELS,
    BANDS,
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
    GroupSpec,
)


VALUE_LIMIT = 4.0
COLORBAR_LABEL = r"Signed $-\log_{10}(p_{\mathrm{Tukey}})$"
COLORBAR_DIRECTION_X_OFFSET_MM = 1.0
COLORBAR_DIRECTION_Y_OFFSET_MM = 2.5
COLORBAR_DIRECTION_FONTSIZE = 6


@dataclass(frozen=True)
class HeatmapOutputSpec:
    key: str
    pdf_path: Path
    png_path: Path


HEATMAP_OUTPUTS = {
    "stn_local": HeatmapOutputSpec(
        key="stn_local",
        pdf_path=OUTPUT_DIR / "med_stn_local_medon_vs_medoff_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "med_stn_local_medon_vs_medoff_signed_logp_heatmap.png",
    ),
    "snr_local": HeatmapOutputSpec(
        key="snr_local",
        pdf_path=OUTPUT_DIR / "med_snr_local_medon_vs_medoff_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "med_snr_local_medon_vs_medoff_signed_logp_heatmap.png",
    ),
    "connectivity": HeatmapOutputSpec(
        key="connectivity",
        pdf_path=OUTPUT_DIR / "med_connectivity_medon_vs_medoff_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "med_connectivity_medon_vs_medoff_signed_logp_heatmap.png",
    ),
    PARAMETER_GROUP_KEY: HeatmapOutputSpec(
        key=PARAMETER_GROUP_KEY,
        pdf_path=OUTPUT_DIR / "med_params_medon_vs_medoff_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "med_params_medon_vs_medoff_signed_logp_heatmap.png",
    ),
}


def load_signed_logp_table(input_path: str | Path = SIGNED_LOGP_OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP input table: {input_path}. "
            "Run recipes/stnsnr/stats/med/medon_vs_medoff_signed_logp.py first."
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
        "source_model",
        "source_contrast",
        "positive_direction",
        "negative_direction",
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
        "parameter",
        "parameter_label",
        "region",
        "source_model",
        "source_contrast",
        "positive_direction",
        "negative_direction",
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
    if (table["positive_direction"] != POSITIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Positive direction must be {POSITIVE_DIRECTION_LABEL!r}.")
    if (table["negative_direction"] != NEGATIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Negative direction must be {NEGATIVE_DIRECTION_LABEL!r}.")

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


def build_group_matrices(group: GroupSpec, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [metric.label for metric in group.metrics]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, band_label in zip(BANDS, column_labels, strict=True):
            row = select_single_row(
                table,
                group=group.key,
                metric=metric.key,
                band=band,
                parameter="",
                region=group.region or "",
            )
            value_matrix.loc[row_label, band_label] = clip_signed_logp(row["signed_logp"])
            p_matrix.loc[row_label, band_label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def build_parameter_matrices(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = list(PARAMETER_REGIONS)
    column_labels = [parameter.label for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for parameter in PARAMETERS:
        for region in PARAMETER_REGIONS:
            row = select_single_row(
                table,
                group=PARAMETER_GROUP_KEY,
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


def plot_heatmap(
    value_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    *,
    label_top: str,
    cellsize: tuple[float, float] = (4.0, 4.0),
    direction_y_offset_mm: float = COLORBAR_DIRECTION_Y_OFFSET_MM,
    direction_fontsize: float = COLORBAR_DIRECTION_FONTSIZE,
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
        cellsize=cellsize,
        label_top_height_mm=cfg.strip_top_height_mm,
        label_top_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_top_text_color=cfg.label_text_color,
        label_top_fontsize=7,
        label_top_fontweight="bold",
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=5.0,
        grid_lines=None,
        gline_width=0.6,
        gline_alpha=0.9,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_width_mm=3.0,
        colorbar_pad_mm=2.0,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(
        fig,
        y_offset_mm=direction_y_offset_mm,
        fontsize=direction_fontsize,
    )
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
    x_offset_mm: float = COLORBAR_DIRECTION_X_OFFSET_MM,
    y_offset_mm: float = COLORBAR_DIRECTION_Y_OFFSET_MM,
    fontsize: float = COLORBAR_DIRECTION_FONTSIZE,
) -> None:
    colorbar_axis = find_colorbar_axis(fig)
    if colorbar_axis is None:
        return
    heatmap_axis = find_heatmap_axis(fig, colorbar_axis)
    if heatmap_axis is None:
        return

    pos = colorbar_axis.get_position()
    heatmap_pos = heatmap_axis.get_position()
    fig_width_in = fig.get_size_inches()[0]
    fig_height_in = fig.get_size_inches()[1]
    x_offset = (x_offset_mm / 25.4) / fig_width_in
    y_offset = (y_offset_mm / 25.4) / fig_height_in
    x_label = heatmap_pos.x1 + x_offset
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


def find_colorbar_axis(fig: plt.Figure) -> plt.Axes | None:
    candidates = []
    for ax in fig.axes:
        pos = ax.get_position()
        if pos.width < 0.18 and pos.height > 0.25 and len(ax.get_xticks()) == 0 and len(ax.get_yticks()) > 0:
            candidates.append(ax)
    if not candidates:
        return None
    return max(candidates, key=lambda ax: ax.get_position().x0)


def find_heatmap_axis(fig: plt.Figure, colorbar_axis: plt.Axes) -> plt.Axes | None:
    candidates = []
    for ax in fig.axes:
        if ax is colorbar_axis:
            continue
        pos = ax.get_position()
        if pos.width > 0.2 and pos.height > 0.3:
            candidates.append(ax)
    if not candidates:
        return None
    return max(candidates, key=lambda ax: ax.get_position().width * ax.get_position().height)


def render_group_heatmap(
    group: GroupSpec,
    table: pd.DataFrame,
) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS[group.key]
    value_matrix, p_matrix = build_group_matrices(group, table)
    fig = plot_heatmap(value_matrix, p_matrix, label_top=group.label_top)
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def render_parameter_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    output = HEATMAP_OUTPUTS[PARAMETER_GROUP_KEY]
    value_matrix, p_matrix = build_parameter_matrices(table)
    fig = plot_heatmap(
        value_matrix,
        p_matrix,
        label_top=PARAMETER_LABEL_TOP,
        cellsize=(4.0, 4.0),
        direction_y_offset_mm=7.0,
        direction_fontsize=5.0,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def run_workflow() -> dict[str, tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table(SIGNED_LOGP_OUTPUT_PATH)

    outputs: dict[str, tuple[Path, Path]] = {}
    for group in GROUPS:
        outputs[group.key] = render_group_heatmap(group, table)
    outputs[PARAMETER_GROUP_KEY] = render_parameter_heatmap(table)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[str, tuple[Path, Path]]) -> None:
    expected_keys = {"stn_local", "snr_local", "connectivity", PARAMETER_GROUP_KEY}
    if set(outputs) != expected_keys:
        raise ValueError(f"Expected output keys {sorted(expected_keys)}; got {sorted(outputs)}.")

    missing = [
        str(path)
        for paths in outputs.values()
        for path in paths
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Expected heatmap files were not created: {missing}")


if __name__ == "__main__":
    heatmap_outputs = run_workflow()
    for group_name, paths in heatmap_outputs.items():
        print(f"{group_name}: saved heatmap PDF: {paths[0]}")
        print(f"{group_name}: saved heatmap PNG: {paths[1]}")
