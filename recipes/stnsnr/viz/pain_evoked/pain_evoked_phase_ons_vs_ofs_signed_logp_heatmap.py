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
from recipes.stnsnr.stats.pain_evoked.pain_evoked_phase_ons_vs_ofs_signed_logp import (
    COMPARISON_TEXT,
    CONNECTIVITY_REGION,
    EVOKED_METRICS,
    EXPECTED_ROW_COUNT,
    MODEL_DIR,
    NEGATIVE_DIRECTION_LABEL,
    OUTPUT_DIR,
    OUTPUT_PATH,
    POSITIVE_DIRECTION_LABEL,
    REGIONS,
    SOURCE_CONTRAST,
    VALUE_LIMIT,
)


COLORBAR_LABEL = r"Signed (ONS-OFS) $-\log_{10}(p_{\mathrm{Tukey}})$"
COLORBAR_POSITIVE_ENDPOINT_LABEL = POSITIVE_DIRECTION_LABEL
COLORBAR_NEGATIVE_ENDPOINT_LABEL = NEGATIVE_DIRECTION_LABEL
COLORBAR_DIRECTION_Y_OFFSET_MM = 2.5
COLORBAR_DIRECTION_FONTSIZE = 6
CELL_SIZE_MM = (3.0, 3.0)
TOP_STRIP_HEIGHT_MM = cfg.strip_top_height_mm
RIGHT_STRIP_WIDTH_MM = cfg.strip_right_width_mm
COLORBAR_PAD_MM = 8.0
REGION_TOP_COLORBAR_PAD_MM = 12.0
REGION_TOP_LABEL_FONTSIZE = 7
GRID_LINE_COLOR = "black"
GRID_LINE_WIDTH = 0.6
GRID_LINE_ALPHA = 0.9
OUTPUT_PDF = OUTPUT_DIR / "pain_evoked_phase_ons_vs_ofs_signed_logp_summary_heatmap.pdf"
OUTPUT_PNG = OUTPUT_DIR / "pain_evoked_phase_ons_vs_ofs_signed_logp_summary_heatmap.png"
REGION_TOP_OUTPUT_PDF = (
    OUTPUT_DIR / "pain_evoked_phase_ons_vs_ofs_signed_logp_summary_region_top_heatmap.pdf"
)
REGION_TOP_OUTPUT_PNG = (
    OUTPUT_DIR / "pain_evoked_phase_ons_vs_ofs_signed_logp_summary_region_top_heatmap.png"
)


@dataclass(frozen=True)
class RowSpec:
    metric: str
    band: str
    parameter: str
    label: str


LOCAL_ROW_SPECS_BY_REGION = {
    "SNr": (
        RowSpec("raw", "Delta", "", "ΔTotal power δ"),
        RowSpec("raw", "Theta", "", "ΔTotal power θ"),
        RowSpec("params", "Exponent", "Exponent", "ΔExponent"),
        RowSpec("params", "Offset", "Offset", "ΔOffset"),
    ),
    "STN": (
        RowSpec("raw", "Delta", "", "ΔTotal power δ"),
        RowSpec("period", "Delta", "", "ΔPeriodic power δ"),
        RowSpec("raw", "Theta", "", "ΔTotal power θ"),
        RowSpec("period", "Alpha", "", "ΔPeriodic power α"),
        RowSpec("period", "Beta_low", "", "ΔPeriodic power Low-β"),
        RowSpec("period", "Gamma", "", "ΔPeriodic power γ"),
        RowSpec("params", "Exponent", "Exponent", "ΔExponent"),
        RowSpec("params", "Offset", "Offset", "ΔOffset"),
    ),
}

CONNECTIVITY_ROW_SPECS = (
    RowSpec("coh", "Delta", "", "ΔCoherence δ"),
    RowSpec("wpli", "Delta", "", "ΔwPLI δ"),
)


def load_signed_logp_table(path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing signed-logP table: {input_path}. "
            "Run recipes/stnsnr/stats/pain_evoked/pain_evoked_phase_ons_vs_ofs_signed_logp.py first."
        )

    table = pd.read_csv(input_path)
    string_columns = [
        "source_csv",
        "feature_group",
        "region",
        "metric",
        "metric_label",
        "band",
        "band_label",
        "parameter",
        "parameter_label",
        "evoked_metric",
        "evoked_label",
        "source_model",
        "source_contrast",
        "source_phase_first",
        "source_phase_second",
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

    required_columns = {
        "source_csv",
        "feature_group",
        "region",
        "metric",
        "band",
        "parameter",
        "evoked_metric",
        "evoked_label",
        "source_model",
        "source_contrast",
        "comparison",
        "positive_direction",
        "negative_direction",
        "source_estimate_ons_minus_ofs",
        "signed_estimate",
        "p_tukey",
        "signed_logp",
        "signed_logp_display",
    }
    missing = required_columns.difference(table.columns)
    if missing:
        raise ValueError(f"Signed-logP input is missing columns: {sorted(missing)}")

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

    source_estimate = pd.to_numeric(table["source_estimate_ons_minus_ofs"], errors="coerce")
    signed_estimate = pd.to_numeric(table["signed_estimate"], errors="coerce")
    if not np.allclose(source_estimate, signed_estimate, equal_nan=True):
        raise ValueError("signed_estimate must equal source_estimate_ons_minus_ofs.")

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


def build_group_matrix(
    table: pd.DataFrame,
    *,
    region: str,
    row_specs: tuple[RowSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    column_labels = [label for _, label in EVOKED_METRICS]
    row_labels = [row_spec.label for row_spec in row_specs]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for row_spec in row_specs:
        for evoked_metric, evoked_label in EVOKED_METRICS:
            row = select_single_row(
                table,
                region=region,
                row_spec=row_spec,
                evoked_metric=evoked_metric,
            )
            value_matrix.loc[row_spec.label, evoked_label] = float(row["signed_logp_display"])
            p_matrix.loc[row_spec.label, evoked_label] = float(row["p_tukey"])

    return value_matrix, p_matrix


def select_single_row(
    table: pd.DataFrame,
    *,
    region: str,
    row_spec: RowSpec,
    evoked_metric: str,
) -> pd.Series:
    mask = (
        (table["region"] == region)
        & (table["metric"] == row_spec.metric)
        & (table["band"] == row_spec.band)
        & (table["evoked_metric"] == evoked_metric)
    )
    if row_spec.parameter:
        mask &= table["parameter"] == row_spec.parameter
    else:
        mask &= table["parameter"].fillna("").eq("")

    matches = table.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one row for "
            f"region={region!r}, metric={row_spec.metric!r}, band={row_spec.band!r}, "
            f"parameter={row_spec.parameter!r}, evoked_metric={evoked_metric!r}; "
            f"got {len(matches)}."
        )
    return matches.iloc[0]


def build_summary_grid(table: pd.DataFrame) -> tuple[list[list[pd.DataFrame]], list[list[pd.DataFrame]]]:
    value_grid: list[list[pd.DataFrame]] = []
    p_grid: list[list[pd.DataFrame]] = []

    for region in (*REGIONS, CONNECTIVITY_REGION):
        row_specs = CONNECTIVITY_ROW_SPECS if region == CONNECTIVITY_REGION else LOCAL_ROW_SPECS_BY_REGION[region]
        value_matrix, p_matrix = build_group_matrix(
            table,
            region=region,
            row_specs=row_specs,
        )
        value_grid.append([value_matrix])
        p_grid.append([p_matrix])

    return value_grid, p_grid


def build_region_top_summary_grid(
    table: pd.DataFrame,
) -> tuple[list[list[pd.DataFrame]], list[list[pd.DataFrame]]]:
    value_row: list[pd.DataFrame] = []
    p_row: list[pd.DataFrame] = []

    for region in (*REGIONS, CONNECTIVITY_REGION):
        row_specs = CONNECTIVITY_ROW_SPECS if region == CONNECTIVITY_REGION else LOCAL_ROW_SPECS_BY_REGION[region]
        value_matrix, p_matrix = build_group_matrix(
            table,
            region=region,
            row_specs=row_specs,
        )
        value_matrix = value_matrix.T.copy()
        p_matrix = p_matrix.T.copy()
        value_row.append(value_matrix)
        p_row.append(p_matrix)

    return [value_row], [p_row]


def render_summary_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    value_grid, p_grid = build_summary_grid(table)
    fig = vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=p_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=[COMPARISON_TEXT],
        label_right=[*REGIONS, CONNECTIVITY_REGION],
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=CELL_SIZE_MM,
        panel_gap_mm=cfg.panel_gap,
        label_top_height_mm=TOP_STRIP_HEIGHT_MM,
        label_right_width_mm=RIGHT_STRIP_WIDTH_MM,
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
        colorbar_pad_mm=COLORBAR_PAD_MM,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(fig)
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return OUTPUT_PDF, OUTPUT_PNG


def render_region_top_summary_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    value_grid, p_grid = build_region_top_summary_grid(table)
    fig = vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=p_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=[*REGIONS, CONNECTIVITY_REGION],
        label_right=None,
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=CELL_SIZE_MM,
        panel_gap_mm=cfg.panel_gap,
        label_top_height_mm=TOP_STRIP_HEIGHT_MM,
        label_right_width_mm=RIGHT_STRIP_WIDTH_MM,
        strip_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_right_bg_color=cfg.label_right_bg_color,
        label_text_color=cfg.label_text_color,
        label_fontsize=REGION_TOP_LABEL_FONTSIZE,
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
        colorbar_pad_mm=REGION_TOP_COLORBAR_PAD_MM,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(fig)
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(REGION_TOP_OUTPUT_PDF, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(REGION_TOP_OUTPUT_PNG, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return REGION_TOP_OUTPUT_PDF, REGION_TOP_OUTPUT_PNG


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


def run_workflow() -> tuple[Path, ...]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_signed_logp_table(OUTPUT_PATH)
    paths = (*render_summary_heatmap(table), *render_region_top_summary_heatmap(table))
    validate_output_files(paths)
    return paths


def validate_output_files(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not Path(path).exists() or Path(path).stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Expected non-empty heatmap files were not created: {missing}")

    wrong_parent = [str(path) for path in paths if Path(path).parent != OUTPUT_DIR]
    if wrong_parent:
        raise ValueError(f"Heatmap outputs must be written only under {OUTPUT_DIR}: {wrong_parent}")


if __name__ == "__main__":
    for output_path in run_workflow():
        print(f"pain-evoked Phase ONS-vs-OFS heatmap: saved: {output_path}")
