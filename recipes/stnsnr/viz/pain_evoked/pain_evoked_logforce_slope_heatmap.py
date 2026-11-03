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
from recipes.stnsnr.stats.pain_evoked.pain_evoked_logforce_slope import (
    CONNECTIVITY_REGION,
    EVOKED_METRICS,
    OUTPUT_DIR,
    OUTPUT_PATH,
    PHASES,
    REGIONS,
    VALUE_LIMIT,
)


COLORBAR_LABEL = r"Slope of evoked ΔLFP metric per centered $\log_{10}(\mathrm{Force})$"
CELL_SIZE_MM = (3.0, 3.0)
OUTPUT_PDF = OUTPUT_DIR / "pain_evoked_logforce_slope_summary_heatmap.pdf"
OUTPUT_PNG = OUTPUT_DIR / "pain_evoked_logforce_slope_summary_heatmap.png"


@dataclass(frozen=True)
class RowSpec:
    metric: str
    band: str
    parameter: str
    label: str


LOCAL_ROW_SPECS_BY_REGION = {
    "SNr": (
        RowSpec("raw", "Delta", "", "ΔTotal power δ"),
        RowSpec("period", "Delta", "", "ΔPeriodic power δ"),
        RowSpec("raw", "Theta", "", "ΔTotal power θ"),
        RowSpec("period", "Alpha", "", "ΔPeriodic power α"),
        RowSpec("period", "Beta_low", "", "ΔPeriodic power Low-β"),
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


def load_logforce_slope_table(path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing LogForce slope table: {input_path}. "
            "Run recipes/stnsnr/stats/pain_evoked/pain_evoked_logforce_slope.py first."
        )

    table = pd.read_csv(input_path)
    required = {
        "feature_group",
        "metric",
        "band",
        "parameter",
        "evoked_metric",
        "evoked_label",
        "region",
        "phase",
        "slope_display",
        "q",
        "source_slope_csv",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")

    validate_logforce_slope_input(table)
    return table


def validate_logforce_slope_input(table: pd.DataFrame) -> None:
    if len(table) != 288:
        raise ValueError(f"Expected 288 LogForce slope rows; got {len(table)}.")
    if set(table["phase"].dropna()) != set(PHASES):
        raise ValueError(f"Unexpected phases: {sorted(table['phase'].dropna().unique())}")
    invalid_display = pd.to_numeric(table["slope_display"], errors="coerce").isna().sum()
    invalid_q = pd.to_numeric(table["q"], errors="coerce").isna().sum()
    if invalid_display or invalid_q:
        raise ValueError(f"Invalid slope_display rows={invalid_display}, invalid q rows={invalid_q}.")


def build_group_matrix(
    table: pd.DataFrame,
    *,
    region: str,
    phase: str,
    row_specs: tuple[RowSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    column_labels = [label for _, label in EVOKED_METRICS]
    row_labels = [row_spec.label for row_spec in row_specs]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    for row_spec in row_specs:
        for evoked_metric, evoked_label in EVOKED_METRICS:
            row = select_single_row(
                table,
                region=region,
                phase=phase,
                row_spec=row_spec,
                evoked_metric=evoked_metric,
            )
            value_matrix.loc[row_spec.label, evoked_label] = float(row["slope_display"])
            q_matrix.loc[row_spec.label, evoked_label] = float(row["q"])

    return value_matrix, q_matrix


def select_single_row(
    table: pd.DataFrame,
    *,
    region: str,
    phase: str,
    row_spec: RowSpec,
    evoked_metric: str,
) -> pd.Series:
    mask = (
        (table["region"] == region)
        & (table["phase"] == phase)
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
            f"region={region!r}, phase={phase!r}, metric={row_spec.metric!r}, "
            f"band={row_spec.band!r}, parameter={row_spec.parameter!r}, "
            f"evoked_metric={evoked_metric!r}; got {len(matches)}."
        )
    return matches.iloc[0]


def build_summary_grid(table: pd.DataFrame) -> tuple[list[list[pd.DataFrame]], list[list[pd.DataFrame]]]:
    value_grid: list[list[pd.DataFrame]] = []
    q_grid: list[list[pd.DataFrame]] = []

    for region in (*REGIONS, CONNECTIVITY_REGION):
        row_specs = CONNECTIVITY_ROW_SPECS if region == CONNECTIVITY_REGION else LOCAL_ROW_SPECS_BY_REGION[region]
        value_row: list[pd.DataFrame] = []
        q_row: list[pd.DataFrame] = []
        for phase in PHASES:
            value_matrix, q_matrix = build_group_matrix(
                table,
                region=region,
                phase=phase,
                row_specs=row_specs,
            )
            value_row.append(value_matrix)
            q_row.append(q_matrix)
        value_grid.append(value_row)
        q_grid.append(q_row)

    return value_grid, q_grid


def render_summary_heatmap(table: pd.DataFrame) -> tuple[Path, Path]:
    value_grid, q_grid = build_summary_grid(table)
    fig = vdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=q_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=list(PHASES),
        label_right=[*REGIONS, CONNECTIVITY_REGION],
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
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return OUTPUT_PDF, OUTPUT_PNG


def run_workflow() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = load_logforce_slope_table(OUTPUT_PATH)
    paths = render_summary_heatmap(table)
    validate_output_files(paths)
    return paths


def validate_output_files(paths: tuple[Path, Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).exists() or Path(path).stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Expected non-empty heatmap files were not created: {missing}")


if __name__ == "__main__":
    output_pdf, output_png = run_workflow()
    print(f"pain-evoked LogForce slope heatmap: saved PDF: {output_pdf}")
    print(f"pain-evoked LogForce slope heatmap: saved PNG: {output_png}")
