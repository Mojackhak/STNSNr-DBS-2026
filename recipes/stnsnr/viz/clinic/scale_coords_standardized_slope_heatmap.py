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
from recipes.stnsnr.stats.clinic.scale_coords_standardized_slope import (
    COORDINATES,
    DELTA_3M_SCALE_COLUMNS,
    OUTPUT_PATH,
    SCALE_DIR,
    STN_3M_SCALE_COLUMNS,
    VALUE_LIMIT,
)


COLORBAR_LABEL = "Standardized slope"

SCALE_LABEL_OVERRIDES = {
    "MDS-UDPRS II": "MDS-UPDRS II",
    "Rigidity subscore (Med OFF)": "Rigidity",
    "Bradykinesia subscore (Med OFF)": "Bradykinesia",
    "Tremor subscore (Med OFF)": "Tremor",
}


@dataclass(frozen=True)
class HeatmapSpec:
    key: str
    group: str
    scales: tuple[str, ...]
    label_top: str
    pdf_path: Path
    png_path: Path
    transposed: bool = False


HEATMAPS = (
    HeatmapSpec(
        key="stn_3m_efficacy",
        group="stn_3m_efficacy",
        scales=STN_3M_SCALE_COLUMNS,
        label_top="STN",
        pdf_path=SCALE_DIR / "scale_stn_3m_efficacy_standardized_slope_heatmap.pdf",
        png_path=SCALE_DIR / "scale_stn_3m_efficacy_standardized_slope_heatmap.png",
    ),
    HeatmapSpec(
        key="stn_3m_efficacy_transposed",
        group="stn_3m_efficacy",
        scales=STN_3M_SCALE_COLUMNS,
        label_top="STN",
        pdf_path=SCALE_DIR / "scale_stn_3m_efficacy_standardized_slope_transposed_heatmap.pdf",
        png_path=SCALE_DIR / "scale_stn_3m_efficacy_standardized_slope_transposed_heatmap.png",
        transposed=True,
    ),
    HeatmapSpec(
        key="stn_snr_3m_delta_efficacy",
        group="stn_snr_3m_delta_efficacy",
        scales=DELTA_3M_SCALE_COLUMNS,
        label_top="+SNr",
        pdf_path=SCALE_DIR / "scale_stn_snr_3m_delta_efficacy_standardized_slope_heatmap.pdf",
        png_path=SCALE_DIR / "scale_stn_snr_3m_delta_efficacy_standardized_slope_heatmap.png",
    ),
    HeatmapSpec(
        key="stn_snr_3m_delta_efficacy_transposed",
        group="stn_snr_3m_delta_efficacy",
        scales=DELTA_3M_SCALE_COLUMNS,
        label_top="+SNr",
        pdf_path=SCALE_DIR / "scale_stn_snr_3m_delta_efficacy_standardized_slope_transposed_heatmap.pdf",
        png_path=SCALE_DIR / "scale_stn_snr_3m_delta_efficacy_standardized_slope_transposed_heatmap.png",
        transposed=True,
    ),
)


def load_standardized_slope_table(path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing standardized slope table: {input_path}. "
            "Run recipes/stnsnr/stats/clinic/scale_coords_standardized_slope.py first."
        )

    table = pd.read_csv(input_path)
    required = {
        "group",
        "scale",
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


def prepare_heatmap_matrices(
    table: pd.DataFrame,
    spec: HeatmapSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_labels = [axis_label_for_scale(scale) for scale in spec.scales]
    coordinate_order = [coordinate.key for coordinate in COORDINATES]
    column_labels = [coordinate.label for coordinate in COORDINATES]

    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    q_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)

    subset = table[table["group"].eq(spec.group)].copy()
    duplicate_mask = subset.duplicated(["scale", "coordinate"], keep=False)
    if duplicate_mask.any():
        duplicates = subset.loc[duplicate_mask, ["scale", "coordinate"]].to_dict("records")
        raise ValueError(f"{spec.key}: duplicated scale-coordinate rows: {duplicates[:5]}")

    for _, row in subset.iterrows():
        scale = str(row["scale"])
        coordinate = str(row["coordinate"])
        if scale not in spec.scales or coordinate not in coordinate_order:
            continue
        row_label = axis_label_for_scale(scale)
        coordinate_label = next(item.label for item in COORDINATES if item.key == coordinate)
        value_matrix.loc[row_label, coordinate_label] = row["standardized_slope_display"]
        q_matrix.loc[row_label, coordinate_label] = row["q"]

    return value_matrix, q_matrix


def axis_label_for_scale(scale: str) -> str:
    is_delta = scale.startswith("Δ")
    base = scale.removeprefix("Δ") if is_delta else scale
    base = base.removesuffix(" Δ% (STN, 3m)")
    base = base.removesuffix(" Δ% (+SNr, 3m)")
    base = SCALE_LABEL_OVERRIDES.get(base, base)
    return f"Δ{base} Δ%" if is_delta else f"{base} Δ%"


def plot_standardized_slope_heatmap(
    value_matrix: pd.DataFrame,
    q_matrix: pd.DataFrame,
    label_top: str,
) -> plt.Figure:
    return vdf.plot_well_heatmap_df(
        df_value=value_matrix,
        df_p=q_matrix,
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
        cellsize=(3.0, 3.0),
        label_top_height_mm=cfg.strip_top_height_mm,
        label_top_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_top_text_color=cfg.label_text_color,
        label_top_fontsize=7,
        label_top_fontweight="bold",
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=8.0,
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


def transpose_heatmap_matrices(
    value_matrix: pd.DataFrame,
    q_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return value_matrix.T.copy(), q_matrix.T.copy()


def run_workflow() -> dict[str, tuple[Path, Path]]:
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    table = load_standardized_slope_table()

    outputs: dict[str, tuple[Path, Path]] = {}
    for spec in HEATMAPS:
        value_matrix, q_matrix = prepare_heatmap_matrices(table, spec)
        _validate_plot_matrices(spec, value_matrix, q_matrix)
        if spec.transposed:
            value_matrix, q_matrix = transpose_heatmap_matrices(value_matrix, q_matrix)

        fig = plot_standardized_slope_heatmap(
            value_matrix,
            q_matrix,
            label_top=spec.label_top,
        )
        vdf._ensure_strip_background_opaque(fig)
        fig.savefig(spec.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
        fig.savefig(spec.png_path, bbox_inches="tight", dpi=cfg.dpi)
        plt.close(fig)

        outputs[spec.key] = (spec.pdf_path, spec.png_path)

    _validate_output_files(outputs)
    return outputs


def _validate_plot_matrices(
    spec: HeatmapSpec,
    value_matrix: pd.DataFrame,
    q_matrix: pd.DataFrame,
) -> None:
    expected_columns = [coordinate.label for coordinate in COORDINATES]
    if not value_matrix.index.equals(q_matrix.index) or not value_matrix.columns.equals(q_matrix.columns):
        raise ValueError(f"{spec.key}: value and q plot matrices are not aligned.")
    if value_matrix.columns.tolist() != expected_columns:
        raise ValueError(f"{spec.key}: coordinate columns do not match the configured order.")
    expected_shape = (len(spec.scales), len(COORDINATES))
    if value_matrix.shape != expected_shape:
        raise ValueError(f"{spec.key}: expected a {expected_shape} matrix; got {value_matrix.shape}.")
    if value_matrix.isna().all().all():
        raise ValueError(f"{spec.key}: value matrix contains no plottable cells.")
    if q_matrix.isna().all().all():
        raise ValueError(f"{spec.key}: q matrix contains no plottable cells.")


def _validate_output_files(outputs: dict[str, tuple[Path, Path]]) -> None:
    expected_keys = {spec.key for spec in HEATMAPS}
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
    for group, (pdf_path, png_path) in heatmap_outputs.items():
        print(f"{group}: saved heatmap PDF: {pdf_path}")
        print(f"{group}: saved heatmap PNG: {png_path}")
