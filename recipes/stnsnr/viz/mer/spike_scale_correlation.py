from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.viz.visualdf as vdf
from recipes.stnsnr.stats.mer.spike_scale_correlation import (
    CORRELATION_OUTPUT_PATH,
    OUTPUT_DIR,
    SPIKE_PARAM_ORDER,
    SPIKE_X_LABELS,
    Y_COLUMNS,
)


HEATMAP_PDF_PATH = OUTPUT_DIR / "mer_spike_scale_corr_heatmap.pdf"
HEATMAP_PNG_PATH = OUTPUT_DIR / "mer_spike_scale_corr_heatmap.png"

Y_AXIS_LABELS = {
    "LCT improvement rate": "LCT Δ%",
    "MDS-UPDRS III (Med-OFF)": "MDS-UPDRS III",
    "MDS-UPDRS III axial (Med-OFF)": "Axial",
    "MDS-UPDRS III rigidity (Med-OFF)": "Rigidity",
    "MDS-UPDRS III bradykinesia (Med-OFF)": "Bradykinesia",
    "MDS-UPDRS III rest tremor (Med-OFF)": "Rest tremor",
    "MDS-UPDRS III action tremor (Med-OFF)": "Action tremor",
}


def load_correlation_matrices(
    correlation_path: str | Path = CORRELATION_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load r and raw p-value matrices from the correlation workbook."""
    correlation_path = Path(correlation_path)
    r = pd.read_excel(correlation_path, sheet_name="r", index_col=0)
    p_raw = pd.read_excel(correlation_path, sheet_name="p_raw", index_col=0)
    return r, p_raw


def prepare_heatmap_matrices(r: pd.DataFrame, p_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reorder matrices and replace internal spike feature keys with display labels."""
    x_features = _ordered_x_features(r.columns.tolist())

    missing_y = [column for column in Y_COLUMNS if column not in r.index]
    if missing_y:
        raise ValueError(f"Correlation matrix is missing y-axis rows: {missing_y}.")

    r_plot = r.loc[Y_COLUMNS, x_features].copy()
    p_plot = p_raw.loc[Y_COLUMNS, x_features].copy()

    if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
        raise ValueError("r and p_raw matrices are not aligned after reordering.")

    rename_map = dict(zip(x_features, SPIKE_X_LABELS, strict=True))
    r_plot = r_plot.rename(columns=rename_map)
    p_plot = p_plot.rename(columns=rename_map)
    r_plot = r_plot.rename(index=Y_AXIS_LABELS)
    p_plot = p_plot.rename(index=Y_AXIS_LABELS)
    return r_plot, p_plot


def plot_correlation_heatmap(r_plot: pd.DataFrame, p_plot: pd.DataFrame) -> plt.Figure:
    """Create the MER spike-scale correlation heatmap."""
    return vdf.plot_well_heatmap_df(
        df_value=r_plot,
        df_p=p_plot,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        vmin=-1,
        vmax=1,
        vmode="sym",
        cellsize=(3, 3),
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=8.0,
        grid_lines=None,
        gline_width=0.6,
        gline_alpha=0.9,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label="Spearman's ρ",
        colorbar_width_mm=3.0,
        colorbar_pad_mm=2.0,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        transparent=True,
    )


def run_workflow() -> tuple[Path, Path]:
    """Render the MER spike-scale correlation heatmap."""
    r, p_raw = load_correlation_matrices(CORRELATION_OUTPUT_PATH)
    r_plot, p_plot = prepare_heatmap_matrices(r, p_raw)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_correlation_heatmap(r_plot, p_plot)
    fig.savefig(HEATMAP_PDF_PATH, bbox_inches="tight", dpi=300)
    fig.savefig(HEATMAP_PNG_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return HEATMAP_PDF_PATH, HEATMAP_PNG_PATH


def _ordered_x_features(columns: list[str]) -> list[str]:
    ordered: list[str] = []
    for param in SPIKE_PARAM_ORDER:
        suffix = f"__{param}"
        matches = [column for column in columns if str(column).endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one correlation column ending with '{suffix}', "
                f"found {matches}."
            )
        ordered.append(matches[0])
    return ordered


if __name__ == "__main__":
    pdf_path, png_path = run_workflow()
    print(f"Saved heatmap PDF: {pdf_path}")
    print(f"Saved heatmap PNG: {png_path}")
