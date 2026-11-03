from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plugin.default.viz_defaults as cfg
import src.viz.visualdf as vdf
from recipes.stnsnr.stats.rest.rest_scale_correlation import (
    BAND_ORDER,
    CONNECTIVITY_CORRELATION_OUTPUT_PATH,
    CONNECTIVITY_FEATURE_ORDER,
    CONNECTIVITY_SPECS,
    LOCAL_BAND_SPECS,
    LOCAL_FEATURE_ORDER_BY_REGION,
    OUTPUT_DIR,
    PARAM_ORDER,
    SNR_LOCAL_CORRELATION_OUTPUT_PATH,
    STN_LOCAL_CORRELATION_OUTPUT_PATH,
    Y_COLUMNS,
)


UNIT_SUFFIX_RE = re.compile(r"\s+\((?:dB|Hz|s|V|a\.u\.)\)\s*$")

LOCAL_PARAM_TYPE_BY_KEY = {
    "raw": "raw-mean",
    "period": "period-mean",
    "burst_duration": "burst-duration",
    "burst_mean": "burst-mean",
    "burst_occupation": "burst-occupation",
    "burst_rate": "burst-rate",
}

CONNECTIVITY_PARAM_TYPE_BY_KEY = {
    "coh": "coh-mean",
    "wpli": "wpli-mean",
    "ciplv": "ciplv-mean",
    "gc": "delta_net_gc-mean",
    "psi": "psi-mean",
}


def make_local_x_labels() -> list[str]:
    """Return local heatmap labels using unitless visual default templates."""
    return [
        _format_unitless_property_label(LOCAL_PARAM_TYPE_BY_KEY[spec.key], band)
        for band in BAND_ORDER
        for spec in LOCAL_BAND_SPECS
    ] + PARAM_ORDER.copy()


def make_connectivity_x_labels() -> list[str]:
    """Return connectivity heatmap labels using unitless visual default templates."""
    return [
        _format_unitless_property_label(CONNECTIVITY_PARAM_TYPE_BY_KEY[spec.key], band)
        for band in BAND_ORDER
        for spec in CONNECTIVITY_SPECS
    ]


def _format_unitless_property_label(param_type: str, band: str) -> str:
    band_label = cfg.BAND_TPL[band]
    label = cfg.format_label(param_type, prefix="", key=band_label)
    label = UNIT_SUFFIX_RE.sub("", label)
    return " ".join(label.split())


HEATMAP_OUTPUTS = {
    "stn_local": {
        "correlation_path": STN_LOCAL_CORRELATION_OUTPUT_PATH,
        "feature_order": LOCAL_FEATURE_ORDER_BY_REGION["STN"],
        "x_labels": make_local_x_labels(),
        "label_top": "STN",
        "pdf_path": OUTPUT_DIR / "rest_stn_local_scale_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "rest_stn_local_scale_corr_heatmap.png",
    },
    "snr_local": {
        "correlation_path": SNR_LOCAL_CORRELATION_OUTPUT_PATH,
        "feature_order": LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
        "x_labels": make_local_x_labels(),
        "label_top": "SNr",
        "pdf_path": OUTPUT_DIR / "rest_snr_local_scale_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "rest_snr_local_scale_corr_heatmap.png",
    },
    "connectivity": {
        "correlation_path": CONNECTIVITY_CORRELATION_OUTPUT_PATH,
        "feature_order": CONNECTIVITY_FEATURE_ORDER,
        "x_labels": make_connectivity_x_labels(),
        "label_top": "SNr-STN / SNr→STN",
        "pdf_path": OUTPUT_DIR / "rest_connectivity_scale_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "rest_connectivity_scale_corr_heatmap.png",
    },
}

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
    correlation_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load r and raw p-value matrices from a correlation workbook."""
    correlation_path = Path(correlation_path)
    r = pd.read_excel(correlation_path, sheet_name="r", index_col=0)
    p_raw = pd.read_excel(correlation_path, sheet_name="p_raw", index_col=0)
    return r, p_raw


def prepare_heatmap_matrices(
    r: pd.DataFrame,
    p_raw: pd.DataFrame,
    feature_order: list[str],
    x_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reorder matrices and replace internal feature keys with display labels."""
    missing_y = [column for column in Y_COLUMNS if column not in r.index]
    if missing_y:
        raise ValueError(f"Correlation matrix is missing y-axis rows: {missing_y}.")

    missing_x = [column for column in feature_order if column not in r.columns]
    if missing_x:
        raise ValueError(f"Correlation matrix is missing x-axis columns: {missing_x}.")

    if len(feature_order) != len(x_labels):
        raise ValueError("Feature order and x-label lists must have the same length.")

    r_plot = r.loc[Y_COLUMNS, feature_order].copy()
    p_plot = p_raw.loc[Y_COLUMNS, feature_order].copy()

    if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
        raise ValueError("r and p_raw matrices are not aligned after reordering.")

    rename_map = dict(zip(feature_order, x_labels, strict=True))
    r_plot = r_plot.rename(columns=rename_map, index=Y_AXIS_LABELS)
    p_plot = p_plot.rename(columns=rename_map, index=Y_AXIS_LABELS)
    return r_plot, p_plot


def plot_correlation_heatmap(
    r_plot: pd.DataFrame,
    p_plot: pd.DataFrame,
    label_top: str,
) -> plt.Figure:
    """Create one rest scalar-scale correlation heatmap."""
    return vdf.plot_well_heatmap_df(
        df_value=r_plot,
        df_p=p_plot,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=label_top,
        vmin=-1,
        vmax=1,
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


def run_workflow() -> dict[str, tuple[Path, Path]]:
    """Render all rest scalar-scale correlation heatmaps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[Path, Path]] = {}
    for group_name, spec in HEATMAP_OUTPUTS.items():
        r, p_raw = load_correlation_matrices(spec["correlation_path"])
        r_plot, p_plot = prepare_heatmap_matrices(
            r,
            p_raw,
            feature_order=spec["feature_order"],
            x_labels=spec["x_labels"],
        )
        _validate_plot_matrices(group_name, r_plot, p_plot, spec["x_labels"])

        fig = plot_correlation_heatmap(r_plot, p_plot, label_top=spec["label_top"])
        fig.savefig(spec["pdf_path"], bbox_inches="tight", dpi=300)
        fig.savefig(spec["png_path"], bbox_inches="tight", dpi=300)
        plt.close(fig)

        outputs[group_name] = (spec["pdf_path"], spec["png_path"])

    return outputs


def _validate_plot_matrices(
    group_name: str,
    r_plot: pd.DataFrame,
    p_plot: pd.DataFrame,
    x_labels: list[str],
) -> None:
    if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
        raise ValueError(f"{group_name}: r and p_raw plot matrices are not aligned.")
    if r_plot.columns.tolist() != x_labels:
        raise ValueError(f"{group_name}: x-axis labels do not match the configured order.")
    if r_plot.shape[0] != len(Y_COLUMNS):
        raise ValueError(f"{group_name}: expected {len(Y_COLUMNS)} y-axis rows; got {r_plot.shape[0]}.")


if __name__ == "__main__":
    heatmap_outputs = run_workflow()
    for group, (pdf_path, png_path) in heatmap_outputs.items():
        print(f"{group}: saved heatmap PDF: {pdf_path}")
        print(f"{group}: saved heatmap PNG: {png_path}")
