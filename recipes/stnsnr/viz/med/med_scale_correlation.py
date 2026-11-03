from __future__ import annotations

import argparse
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
from recipes.stnsnr.stats.med.med_scale_correlation import (
    BAND_ORDER,
    CONNECTIVITY_COMBINED_TREMOR_CORRELATION_OUTPUT_PATH,
    CONNECTIVITY_CORRELATION_OUTPUT_PATH,
    CONNECTIVITY_FEATURE_ORDER,
    CONNECTIVITY_SPECS,
    LOCAL_BAND_SPECS,
    LOCAL_FEATURE_ORDER_BY_REGION,
    OUTPUT_DIR,
    PARAM_ORDER,
    SCALE_COMBINED_TREMOR_DELTA_PERCENT_COLUMNS,
    SCALE_DELTA_PERCENT_COLUMNS,
    SNR_LOCAL_COMBINED_TREMOR_CORRELATION_OUTPUT_PATH,
    SNR_LOCAL_CORRELATION_OUTPUT_PATH,
    STN_LOCAL_COMBINED_TREMOR_CORRELATION_OUTPUT_PATH,
    STN_LOCAL_CORRELATION_OUTPUT_PATH,
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
    "delta_net_gc": "delta_net_gc-mean",
    "psi": "psi-mean",
}

HEATMAP_OUTPUTS = {
    "stn_local": {
        "correlation_path": STN_LOCAL_CORRELATION_OUTPUT_PATH,
        "feature_order": LOCAL_FEATURE_ORDER_BY_REGION["STN"],
        "x_labels": None,
        "label_top": "STN",
        "pdf_path": OUTPUT_DIR / "med_stn_local_scale_delta_percent_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "med_stn_local_scale_delta_percent_corr_heatmap.png",
        "scale_columns": SCALE_DELTA_PERCENT_COLUMNS,
    },
    "snr_local": {
        "correlation_path": SNR_LOCAL_CORRELATION_OUTPUT_PATH,
        "feature_order": LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
        "x_labels": None,
        "label_top": "SNr",
        "pdf_path": OUTPUT_DIR / "med_snr_local_scale_delta_percent_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "med_snr_local_scale_delta_percent_corr_heatmap.png",
        "scale_columns": SCALE_DELTA_PERCENT_COLUMNS,
    },
    "connectivity": {
        "correlation_path": CONNECTIVITY_CORRELATION_OUTPUT_PATH,
        "feature_order": CONNECTIVITY_FEATURE_ORDER,
        "x_labels": None,
        "label_top": "SNr-STN / SNr→STN",
        "pdf_path": OUTPUT_DIR / "med_connectivity_scale_delta_percent_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "med_connectivity_scale_delta_percent_corr_heatmap.png",
        "scale_columns": SCALE_DELTA_PERCENT_COLUMNS,
    },
    "stn_local_combined_tremor": {
        "correlation_path": STN_LOCAL_COMBINED_TREMOR_CORRELATION_OUTPUT_PATH,
        "feature_order": LOCAL_FEATURE_ORDER_BY_REGION["STN"],
        "x_labels": None,
        "label_top": "STN",
        "pdf_path": OUTPUT_DIR / "med_stn_local_scale_combined_tremor_delta_percent_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "med_stn_local_scale_combined_tremor_delta_percent_corr_heatmap.png",
        "scale_columns": SCALE_COMBINED_TREMOR_DELTA_PERCENT_COLUMNS,
    },
    "snr_local_combined_tremor": {
        "correlation_path": SNR_LOCAL_COMBINED_TREMOR_CORRELATION_OUTPUT_PATH,
        "feature_order": LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
        "x_labels": None,
        "label_top": "SNr",
        "pdf_path": OUTPUT_DIR / "med_snr_local_scale_combined_tremor_delta_percent_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "med_snr_local_scale_combined_tremor_delta_percent_corr_heatmap.png",
        "scale_columns": SCALE_COMBINED_TREMOR_DELTA_PERCENT_COLUMNS,
    },
    "connectivity_combined_tremor": {
        "correlation_path": CONNECTIVITY_COMBINED_TREMOR_CORRELATION_OUTPUT_PATH,
        "feature_order": CONNECTIVITY_FEATURE_ORDER,
        "x_labels": None,
        "label_top": "SNr-STN / SNr→STN",
        "pdf_path": OUTPUT_DIR / "med_connectivity_scale_combined_tremor_delta_percent_corr_heatmap.pdf",
        "png_path": OUTPUT_DIR / "med_connectivity_scale_combined_tremor_delta_percent_corr_heatmap.png",
        "scale_columns": SCALE_COMBINED_TREMOR_DELTA_PERCENT_COLUMNS,
    },
}

COMBINED_TREMOR_HEATMAP_KEYS = (
    "stn_local_combined_tremor",
    "snr_local_combined_tremor",
    "connectivity_combined_tremor",
)

Y_AXIS_LABELS = {
    "LCT improvement rate": "LCT Δ%",
    "MDS-UPDRS III Δ%": "UPDRS III Δ%",
    "MDS-UPDRS III axial Δ%": "Axial Δ%",
    "MDS-UPDRS III rigidity Δ%": "Rigidity Δ%",
    "MDS-UPDRS III bradykinesia Δ%": "Bradykinesia Δ%",
    "MDS-UPDRS III rest tremor Δ%": "Rest tremor Δ%",
    "MDS-UPDRS III action tremor Δ%": "Action tremor Δ%",
    "MDS-UPDRS III tremor Δ%": "Tremor Δ%",
}


def make_local_x_labels() -> list[str]:
    """Return local heatmap labels using unitless visual default templates."""
    return [
        _medication_effect_label(_format_unitless_property_label(LOCAL_PARAM_TYPE_BY_KEY[spec.key], band))
        for band in BAND_ORDER
        for spec in LOCAL_BAND_SPECS
    ] + [_medication_effect_label(param) for param in PARAM_ORDER]


def make_connectivity_x_labels() -> list[str]:
    """Return connectivity heatmap labels using unitless visual default templates."""
    return [
        _medication_effect_label(_format_unitless_property_label(CONNECTIVITY_PARAM_TYPE_BY_KEY[spec.key], band))
        for band in BAND_ORDER
        for spec in CONNECTIVITY_SPECS
    ]


def _format_unitless_property_label(param_type: str, band: str) -> str:
    band_label = cfg.BAND_TPL[band]
    label = cfg.format_label(param_type, prefix="", key=band_label)
    label = UNIT_SUFFIX_RE.sub("", label)
    return " ".join(label.split())


def _medication_effect_label(label: object) -> str:
    text = str(label)
    return text if text.startswith("Δ") else f"Δ{text}"


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
    scale_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reorder matrices and replace internal feature keys with display labels."""
    missing_y = [column for column in scale_columns if column not in r.index]
    if missing_y:
        raise ValueError(f"Correlation matrix is missing y-axis rows: {missing_y}.")

    missing_x = [column for column in feature_order if column not in r.columns]
    if missing_x:
        raise ValueError(f"Correlation matrix is missing x-axis columns: {missing_x}.")

    if len(feature_order) != len(x_labels):
        raise ValueError("Feature order and x-label lists must have the same length.")

    r_plot = r.loc[scale_columns, feature_order].copy()
    p_plot = p_raw.loc[scale_columns, feature_order].copy()

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
    """Create one medication LFP-scale delta-percent correlation heatmap."""
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
    """Render all medication LFP-scale delta-percent correlation heatmaps."""
    return _render_heatmaps(HEATMAP_OUTPUTS)


def run_combined_tremor_workflow() -> dict[str, tuple[Path, Path]]:
    """Render only the additional combined-tremor heatmaps."""
    selected_outputs = {
        key: HEATMAP_OUTPUTS[key]
        for key in COMBINED_TREMOR_HEATMAP_KEYS
    }
    return _render_heatmaps(selected_outputs)


def _render_heatmaps(
    heatmap_outputs: dict[str, dict[str, object]],
) -> dict[str, tuple[Path, Path]]:
    """Render the selected medication correlation heatmaps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_label_map = {
        "stn_local": make_local_x_labels(),
        "snr_local": make_local_x_labels(),
        "connectivity": make_connectivity_x_labels(),
        "stn_local_combined_tremor": make_local_x_labels(),
        "snr_local_combined_tremor": make_local_x_labels(),
        "connectivity_combined_tremor": make_connectivity_x_labels(),
    }

    outputs: dict[str, tuple[Path, Path]] = {}
    for group_name, spec in heatmap_outputs.items():
        x_labels = x_label_map[group_name]
        r, p_raw = load_correlation_matrices(spec["correlation_path"])
        r_plot, p_plot = prepare_heatmap_matrices(
            r,
            p_raw,
            feature_order=spec["feature_order"],
            x_labels=x_labels,
            scale_columns=spec["scale_columns"],
        )
        _validate_plot_matrices(
            group_name,
            r_plot,
            p_plot,
            x_labels,
            scale_columns=spec["scale_columns"],
        )

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
    scale_columns: list[str],
) -> None:
    if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
        raise ValueError(f"{group_name}: r and p_raw plot matrices are not aligned.")
    if r_plot.columns.tolist() != x_labels:
        raise ValueError(f"{group_name}: x-axis labels do not match the configured order.")
    if r_plot.shape[0] != len(scale_columns):
        raise ValueError(
            f"{group_name}: expected {len(scale_columns)} y-axis rows; "
            f"got {r_plot.shape[0]}."
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for selective heatmap rendering."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combined-tremor-only",
        action="store_true",
        help="Render only the additional combined-tremor heatmaps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    heatmap_outputs = (
        run_combined_tremor_workflow()
        if args.combined_tremor_only
        else run_workflow()
    )
    for group, (pdf_path, png_path) in heatmap_outputs.items():
        print(f"{group}: saved heatmap PDF: {pdf_path}")
        print(f"{group}: saved heatmap PNG: {png_path}")
