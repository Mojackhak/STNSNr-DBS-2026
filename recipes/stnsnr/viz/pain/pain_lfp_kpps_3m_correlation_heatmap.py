from __future__ import annotations

from dataclasses import dataclass
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
from recipes.stnsnr.stats.pain.pain_lfp_kpps_3m_correlation import (
    BAND_ORDER,
    CONNECTIVITY_CORRELATION_OUTPUT_PATH,
    CONNECTIVITY_FEATURE_ORDER,
    CONNECTIVITY_SPECS,
    KPPS_COLUMNS,
    LOCAL_BAND_SPECS,
    LOCAL_FEATURE_ORDER_BY_REGION,
    OUTPUT_DIR,
    PARAM_ORDER,
    PHASE_ORDER,
    SNR_LOCAL_CORRELATION_OUTPUT_PATH,
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

COLORBAR_LABEL = "Spearman's ρ"
CELL_SIZE_MM = (3.0, 3.0)
GRID_LINE_COLOR = "black"
GRID_LINE_WIDTH = 0.6
GRID_LINE_ALPHA = 0.9
VALUE_LIMIT = 1.0
PANEL_ORDER = tuple(PHASE_ORDER)
KPPS_IMPROVEMENT_COLUMNS = [
    "KPPS Δ% (STN, 3m)",
    "ΔKPPS Δ% (+SNr, 3m)",
]


@dataclass(frozen=True)
class HeatmapOutputSpec:
    correlation_path: Path
    feature_order: list[str]
    x_labels: list[str]
    y_labels: list[str]
    label_top: str
    pdf_path: Path
    png_path: Path


def make_local_x_labels() -> list[str]:
    """Return local heatmap labels using unitless visual default templates."""
    return [
        _pain_change_label(_format_unitless_property_label(LOCAL_PARAM_TYPE_BY_KEY[spec.key], band))
        for band in BAND_ORDER
        for spec in LOCAL_BAND_SPECS
    ] + [_pain_change_label(param) for param in PARAM_ORDER]


def make_connectivity_x_labels() -> list[str]:
    """Return connectivity heatmap labels using unitless visual default templates."""
    return [
        _pain_change_label(_format_unitless_property_label(CONNECTIVITY_PARAM_TYPE_BY_KEY[spec.key], band))
        for band in BAND_ORDER
        for spec in CONNECTIVITY_SPECS
    ]


def _format_unitless_property_label(param_type: str, band: str) -> str:
    band_label = cfg.BAND_TPL[band]
    label = cfg.format_label(param_type, prefix="", key=band_label)
    label = UNIT_SUFFIX_RE.sub("", label)
    return " ".join(label.split())


def _pain_change_label(label: object) -> str:
    text = str(label)
    return text if text.startswith("Δ") else f"Δ{text}"


def build_heatmap_outputs() -> dict[str, HeatmapOutputSpec]:
    local_x_labels = make_local_x_labels()
    connectivity_x_labels = make_connectivity_x_labels()
    return {
        "stn_local": HeatmapOutputSpec(
            correlation_path=STN_LOCAL_CORRELATION_OUTPUT_PATH,
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["STN"],
            x_labels=local_x_labels,
            y_labels=KPPS_COLUMNS,
            label_top="STN",
            pdf_path=OUTPUT_DIR / "pain_stn_local_kpps_3m_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "pain_stn_local_kpps_3m_corr_heatmap.png",
        ),
        "snr_local": HeatmapOutputSpec(
            correlation_path=SNR_LOCAL_CORRELATION_OUTPUT_PATH,
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
            x_labels=local_x_labels,
            y_labels=KPPS_COLUMNS,
            label_top="SNr",
            pdf_path=OUTPUT_DIR / "pain_snr_local_kpps_3m_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "pain_snr_local_kpps_3m_corr_heatmap.png",
        ),
        "connectivity": HeatmapOutputSpec(
            correlation_path=CONNECTIVITY_CORRELATION_OUTPUT_PATH,
            feature_order=CONNECTIVITY_FEATURE_ORDER,
            x_labels=connectivity_x_labels,
            y_labels=KPPS_COLUMNS,
            label_top="SNr-STN / SNr→STN",
            pdf_path=OUTPUT_DIR / "pain_connectivity_kpps_3m_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "pain_connectivity_kpps_3m_corr_heatmap.png",
        ),
    }


HEATMAP_OUTPUTS = build_heatmap_outputs()


def build_improvement_heatmap_outputs() -> dict[str, HeatmapOutputSpec]:
    local_x_labels = make_local_x_labels()
    connectivity_x_labels = make_connectivity_x_labels()
    return {
        "stn_local_delta_percent": HeatmapOutputSpec(
            correlation_path=STN_LOCAL_CORRELATION_OUTPUT_PATH,
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["STN"],
            x_labels=local_x_labels,
            y_labels=KPPS_IMPROVEMENT_COLUMNS,
            label_top="STN",
            pdf_path=OUTPUT_DIR / "pain_stn_local_kpps_3m_delta_percent_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "pain_stn_local_kpps_3m_delta_percent_corr_heatmap.png",
        ),
        "snr_local_delta_percent": HeatmapOutputSpec(
            correlation_path=SNR_LOCAL_CORRELATION_OUTPUT_PATH,
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
            x_labels=local_x_labels,
            y_labels=KPPS_IMPROVEMENT_COLUMNS,
            label_top="SNr",
            pdf_path=OUTPUT_DIR / "pain_snr_local_kpps_3m_delta_percent_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "pain_snr_local_kpps_3m_delta_percent_corr_heatmap.png",
        ),
        "connectivity_delta_percent": HeatmapOutputSpec(
            correlation_path=CONNECTIVITY_CORRELATION_OUTPUT_PATH,
            feature_order=CONNECTIVITY_FEATURE_ORDER,
            x_labels=connectivity_x_labels,
            y_labels=KPPS_IMPROVEMENT_COLUMNS,
            label_top="SNr-STN / SNr→STN",
            pdf_path=OUTPUT_DIR / "pain_connectivity_kpps_3m_delta_percent_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "pain_connectivity_kpps_3m_delta_percent_corr_heatmap.png",
        ),
    }


IMPROVEMENT_HEATMAP_OUTPUTS = build_improvement_heatmap_outputs()
ALL_HEATMAP_OUTPUTS = {
    **HEATMAP_OUTPUTS,
    **IMPROVEMENT_HEATMAP_OUTPUTS,
}


def load_correlation_matrices_by_phase(
    correlation_path: str | Path,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load phase-specific r and raw p-value matrices from a workbook."""
    correlation_path = Path(correlation_path)
    if not correlation_path.exists():
        raise FileNotFoundError(
            f"Missing correlation workbook: {correlation_path}. "
            "Run recipes/stnsnr/stats/pain/pain_lfp_kpps_3m_correlation.py first."
        )

    matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for phase in PANEL_ORDER:
        r = pd.read_excel(correlation_path, sheet_name=f"{phase}_r", index_col=0)
        p_raw = pd.read_excel(correlation_path, sheet_name=f"{phase}_p_raw", index_col=0)
        matrices[phase] = (r, p_raw)

    return matrices


def prepare_heatmap_matrices_by_phase(
    matrices_by_phase: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    feature_order: list[str],
    x_labels: list[str],
    y_labels: list[str],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Reorder phase matrices and replace internal feature keys with display labels."""
    if len(feature_order) != len(x_labels):
        raise ValueError("Feature order and x-label lists must have the same length.")
    if not y_labels:
        raise ValueError("At least one y-axis label is required.")

    r_plots: list[pd.DataFrame] = []
    p_plots: list[pd.DataFrame] = []
    for phase in PANEL_ORDER:
        r, p_raw = matrices_by_phase[phase]
        r_plot, p_plot = prepare_heatmap_matrices(
            r,
            p_raw,
            feature_order=feature_order,
            x_labels=x_labels,
            y_labels=y_labels,
            phase=phase,
        )
        r_plots.append(r_plot)
        p_plots.append(p_plot)

    return r_plots, p_plots


def prepare_heatmap_matrices(
    r: pd.DataFrame,
    p_raw: pd.DataFrame,
    *,
    feature_order: list[str],
    x_labels: list[str],
    y_labels: list[str],
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one phase panel for plotting."""
    missing_y = [column for column in y_labels if column not in r.index]
    if missing_y:
        raise ValueError(f"{phase}: correlation matrix is missing y-axis rows: {missing_y}.")

    missing_x = [column for column in feature_order if column not in r.columns]
    if missing_x:
        raise ValueError(f"{phase}: correlation matrix is missing x-axis columns: {missing_x}.")

    r_plot = r.loc[y_labels, feature_order].copy()
    p_plot = p_raw.loc[y_labels, feature_order].copy()

    if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
        raise ValueError(f"{phase}: r and p_raw matrices are not aligned after reordering.")

    rename_map = dict(zip(feature_order, x_labels, strict=True))
    r_plot = r_plot.rename(columns=rename_map)
    p_plot = p_plot.rename(columns=rename_map)
    return r_plot, p_plot


def plot_layered_correlation_heatmap(
    r_plots: list[pd.DataFrame],
    p_plots: list[pd.DataFrame],
    *,
    label_top: str,
) -> plt.Figure:
    """Create one layered pain LFP-KPPS correlation heatmap."""
    if len(r_plots) != len(PANEL_ORDER):
        raise ValueError(f"Expected {len(PANEL_ORDER)} r matrices; got {len(r_plots)}.")
    if len(p_plots) != len(PANEL_ORDER):
        raise ValueError(f"Expected {len(PANEL_ORDER)} p matrices; got {len(p_plots)}.")
    if not r_plots:
        raise ValueError("At least one phase panel is required.")

    return vdf.plot_well_heatmap_grid_df(
        df_values=[[r_plot] for r_plot in r_plots],
        df_ps=[[p_plot] for p_plot in p_plots],
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=[label_top],
        label_right=list(PANEL_ORDER),
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=CELL_SIZE_MM,
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


def render_group_heatmap(output: HeatmapOutputSpec) -> tuple[Path, Path]:
    """Render one analysis-group heatmap."""
    matrices_by_phase = load_correlation_matrices_by_phase(output.correlation_path)
    r_plots, p_plots = prepare_heatmap_matrices_by_phase(
        matrices_by_phase,
        feature_order=output.feature_order,
        x_labels=output.x_labels,
        y_labels=output.y_labels,
    )
    _validate_plot_matrices(output.label_top, r_plots, p_plots, output.x_labels, output.y_labels)

    fig = plot_layered_correlation_heatmap(r_plots, p_plots, label_top=output.label_top)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return output.pdf_path, output.png_path


def run_workflow() -> dict[str, tuple[Path, Path]]:
    """Render all pain LFP-KPPS correlation heatmaps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[Path, Path]] = {}
    for group_name, spec in ALL_HEATMAP_OUTPUTS.items():
        outputs[group_name] = render_group_heatmap(spec)

    return outputs


def _validate_plot_matrices(
    group_name: str,
    r_plots: list[pd.DataFrame],
    p_plots: list[pd.DataFrame],
    x_labels: list[str],
    y_labels: list[str],
) -> None:
    expected_shape = (len(y_labels), len(x_labels))
    for phase, r_plot, p_plot in zip(PANEL_ORDER, r_plots, p_plots, strict=True):
        if r_plot.shape != expected_shape:
            raise ValueError(f"{group_name} {phase}: expected r plot shape {expected_shape}; got {r_plot.shape}.")
        if p_plot.shape != expected_shape:
            raise ValueError(f"{group_name} {phase}: expected p plot shape {expected_shape}; got {p_plot.shape}.")
        if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
            raise ValueError(f"{group_name} {phase}: r and p plot matrices are not aligned.")
        if r_plot.index.tolist() != y_labels:
            raise ValueError(f"{group_name} {phase}: y-axis labels do not match the configured KPPS order.")
        if r_plot.columns.tolist() != x_labels:
            raise ValueError(f"{group_name} {phase}: x-axis labels do not match the configured LFP order.")


if __name__ == "__main__":
    run_workflow()
