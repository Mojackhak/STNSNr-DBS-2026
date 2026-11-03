from __future__ import annotations

import os

from collections.abc import Sequence
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
from recipes.stnsnr.stats.walk.walk_lfp_gait_pca_3m_correlation import (
    BAND_ORDER,
    CONNECTIVITY_FEATURE_ORDER,
    CONNECTIVITY_SPECS,
    GAIT_PCA_COLUMNS,
    LOCAL_BAND_SPECS,
    LOCAL_FEATURE_ORDER_BY_REGION,
    PARAM_ORDER,
    POSE_ORDER,
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
PANEL_ORDER = tuple(POSE_ORDER)
CORRELATION_DIR = Path(
    os.path.join(os.environ["MYLFP_PROJECT_ROOT"], 'summary/stats/walk/corr/walk_lfp_gait_pca_3m')
)

COMPACT_GAIT_PCA_ROWS = (
    "ΔGait PCA STN vs Pre-op",
    "ΔGait PCA STN+SNr vs STN",
)
COMPACT_GAIT_PCA_LABELS = (
    "Gait PCA Δ% (STN, 3m)",
    "Gait PCA Δ% (SNr, 3m)",
)


@dataclass(frozen=True)
class HeatmapOutputSpec:
    correlation_path: Path
    feature_order: list[str]
    x_labels: list[str]
    label_top: str
    pdf_path: Path
    png_path: Path


@dataclass(frozen=True)
class HeatmapVariantSpec:
    source_rows: tuple[str, ...]
    y_labels: tuple[str, ...]
    output_suffix: str


HEATMAP_VARIANTS = {
    "full": HeatmapVariantSpec(
        source_rows=tuple(GAIT_PCA_COLUMNS),
        y_labels=tuple(GAIT_PCA_COLUMNS),
        output_suffix="",
    ),
    "stn_snr_effects": HeatmapVariantSpec(
        source_rows=COMPACT_GAIT_PCA_ROWS,
        y_labels=COMPACT_GAIT_PCA_LABELS,
        output_suffix="_stn_snr_effects",
    ),
}


def make_local_x_labels() -> list[str]:
    """Return local heatmap labels using unitless visual default templates."""
    return [
        _walk_change_label(_format_unitless_property_label(LOCAL_PARAM_TYPE_BY_KEY[spec.key], band))
        for band in BAND_ORDER
        for spec in LOCAL_BAND_SPECS
    ] + [_walk_change_label(param) for param in PARAM_ORDER]


def make_connectivity_x_labels() -> list[str]:
    """Return connectivity heatmap labels using unitless visual default templates."""
    return [
        _walk_change_label(_format_unitless_property_label(CONNECTIVITY_PARAM_TYPE_BY_KEY[spec.key], band))
        for band in BAND_ORDER
        for spec in CONNECTIVITY_SPECS
    ]


def _format_unitless_property_label(param_type: str, band: str) -> str:
    band_label = cfg.BAND_TPL[band]
    label = cfg.format_label(param_type, prefix="", key=band_label)
    label = UNIT_SUFFIX_RE.sub("", label)
    return " ".join(label.split())


def _walk_change_label(label: object) -> str:
    text = str(label)
    return text if text.startswith("Δ") else f"Δ{text}"


def build_heatmap_outputs() -> dict[str, HeatmapOutputSpec]:
    local_x_labels = make_local_x_labels()
    connectivity_x_labels = make_connectivity_x_labels()
    return {
        "stn_local": HeatmapOutputSpec(
            correlation_path=CORRELATION_DIR
            / "walk_single_stn_local_gait_pca_3m_corr_by_pose.xlsx",
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["STN"],
            x_labels=local_x_labels,
            label_top="STN",
            pdf_path=CORRELATION_DIR
            / "walk_single_stn_local_gait_pca_3m_corr_heatmap.pdf",
            png_path=CORRELATION_DIR
            / "walk_single_stn_local_gait_pca_3m_corr_heatmap.png",
        ),
        "snr_local": HeatmapOutputSpec(
            correlation_path=CORRELATION_DIR
            / "walk_single_snr_local_gait_pca_3m_corr_by_pose.xlsx",
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
            x_labels=local_x_labels,
            label_top="SNr",
            pdf_path=CORRELATION_DIR
            / "walk_single_snr_local_gait_pca_3m_corr_heatmap.pdf",
            png_path=CORRELATION_DIR
            / "walk_single_snr_local_gait_pca_3m_corr_heatmap.png",
        ),
        "connectivity": HeatmapOutputSpec(
            correlation_path=CORRELATION_DIR
            / "walk_single_connectivity_gait_pca_3m_corr_by_pose.xlsx",
            feature_order=CONNECTIVITY_FEATURE_ORDER,
            x_labels=connectivity_x_labels,
            label_top="SNr-STN / SNr→STN",
            pdf_path=CORRELATION_DIR
            / "walk_single_connectivity_gait_pca_3m_corr_heatmap.pdf",
            png_path=CORRELATION_DIR
            / "walk_single_connectivity_gait_pca_3m_corr_heatmap.png",
        ),
    }


HEATMAP_OUTPUTS = build_heatmap_outputs()


def load_correlation_matrices_by_pose(
    correlation_path: str | Path,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load pose-specific r and raw p-value matrices from a workbook."""
    correlation_path = Path(correlation_path)
    if not correlation_path.exists():
        raise FileNotFoundError(
            f"Missing correlation workbook: {correlation_path}. "
            "Run recipes/stnsnr/stats/walk/walk_lfp_gait_pca_3m_correlation.py first."
        )

    matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for pose in PANEL_ORDER:
        r = pd.read_excel(correlation_path, sheet_name=f"{pose}_r", index_col=0)
        p_raw = pd.read_excel(correlation_path, sheet_name=f"{pose}_p_raw", index_col=0)
        matrices[pose] = (r, p_raw)

    return matrices


def prepare_heatmap_matrices_by_pose(
    matrices_by_pose: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    feature_order: list[str],
    x_labels: list[str],
    source_rows: Sequence[str],
    y_labels: Sequence[str],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Reorder pose matrices and replace internal feature keys with display labels."""
    if len(feature_order) != len(x_labels):
        raise ValueError("Feature order and x-label lists must have the same length.")
    if len(source_rows) != len(y_labels):
        raise ValueError("Source-row and y-label lists must have the same length.")

    r_plots: list[pd.DataFrame] = []
    p_plots: list[pd.DataFrame] = []
    for pose in PANEL_ORDER:
        r, p_raw = matrices_by_pose[pose]
        r_plot, p_plot = prepare_heatmap_matrices(
            r,
            p_raw,
            feature_order=feature_order,
            x_labels=x_labels,
            source_rows=source_rows,
            y_labels=y_labels,
            pose=pose,
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
    source_rows: Sequence[str],
    y_labels: Sequence[str],
    pose: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one pose panel for plotting."""
    missing_y = [column for column in source_rows if column not in r.index]
    if missing_y:
        raise ValueError(f"{pose}: correlation matrix is missing y-axis rows: {missing_y}.")

    missing_x = [column for column in feature_order if column not in r.columns]
    if missing_x:
        raise ValueError(f"{pose}: correlation matrix is missing x-axis columns: {missing_x}.")

    r_plot = r.loc[list(source_rows), feature_order].copy()
    p_plot = p_raw.loc[list(source_rows), feature_order].copy()

    if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
        raise ValueError(f"{pose}: r and p_raw matrices are not aligned after reordering.")

    rename_map = dict(zip(feature_order, x_labels, strict=True))
    y_rename_map = dict(zip(source_rows, y_labels, strict=True))
    r_plot = r_plot.rename(index=y_rename_map, columns=rename_map)
    p_plot = p_plot.rename(index=y_rename_map, columns=rename_map)
    return r_plot, p_plot


def plot_layered_correlation_heatmap(
    r_plots: list[pd.DataFrame],
    p_plots: list[pd.DataFrame],
    *,
    label_top: str,
) -> plt.Figure:
    """Create one layered walk LFP-gait PCA correlation heatmap."""
    if len(r_plots) != len(PANEL_ORDER):
        raise ValueError(f"Expected {len(PANEL_ORDER)} r matrices; got {len(r_plots)}.")
    if len(p_plots) != len(PANEL_ORDER):
        raise ValueError(f"Expected {len(PANEL_ORDER)} p matrices; got {len(p_plots)}.")
    if not r_plots:
        raise ValueError("At least one pose panel is required.")

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


def output_paths_for_variant(
    output: HeatmapOutputSpec,
    variant: HeatmapVariantSpec,
) -> tuple[Path, Path]:
    """Return PDF and PNG paths for one heatmap variant."""
    if not variant.output_suffix:
        return output.pdf_path, output.png_path
    return tuple(
        path.with_name(f"{path.stem}{variant.output_suffix}{path.suffix}")
        for path in (output.pdf_path, output.png_path)
    )


def render_group_heatmaps(
    output: HeatmapOutputSpec,
) -> dict[str, tuple[Path, Path]]:
    """Render all variants for one analysis group from one workbook load."""
    matrices_by_pose = load_correlation_matrices_by_pose(output.correlation_path)
    rendered: dict[str, tuple[Path, Path]] = {}
    for variant_name, variant in HEATMAP_VARIANTS.items():
        rendered[variant_name] = render_heatmap_variant(
            output,
            matrices_by_pose,
            variant,
        )
    return rendered


def render_heatmap_variant(
    output: HeatmapOutputSpec,
    matrices_by_pose: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    variant: HeatmapVariantSpec,
) -> tuple[Path, Path]:
    """Render one row-selection variant from already loaded matrices."""
    r_plots, p_plots = prepare_heatmap_matrices_by_pose(
        matrices_by_pose,
        feature_order=output.feature_order,
        x_labels=output.x_labels,
        source_rows=variant.source_rows,
        y_labels=variant.y_labels,
    )

    fig = plot_layered_correlation_heatmap(
        r_plots,
        p_plots,
        label_top=output.label_top,
    )
    vdf._ensure_strip_background_opaque(fig)
    pdf_path, png_path = output_paths_for_variant(output, variant)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return pdf_path, png_path


def run_workflow() -> dict[str, dict[str, tuple[Path, Path]]]:
    """Render all walk LFP-gait PCA 3-month correlation heatmaps."""
    CORRELATION_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, tuple[Path, Path]]] = {}
    for group_name, output in HEATMAP_OUTPUTS.items():
        outputs[group_name] = render_group_heatmaps(output)

    validate_output_files(outputs)
    return outputs


def validate_output_files(
    outputs: dict[str, dict[str, tuple[Path, Path]]],
) -> None:
    """Validate the expected heatmap files and matrix dimensions."""
    if set(outputs) != set(HEATMAP_OUTPUTS):
        raise ValueError(f"Expected output keys {sorted(HEATMAP_OUTPUTS)}; got {sorted(outputs)}.")

    for group_name, variants in outputs.items():
        if set(variants) != set(HEATMAP_VARIANTS):
            raise ValueError(
                f"{group_name}: expected variants {sorted(HEATMAP_VARIANTS)}; "
                f"got {sorted(variants)}."
            )
        for variant_name, output_paths in variants.items():
            for output_path in output_paths:
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"{group_name} {variant_name}: missing heatmap output {output_path}."
                    )

        spec = HEATMAP_OUTPUTS[group_name]
        matrices_by_pose = load_correlation_matrices_by_pose(spec.correlation_path)
        for variant_name, variant in HEATMAP_VARIANTS.items():
            expected_shape = (len(variant.source_rows), len(spec.feature_order))
            for pose, (r, p_raw) in matrices_by_pose.items():
                actual = r.loc[list(variant.source_rows), spec.feature_order].shape
                if actual != expected_shape:
                    raise ValueError(
                        f"{group_name} {variant_name} {pose}: expected matrix shape "
                        f"{expected_shape}; got {actual}."
                    )
                p_actual = p_raw.loc[list(variant.source_rows), spec.feature_order].shape
                if p_actual != expected_shape:
                    raise ValueError(
                        f"{group_name} {variant_name} {pose}: expected p_raw matrix "
                        f"shape {expected_shape}; got {p_actual}."
                    )


if __name__ == "__main__":
    workflow_outputs = run_workflow()
    for group_name, variants in workflow_outputs.items():
        for variant_name, (pdf_path, png_path) in variants.items():
            print(f"{group_name} {variant_name}: saved {pdf_path}")
            print(f"{group_name} {variant_name}: saved {png_path}")
