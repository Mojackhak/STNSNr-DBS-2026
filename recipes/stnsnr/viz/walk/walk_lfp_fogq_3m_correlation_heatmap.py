from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from recipes.stnsnr.stats.walk.walk_lfp_fogq_3m_correlation import (
    CONNECTIVITY_CORRELATION_OUTPUT_PATH,
    FOGQ_COLUMNS,
    OUTPUT_DIR,
    SNR_LOCAL_CORRELATION_OUTPUT_PATH,
    STN_LOCAL_CORRELATION_OUTPUT_PATH,
)
from recipes.stnsnr.stats.walk.walk_lfp_gait_pca_3m_correlation import (
    CONNECTIVITY_FEATURE_ORDER,
    LOCAL_FEATURE_ORDER_BY_REGION,
    POSE_ORDER,
)
from recipes.stnsnr.viz.walk.walk_lfp_gait_pca_3m_correlation_heatmap import (
    make_connectivity_x_labels,
    make_local_x_labels,
    plot_layered_correlation_heatmap,
)


PANEL_ORDER = tuple(POSE_ORDER)
FOGQ_SOURCE_ROWS = tuple(FOGQ_COLUMNS)
FOGQ_DISPLAY_LABELS = (
    "FOGQ Δ% (STN, 3m)",
    "FOGQ Δ% (SNr, 3m)",
)


@dataclass(frozen=True)
class HeatmapOutputSpec:
    correlation_path: Path
    feature_order: list[str]
    x_labels: list[str]
    label_top: str
    pdf_path: Path
    png_path: Path


def build_heatmap_outputs() -> dict[str, HeatmapOutputSpec]:
    local_x_labels = make_local_x_labels()
    connectivity_x_labels = make_connectivity_x_labels()
    return {
        "stn_local": HeatmapOutputSpec(
            correlation_path=STN_LOCAL_CORRELATION_OUTPUT_PATH,
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["STN"],
            x_labels=local_x_labels,
            label_top="STN",
            pdf_path=OUTPUT_DIR / "walk_single_stn_local_fogq_3m_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "walk_single_stn_local_fogq_3m_corr_heatmap.png",
        ),
        "snr_local": HeatmapOutputSpec(
            correlation_path=SNR_LOCAL_CORRELATION_OUTPUT_PATH,
            feature_order=LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
            x_labels=local_x_labels,
            label_top="SNr",
            pdf_path=OUTPUT_DIR / "walk_single_snr_local_fogq_3m_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "walk_single_snr_local_fogq_3m_corr_heatmap.png",
        ),
        "connectivity": HeatmapOutputSpec(
            correlation_path=CONNECTIVITY_CORRELATION_OUTPUT_PATH,
            feature_order=CONNECTIVITY_FEATURE_ORDER,
            x_labels=connectivity_x_labels,
            label_top="SNr-STN / SNr→STN",
            pdf_path=OUTPUT_DIR / "walk_single_connectivity_fogq_3m_corr_heatmap.pdf",
            png_path=OUTPUT_DIR / "walk_single_connectivity_fogq_3m_corr_heatmap.png",
        ),
    }


HEATMAP_OUTPUTS = build_heatmap_outputs()


def load_correlation_matrices_by_pose(
    correlation_path: str | Path,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load pose-specific Spearman rho and raw p-value matrices."""
    correlation_path = Path(correlation_path)
    if not correlation_path.exists():
        raise FileNotFoundError(
            f"Missing correlation workbook: {correlation_path}. "
            "Run recipes/stnsnr/stats/walk/walk_lfp_fogq_3m_correlation.py first."
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
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Reorder pose matrices and apply the requested display labels."""
    if len(feature_order) != len(x_labels):
        raise ValueError("Feature order and x-label lists must have the same length.")

    r_plots: list[pd.DataFrame] = []
    p_plots: list[pd.DataFrame] = []
    for pose in PANEL_ORDER:
        r, p_raw = matrices_by_pose[pose]
        r_plot, p_plot = prepare_heatmap_matrices(
            r,
            p_raw,
            feature_order=feature_order,
            x_labels=x_labels,
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
    pose: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one two-row FOGQ pose panel for plotting."""
    missing_y = [row for row in FOGQ_SOURCE_ROWS if row not in r.index]
    if missing_y:
        raise ValueError(f"{pose}: correlation matrix is missing FOGQ rows: {missing_y}.")

    missing_x = [feature for feature in feature_order if feature not in r.columns]
    if missing_x:
        raise ValueError(f"{pose}: correlation matrix is missing LFP features: {missing_x}.")

    r_plot = r.loc[list(FOGQ_SOURCE_ROWS), feature_order].copy()
    p_plot = p_raw.loc[list(FOGQ_SOURCE_ROWS), feature_order].copy()
    if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
        raise ValueError(f"{pose}: r and p_raw matrices are not aligned after reordering.")

    x_rename_map = dict(zip(feature_order, x_labels, strict=True))
    y_rename_map = dict(zip(FOGQ_SOURCE_ROWS, FOGQ_DISPLAY_LABELS, strict=True))
    r_plot = r_plot.rename(index=y_rename_map, columns=x_rename_map)
    p_plot = p_plot.rename(index=y_rename_map, columns=x_rename_map)
    return r_plot, p_plot


def render_group_heatmap(output: HeatmapOutputSpec) -> tuple[Path, Path]:
    """Render one analysis-group FOGQ heatmap."""
    matrices_by_pose = load_correlation_matrices_by_pose(output.correlation_path)
    r_plots, p_plots = prepare_heatmap_matrices_by_pose(
        matrices_by_pose,
        feature_order=output.feature_order,
        x_labels=output.x_labels,
    )
    _validate_plot_matrices(
        group_name=output.label_top,
        r_plots=r_plots,
        p_plots=p_plots,
        x_labels=output.x_labels,
    )

    fig = plot_layered_correlation_heatmap(
        r_plots,
        p_plots,
        label_top=output.label_top,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(output.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(output.png_path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    return output.pdf_path, output.png_path


def run_workflow() -> dict[str, tuple[Path, Path]]:
    """Render all walking LFP-FOGQ 3-month correlation heatmaps."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[Path, Path]] = {}
    for group_name, output in HEATMAP_OUTPUTS.items():
        outputs[group_name] = render_group_heatmap(output)

    validate_output_files(outputs)
    return outputs


def validate_output_files(outputs: dict[str, tuple[Path, Path]]) -> None:
    """Validate generated files and source matrix dimensions."""
    if set(outputs) != set(HEATMAP_OUTPUTS):
        raise ValueError(f"Expected output keys {sorted(HEATMAP_OUTPUTS)}; got {sorted(outputs)}.")

    for group_name, output_paths in outputs.items():
        for output_path in output_paths:
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise FileNotFoundError(f"{group_name}: missing heatmap output {output_path}.")

        spec = HEATMAP_OUTPUTS[group_name]
        expected_shape = (len(FOGQ_SOURCE_ROWS), len(spec.feature_order))
        matrices_by_pose = load_correlation_matrices_by_pose(spec.correlation_path)
        for pose, (r, p_raw) in matrices_by_pose.items():
            r_shape = r.loc[list(FOGQ_SOURCE_ROWS), spec.feature_order].shape
            p_shape = p_raw.loc[list(FOGQ_SOURCE_ROWS), spec.feature_order].shape
            if r_shape != expected_shape:
                raise ValueError(
                    f"{group_name} {pose}: expected r shape {expected_shape}; got {r_shape}."
                )
            if p_shape != expected_shape:
                raise ValueError(
                    f"{group_name} {pose}: expected p_raw shape {expected_shape}; got {p_shape}."
                )


def _validate_plot_matrices(
    *,
    group_name: str,
    r_plots: list[pd.DataFrame],
    p_plots: list[pd.DataFrame],
    x_labels: list[str],
) -> None:
    expected_shape = (len(FOGQ_DISPLAY_LABELS), len(x_labels))
    for pose, r_plot, p_plot in zip(PANEL_ORDER, r_plots, p_plots, strict=True):
        if r_plot.shape != expected_shape:
            raise ValueError(
                f"{group_name} {pose}: expected r plot shape {expected_shape}; got {r_plot.shape}."
            )
        if p_plot.shape != expected_shape:
            raise ValueError(
                f"{group_name} {pose}: expected p plot shape {expected_shape}; got {p_plot.shape}."
            )
        if not r_plot.index.equals(p_plot.index) or not r_plot.columns.equals(p_plot.columns):
            raise ValueError(f"{group_name} {pose}: r and p plot matrices are not aligned.")
        if r_plot.index.tolist() != list(FOGQ_DISPLAY_LABELS):
            raise ValueError(f"{group_name} {pose}: y-axis labels do not match the FOGQ contract.")
        if r_plot.columns.tolist() != x_labels:
            raise ValueError(f"{group_name} {pose}: x-axis labels do not match the LFP order.")


if __name__ == "__main__":
    workflow_outputs = run_workflow()
    for group_name, (pdf_path, png_path) in workflow_outputs.items():
        print(f"{group_name}: saved {pdf_path}")
        print(f"{group_name}: saved {png_path}")
