from __future__ import annotations

import numpy as np
import pandas as pd

from recipes.stnsnr.viz.walk import walk_lfp_gait_pca_3m_correlation_heatmap as heatmap


def _correlation_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_order = ["feature_a", "feature_b"]
    values = np.arange(6, dtype=float).reshape(3, 2)
    r = pd.DataFrame(
        values,
        index=heatmap.GAIT_PCA_COLUMNS,
        columns=feature_order,
    )
    p_raw = pd.DataFrame(
        values / 100.0,
        index=heatmap.GAIT_PCA_COLUMNS,
        columns=feature_order,
    )
    return r, p_raw


def test_compact_heatmap_retains_and_relabels_exact_source_rows() -> None:
    feature_order = ["feature_a", "feature_b"]
    x_labels = ["Feature A", "Feature B"]
    r, p_raw = _correlation_frames()
    matrices_by_pose = {
        pose: (r.copy(), p_raw.copy())
        for pose in heatmap.PANEL_ORDER
    }

    r_plots, p_plots = heatmap.prepare_heatmap_matrices_by_pose(
        matrices_by_pose,
        feature_order=feature_order,
        x_labels=x_labels,
        source_rows=heatmap.COMPACT_GAIT_PCA_ROWS,
        y_labels=heatmap.COMPACT_GAIT_PCA_LABELS,
    )

    expected_r = r.loc[list(heatmap.COMPACT_GAIT_PCA_ROWS), feature_order].copy()
    expected_p = p_raw.loc[list(heatmap.COMPACT_GAIT_PCA_ROWS), feature_order].copy()
    for expected in (expected_r, expected_p):
        expected.index = list(heatmap.COMPACT_GAIT_PCA_LABELS)
        expected.columns = x_labels

    assert len(r_plots) == len(heatmap.PANEL_ORDER)
    assert len(p_plots) == len(heatmap.PANEL_ORDER)
    for r_plot, p_plot in zip(r_plots, p_plots, strict=True):
        pd.testing.assert_frame_equal(r_plot, expected_r)
        pd.testing.assert_frame_equal(p_plot, expected_p)


def test_compact_variant_uses_distinct_descriptive_output_paths() -> None:
    variant = heatmap.HEATMAP_VARIANTS["stn_snr_effects"]

    for output in heatmap.HEATMAP_OUTPUTS.values():
        assert output.correlation_path.parent == heatmap.CORRELATION_DIR
        pdf_path, png_path = heatmap.output_paths_for_variant(output, variant)
        assert pdf_path.parent == heatmap.CORRELATION_DIR
        assert png_path.parent == heatmap.CORRELATION_DIR
        assert pdf_path.name.endswith("_corr_heatmap_stn_snr_effects.pdf")
        assert png_path.name.endswith("_corr_heatmap_stn_snr_effects.png")
        assert pdf_path != output.pdf_path
        assert png_path != output.png_path


def test_compact_labels_use_greek_capital_delta() -> None:
    assert heatmap.COMPACT_GAIT_PCA_LABELS == (
        "Gait PCA Δ% (STN, 3m)",
        "Gait PCA Δ% (SNr, 3m)",
    )
