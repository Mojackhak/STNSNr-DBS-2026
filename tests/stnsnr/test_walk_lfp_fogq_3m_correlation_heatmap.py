from __future__ import annotations

import numpy as np
import pandas as pd

from recipes.stnsnr.viz.walk import walk_lfp_fogq_3m_correlation_heatmap as heatmap


def _correlation_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_order = ["feature_a", "feature_b"]
    values = np.arange(4, dtype=float).reshape(2, 2)
    r = pd.DataFrame(
        values,
        index=heatmap.FOGQ_SOURCE_ROWS,
        columns=feature_order,
    )
    p_raw = pd.DataFrame(
        values / 100.0,
        index=heatmap.FOGQ_SOURCE_ROWS,
        columns=feature_order,
    )
    return r, p_raw


def test_prepare_heatmap_uses_exact_requested_fogq_labels() -> None:
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
    )

    expected_r = r.copy()
    expected_p = p_raw.copy()
    for expected in (expected_r, expected_p):
        expected.index = list(heatmap.FOGQ_DISPLAY_LABELS)
        expected.columns = x_labels

    assert len(r_plots) == len(heatmap.PANEL_ORDER)
    assert len(p_plots) == len(heatmap.PANEL_ORDER)
    for r_plot, p_plot in zip(r_plots, p_plots, strict=True):
        pd.testing.assert_frame_equal(r_plot, expected_r)
        pd.testing.assert_frame_equal(p_plot, expected_p)


def test_fogq_labels_use_literal_greek_capital_delta() -> None:
    assert heatmap.FOGQ_DISPLAY_LABELS == (
        "FOGQ Δ% (STN, 3m)",
        "FOGQ Δ% (SNr, 3m)",
    )
    assert all("Δ" in label for label in heatmap.FOGQ_DISPLAY_LABELS)
    assert all("△" not in label for label in heatmap.FOGQ_DISPLAY_LABELS)


def test_output_specs_use_dedicated_fogq_paths_and_delta_prefixed_lfp_labels() -> None:
    expected_names = {
        "stn_local": "walk_single_stn_local_fogq_3m_corr_heatmap",
        "snr_local": "walk_single_snr_local_fogq_3m_corr_heatmap",
        "connectivity": "walk_single_connectivity_fogq_3m_corr_heatmap",
    }

    for group_name, output in heatmap.HEATMAP_OUTPUTS.items():
        assert output.pdf_path.parent == heatmap.OUTPUT_DIR
        assert output.png_path.parent == heatmap.OUTPUT_DIR
        assert output.pdf_path.name == f"{expected_names[group_name]}.pdf"
        assert output.png_path.name == f"{expected_names[group_name]}.png"
        assert all(label.startswith("Δ") for label in output.x_labels)
