from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipes.stnsnr.stats.med import med_scale_correlation as stats  # noqa: E402
from recipes.stnsnr.viz.med import med_scale_correlation as heatmap  # noqa: E402


def test_combined_tremor_delta_percent_uses_summed_scores(tmp_path: Path) -> None:
    input_path = tmp_path / "subj_scale.xlsx"
    rows = [
        _scale_row("valid", rest_off=3.0, rest_on=1.0, action_off=1.0, action_on=1.0),
        _scale_row("zero", rest_off=0.0, rest_on=0.0, action_off=0.0, action_on=0.0),
        _scale_row("missing", rest_off=3.0, rest_on=1.0, action_off=np.nan, action_on=1.0),
    ]
    pd.DataFrame(rows).to_excel(input_path, index=False)

    combined = stats.load_scale_combined_tremor_delta_percent(input_path)

    assert combined.columns.tolist() == [
        stats.PAIR_COL,
        *stats.SCALE_COMBINED_TREMOR_DELTA_PERCENT_COLUMNS,
    ]
    assert combined.loc[combined[stats.PAIR_COL].eq("valid"), stats.COMBINED_TREMOR_DELTA_PERCENT_COLUMN].item() == 50.0
    assert combined.loc[combined[stats.PAIR_COL].eq("zero"), stats.COMBINED_TREMOR_DELTA_PERCENT_COLUMN].isna().item()
    assert combined.loc[combined[stats.PAIR_COL].eq("missing"), stats.COMBINED_TREMOR_DELTA_PERCENT_COLUMN].isna().item()
    assert "MDS-UPDRS III rest tremor Δ%" not in combined.columns
    assert "MDS-UPDRS III action tremor Δ%" not in combined.columns


def test_combined_tremor_heatmap_uses_six_clinical_rows() -> None:
    spec = heatmap.HEATMAP_OUTPUTS["stn_local_combined_tremor"]
    feature_order = spec["feature_order"]
    scale_columns = spec["scale_columns"]
    r = pd.DataFrame(0.25, index=scale_columns, columns=feature_order)
    p_raw = pd.DataFrame(0.01, index=scale_columns, columns=feature_order)
    x_labels = heatmap.make_local_x_labels()

    r_plot, p_plot = heatmap.prepare_heatmap_matrices(
        r,
        p_raw,
        feature_order=feature_order,
        x_labels=x_labels,
        scale_columns=scale_columns,
    )

    assert r_plot.shape == (6, 38)
    assert p_plot.shape == (6, 38)
    assert r_plot.index.tolist() == [
        "LCT Δ%",
        "UPDRS III Δ%",
        "Axial Δ%",
        "Rigidity Δ%",
        "Bradykinesia Δ%",
        "Tremor Δ%",
    ]
    heatmap._validate_plot_matrices(
        "stn_local_combined_tremor",
        r_plot,
        p_plot,
        x_labels,
        scale_columns=scale_columns,
    )


def test_original_and_combined_tremor_heatmaps_are_both_registered() -> None:
    expected_keys = {
        "stn_local",
        "snr_local",
        "connectivity",
        "stn_local_combined_tremor",
        "snr_local_combined_tremor",
        "connectivity_combined_tremor",
    }

    assert set(heatmap.HEATMAP_OUTPUTS) == expected_keys
    assert heatmap.HEATMAP_OUTPUTS["stn_local"]["scale_columns"] == stats.SCALE_DELTA_PERCENT_COLUMNS
    assert (
        heatmap.HEATMAP_OUTPUTS["stn_local_combined_tremor"]["scale_columns"]
        == stats.SCALE_COMBINED_TREMOR_DELTA_PERCENT_COLUMNS
    )
    assert set(heatmap.COMBINED_TREMOR_HEATMAP_KEYS) == {
        "stn_local_combined_tremor",
        "snr_local_combined_tremor",
        "connectivity_combined_tremor",
    }
    assert set(stats.COMBINED_TREMOR_OUTPUT_SPECS) == set(
        heatmap.COMBINED_TREMOR_HEATMAP_KEYS
    )


def _scale_row(
    subject_id: str,
    *,
    rest_off: float,
    rest_on: float,
    action_off: float,
    action_on: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        stats.PAIR_COL: subject_id,
        "LCT improvement rate": 50.0,
    }
    for _, off_column, on_column in stats.SCALE_DELTA_PERCENT_SPECS:
        row[off_column] = 10.0
        row[on_column] = 5.0
    row[stats.REST_TREMOR_OFF_COLUMN] = rest_off
    row[stats.REST_TREMOR_ON_COLUMN] = rest_on
    row[stats.ACTION_TREMOR_OFF_COLUMN] = action_off
    row[stats.ACTION_TREMOR_ON_COLUMN] = action_on
    return row
