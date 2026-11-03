from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipes.stnsnr.stats.clinic import scale_coords_standardized_slope as stats  # noqa: E402
from recipes.stnsnr.viz.clinic import scale_coords_standardized_slope_heatmap as heatmap  # noqa: E402


EXPECTED_ORDER = [
    "MDS-UPDRS I Δ% (STN, 3m)",
    "MDS-UDPRS II Δ% (STN, 3m)",
    "UPDRS-III Δ% (STN, 3m)",
    "UPDRS-III axial Δ% (STN, 3m)",
    "Rigidity subscore (Med OFF) Δ% (STN, 3m)",
    "Bradykinesia subscore (Med OFF) Δ% (STN, 3m)",
    "Tremor subscore (Med OFF) Δ% (STN, 3m)",
    "MDS-UPDRS IV Δ% (STN, 3m)",
    "FOG-Q Δ% (STN, 3m)",
    "BDI Δ% (STN, 3m)",
    "MADRS Δ% (STN, 3m)",
    "HAMD Δ% (STN, 3m)",
    "HAMA Δ% (STN, 3m)",
    "MAES Δ% (STN, 3m)",
    "ESS Δ% (STN, 3m)",
    "RBDSQ Δ% (STN, 3m)",
    "RBDSQ-HK Δ% (STN, 3m)",
    "SCOPA-AUT Δ% (STN, 3m)",
    "SDQ Δ% (STN, 3m)",
    "EAT-10 Δ% (STN, 3m)",
    "DSFS Δ% (STN, 3m)",
    "CCCS Δ% (STN, 3m)",
    "FSS Δ% (STN, 3m)",
    "KPPS Δ% (STN, 3m)",
    "ODQ Δ% (STN, 3m)",
    "ADL Δ% (STN, 3m)",
    "SE-ADL Δ% (STN, 3m)",
    "PDQ-39 Δ% (STN, 3m)",
]


def test_standardized_slope_groups_cover_all_28_three_month_scales() -> None:
    assert len(stats.STN_3M_SCALE_COLUMNS) == 28
    assert len(stats.DELTA_3M_SCALE_COLUMNS) == 28
    assert list(stats.STN_3M_SCALE_COLUMNS) == EXPECTED_ORDER
    assert list(stats.DELTA_3M_SCALE_COLUMNS) == [
        "Δ" + scale.removesuffix(" Δ% (STN, 3m)") + " Δ% (+SNr, 3m)"
        for scale in EXPECTED_ORDER
    ]
    assert all("3m" in scale for scale in stats.STN_3M_SCALE_COLUMNS)
    assert all("immed" not in scale for scale in stats.STN_3M_SCALE_COLUMNS)
    assert all(not scale.startswith("Δ") for scale in stats.STN_3M_SCALE_COLUMNS)
    assert all("3m" in scale for scale in stats.DELTA_3M_SCALE_COLUMNS)
    assert all("immed" not in scale for scale in stats.DELTA_3M_SCALE_COLUMNS)
    assert all(scale.startswith("Δ") for scale in stats.DELTA_3M_SCALE_COLUMNS)


def test_heatmap_matrices_include_all_scales_and_transpose() -> None:
    spec = next(item for item in heatmap.HEATMAPS if item.key == "stn_3m_efficacy")
    table = _standardized_slope_rows(spec.group, stats.STN_3M_SCALE_COLUMNS)

    value_matrix, q_matrix = heatmap.prepare_heatmap_matrices(table, spec)

    assert value_matrix.shape == (28, 3)
    assert q_matrix.shape == (28, 3)
    assert value_matrix.index[0] == "MDS-UPDRS I Δ%"
    assert value_matrix.index[1] == "MDS-UPDRS II Δ%"
    assert value_matrix.index[-1] == "PDQ-39 Δ%"

    transposed_value, transposed_q = heatmap.transpose_heatmap_matrices(
        value_matrix,
        q_matrix,
    )
    assert transposed_value.shape == (3, 28)
    assert transposed_q.shape == (3, 28)
    assert transposed_value.index.tolist() == [coordinate.label for coordinate in stats.COORDINATES]


def test_missing_slope_cells_remain_blank_in_28_scale_heatmap() -> None:
    spec = next(item for item in heatmap.HEATMAPS if item.key == "stn_3m_efficacy")
    table = _standardized_slope_rows(spec.group, stats.STN_3M_SCALE_COLUMNS)
    table.loc[table["scale"].eq("EAT-10 Δ% (STN, 3m)"), ["q", "standardized_slope_display"]] = np.nan

    value_matrix, q_matrix = heatmap.prepare_heatmap_matrices(table, spec)

    assert value_matrix.shape == (28, 3)
    assert q_matrix.shape == (28, 3)
    assert value_matrix.loc["EAT-10 Δ%"].isna().all()
    assert q_matrix.loc["EAT-10 Δ%"].isna().all()
    heatmap._validate_plot_matrices(spec, value_matrix, q_matrix)


def test_heatmap_specs_register_original_and_transposed_outputs() -> None:
    expected_keys = {
        "stn_3m_efficacy",
        "stn_3m_efficacy_transposed",
        "stn_snr_3m_delta_efficacy",
        "stn_snr_3m_delta_efficacy_transposed",
    }

    assert {spec.key for spec in heatmap.HEATMAPS} == expected_keys
    for spec in heatmap.HEATMAPS:
        assert spec.pdf_path.name.endswith(".pdf")
        assert spec.png_path.name.endswith(".png")


def _standardized_slope_rows(group: str, scales: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scale_order, scale in enumerate(scales):
        for coordinate in stats.COORDINATES:
            rows.append(
                {
                    "group": group,
                    "scale": scale,
                    "scale_order": scale_order,
                    "coordinate": coordinate.key,
                    "coordinate_label": coordinate.label,
                    "q": 0.01,
                    "standardized_slope": 0.25,
                    "standardized_slope_display": np.float64(0.25),
                }
            )
    return pd.DataFrame(rows)
