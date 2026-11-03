from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLINIC_STATS_DIR = (
    Path(__file__).resolve().parents[2] / "recipes" / "stnsnr" / "stats" / "clinic"
)
sys.path.insert(0, str(CLINIC_STATS_DIR))

import subject_effect_tables  # noqa: E402


def _origin_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": "SNr001",
                "Protocol": protocol,
                "Phase": phase,
                "Scale": scale,
                "Value": value,
                "Baseline": baseline,
            }
            for scale, baseline, values in [
                (
                    "MDS-UPDRS III score",
                    100.0,
                    {
                        ("STN", "immediate"): 60.0,
                        ("STN+SNr", "immediate"): 50.0,
                        ("STN", "3m"): 30.0,
                        ("STN+SNr", "3m"): 20.0,
                    },
                ),
                (
                    "MDS-UPDRS III axial score",
                    40.0,
                    {
                        ("STN", "immediate"): 30.0,
                        ("STN+SNr", "immediate"): 20.0,
                        ("STN", "3m"): 10.0,
                        ("STN+SNr", "3m"): 8.0,
                    },
                ),
                (
                    "SE-ADL score (%)",
                    40.0,
                    {
                        ("STN", "3m"): 60.0,
                        ("STN+SNr", "3m"): 80.0,
                    },
                ),
                (
                    "KPPS score",
                    0.0,
                    {
                        ("STN", "3m"): 0.0,
                        ("STN+SNr", "3m"): 0.0,
                    },
                ),
                (
                    "BDI score",
                    50.0,
                    {
                        ("STN", "3m"): 40.0,
                        ("STN+SNr", "3m"): 25.0,
                    },
                ),
            ]
            for (protocol, phase), value in values.items()
        ]
    )


def test_expected_columns_cover_all_28_scales() -> None:
    effect_columns = subject_effect_tables.expected_effect_columns(include_metadata=True)
    delta_columns = subject_effect_tables.expected_delta_effect_columns()

    assert len(subject_effect_tables.SCALE_LABELS) == 28
    assert len(effect_columns) == 64
    assert len(delta_columns) == 61
    assert effect_columns[:24] == subject_effect_tables.LEGACY_EFFECT_COLUMNS
    assert delta_columns[:21] == subject_effect_tables.LEGACY_DELTA_EFFECT_COLUMNS
    assert "BDI Δ% (STN, 3m)" in effect_columns
    assert "ΔBDI Δ% (+SNr, 3m)" in delta_columns


def test_build_effect_tables_uses_origin_rules_and_zero_baseline_blanks() -> None:
    metadata = pd.DataFrame(
        {
            "ID": ["SNr001"],
            "NameEn": ["Subject One"],
            "NameZh": ["受试者一"],
            "LCT Δ%": [42.0],
        }
    )

    effect_table, delta_table = subject_effect_tables.build_effect_tables(
        _origin_rows(),
        metadata,
    )

    row = effect_table.set_index("ID").loc["SNr001"]
    delta_row = delta_table.set_index("ID").loc["SNr001"]

    assert row["NameEn"] == "Subject One"
    assert row["UPDRS-III Δ% (STN, immed)"] == pytest.approx(40.0)
    assert row["UPDRS-III Δ% (STN+SNr, immed)"] == pytest.approx(50.0)
    assert row["UPDRS-III Δ% (STN, 3m)"] == pytest.approx(70.0)
    assert row["UPDRS-III Δ% (STN+SNr, 3m)"] == pytest.approx(80.0)
    assert row["SE-ADL Δ% (STN, 3m)"] == pytest.approx(50.0)
    assert row["SE-ADL Δ% (STN+SNr, 3m)"] == pytest.approx(100.0)
    assert pd.isna(row["KPPS Δ% (STN, 3m)"])

    assert delta_row["UPDRS-III Δ% (STN, 3m)"] == pytest.approx(70.0)
    assert delta_row["ΔUPDRS-III Δ% (+SNr, 3m)"] == pytest.approx(10.0)
    assert delta_row["ΔSE-ADL Δ% (+SNr, 3m)"] == pytest.approx(50.0)
    assert pd.isna(delta_row["KPPS Δ% (STN, 3m)"])
    assert pd.isna(delta_row["ΔKPPS Δ% (+SNr, 3m)"])


def test_shared_old_columns_must_match_after_refresh() -> None:
    old_table = pd.DataFrame(
        {
            "ID": ["SNr001", "SNr002"],
            "PDQ-39 Δ% (STN, 3m)": [41.176471, np.nan],
            "KPPS Δ% (STN, 3m)": [np.nan, 12.5],
        }
    )
    refreshed = old_table.assign(**{"New scale Δ% (STN, 3m)": [1.0, 2.0]})

    subject_effect_tables.assert_shared_columns_equal(old_table, refreshed)

    drifted = refreshed.copy()
    drifted.loc[0, "PDQ-39 Δ% (STN, 3m)"] = 41.0
    with pytest.raises(AssertionError, match="PDQ-39"):
        subject_effect_tables.assert_shared_columns_equal(old_table, drifted)
