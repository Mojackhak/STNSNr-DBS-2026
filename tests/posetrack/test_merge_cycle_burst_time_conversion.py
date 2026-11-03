from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.posetrack.gait_burst_time import (
    CYCLE_DURATION_SEC_COL,
    attach_cycle_duration,
    convert_percent_axis_burst_time,
)


KEYS = ["Subject", "Task", "Epoch"]


def _feature_rows(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Subject": ["S1", "S1", "S1"],
            "Task": ["Single", "Single", "Single"],
            "Epoch": [0, 0, 0],
            "Feature": ["Stride_length", "Cycle_duration", "Walk_speed"],
            "Value": [value, value, value],
            "Value_feature": [600.0, 2.0, 300.0],
        }
    )


def _cycle_features(cycle_duration: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Subject": ["S1"],
            "Task": ["Single"],
            "Epoch": [0],
            "Stride_length": [600.0],
            "Cycle_duration": [cycle_duration],
            "Walk_speed": [300.0],
        }
    )


def test_burst_duration_uses_cycle_duration_for_all_gait_features():
    df = attach_cycle_duration(
        _feature_rows(np.log(10.0)),
        _cycle_features(2.0),
        keys=KEYS,
    )

    out = convert_percent_axis_burst_time(
        df,
        source_stem="burst-duration-scalar",
    )

    expected = np.log(10.0 * 2.0 / 100.0)
    assert np.allclose(out["Value"].to_numpy(), expected)
    assert out.groupby(["Subject", "Task", "Epoch"])["Value"].nunique().iloc[0] == 1
    assert out["Value_feature"].tolist() == [600.0, 2.0, 300.0]


def test_burst_rate_stem_converts_percent_axis_rate_to_hz():
    df = attach_cycle_duration(
        _feature_rows(np.arcsinh(0.05)),
        _cycle_features(2.0),
        keys=KEYS,
    )

    out = convert_percent_axis_burst_time(
        df,
        source_stem="burst-rate-scalar",
    )

    expected = np.arcsinh(0.05 * 100.0 / 2.0)
    assert np.allclose(out["Value"].to_numpy(), expected)
    assert out.groupby(["Subject", "Task", "Epoch"])["Value"].nunique().iloc[0] == 1


def test_invalid_cycle_duration_produces_nan_values():
    df = attach_cycle_duration(
        _feature_rows(np.log(10.0)),
        _cycle_features(0.0),
        keys=KEYS,
    )

    out = convert_percent_axis_burst_time(
        df,
        source_stem="burst-duration-scalar",
    )

    assert out[CYCLE_DURATION_SEC_COL].eq(0.0).all()
    assert out["Value"].isna().all()
