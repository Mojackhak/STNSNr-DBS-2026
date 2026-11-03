from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


CYCLE_DURATION_FEATURE = "Cycle_duration"
CYCLE_DURATION_SEC_COL = "Cycle_duration_sec"
BURST_DURATION_STEM = "burst-duration-scalar"
BURST_RATE_STEM = "burst-rate-scalar"


def cycle_duration_lookup(
    df_features: pd.DataFrame,
    *,
    keys: Sequence[str],
    feature_col: str = CYCLE_DURATION_FEATURE,
    out_col: str = CYCLE_DURATION_SEC_COL,
) -> pd.DataFrame:
    if feature_col not in df_features.columns:
        raise KeyError(f"Column '{feature_col}' not found in gait-cycle features.")

    missing_keys = [key for key in keys if key not in df_features.columns]
    if missing_keys:
        raise KeyError(f"Missing gait-cycle feature key columns: {missing_keys}")

    lookup = df_features[list(keys) + [feature_col]].copy()
    lookup[out_col] = pd.to_numeric(lookup[feature_col], errors="coerce")
    lookup = lookup.drop(columns=[feature_col]).drop_duplicates()

    duplicated = lookup.duplicated(subset=list(keys), keep=False)
    if duplicated.any():
        dup_keys = lookup.loc[duplicated, list(keys)].drop_duplicates().head(5)
        raise ValueError(
            "Cycle duration lookup is not unique for keys "
            f"{list(keys)}. Example duplicates: {dup_keys.to_dict(orient='records')}"
        )

    return lookup


def attach_cycle_duration(
    df_long: pd.DataFrame,
    df_features: pd.DataFrame,
    *,
    keys: Sequence[str],
    cycle_duration_col: str = CYCLE_DURATION_SEC_COL,
) -> pd.DataFrame:
    lookup = cycle_duration_lookup(
        df_features,
        keys=keys,
        out_col=cycle_duration_col,
    )
    out = df_long.drop(columns=[cycle_duration_col], errors="ignore")
    return out.merge(lookup, on=list(keys), how="left", validate="many_to_one")


def convert_percent_axis_burst_time(
    df: pd.DataFrame,
    *,
    source_stem: str,
    value_col: str = "Value",
    cycle_duration_col: str = CYCLE_DURATION_SEC_COL,
) -> pd.DataFrame:
    if source_stem not in {BURST_DURATION_STEM, BURST_RATE_STEM}:
        return df.copy()

    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found.")
    if cycle_duration_col not in df.columns:
        raise KeyError(f"Column '{cycle_duration_col}' not found.")

    out = df.copy()
    values = pd.to_numeric(out[value_col], errors="coerce").to_numpy(dtype=float)
    cycle_duration = pd.to_numeric(out[cycle_duration_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(cycle_duration) & (cycle_duration > 0.0)

    converted = np.full(values.shape, np.nan, dtype=float)
    if source_stem == BURST_DURATION_STEM:
        converted[valid] = values[valid] - np.log(100.0 / cycle_duration[valid])
    elif source_stem == BURST_RATE_STEM:
        converted[valid] = np.arcsinh(np.sinh(values[valid]) * 100.0 / cycle_duration[valid])

    out[value_col] = converted
    return out
