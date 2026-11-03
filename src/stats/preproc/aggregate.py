# -*- coding: utf-8 -*-
"""
lfpscope.stats.preproc.aggregate

Aggregation utilities for analysis-ready summary tables.

These functions operate on a DataFrame where `value_col` may contain:
- scalar numeric values
- nested pandas.Series
- nested pandas.DataFrame

Includes:
- summarize_df: group-wise aggregation with nested-aware element-wise reductions
- subtract_reference: global baseline subtraction using a reference subset
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ...tabular.nested_value import (
    NestedTemplate,
    cell_is_empty_or_all_nan,
    infer_nested_template,
    coerce_cell_to_array,
    rebuild_cell_from_array,
    is_scalar_na,
)

AggMode = Literal["mean", "sum", "min", "max", "median", "std", "var"]


def summarize_df(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    value_col: Union[str, List[str]],
    mode: AggMode = "mean",
    *,
    drop_other_cols: bool = True,
    drop_empty: bool = True,
    align: Literal["strict", "reindex", "force"] = "strict",
) -> pd.DataFrame:
    """
    Summarize a DataFrame by grouping and aggregating one or more (possibly nested) value columns.

    Supports:
    - scalar values -> pandas groupby aggregation (when all value columns are scalar)
    - nested Series/DataFrame -> element-wise aggregation (shape preserved)

    Aggregation uses nan-aware numpy functions.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(value_col, str):
        value_cols = [value_col]
    else:
        value_cols = list(value_col)
    if not value_cols:
        raise ValueError("value_col must be a non-empty string or list of strings.")
    missing_cols = [c for c in value_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Column(s) not found: {missing_cols}")

    valid_modes: set[str] = {"mean", "sum", "min", "max", "median", "std", "var"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported aggregation mode: {mode}. Valid modes: {sorted(valid_modes)}")

    if df.empty:
        return df.copy()

    templates = {
        col: infer_nested_template(df[col], value_col=col, drop_empty=drop_empty)
        for col in value_cols
    }

    # Scalar shortcut: standard groupby aggregation
    if all(t.kind == "scalar" for t in templates.values()):
        if drop_other_cols:
            out = df.groupby(group_cols, observed=True, dropna=False)[value_cols].agg(mode).reset_index()
            if group_cols:
                out = out.dropna(subset=group_cols)
            out = out.dropna(subset=value_cols, how="all")
            return out.reset_index(drop=True)
        extra_cols = [c for c in df.columns if c not in set(group_cols + value_cols)]
        agg_map: Dict[str, Any] = {col: mode for col in value_cols}
        for col in extra_cols:
            agg_map[col] = "first"
        out = df.groupby(group_cols, observed=True, dropna=False).agg(agg_map).reset_index()
        return out.dropna(subset=value_cols, how="all").reset_index(drop=True)

    def _agg_stack(stack: np.ndarray, *, ddof: int) -> np.ndarray:
        if mode == "mean":
            return np.nanmean(stack, axis=0)
        if mode == "sum":
            return np.nansum(stack, axis=0)
        if mode == "min":
            return np.nanmin(stack, axis=0)
        if mode == "max":
            return np.nanmax(stack, axis=0)
        if mode == "median":
            return np.nanmedian(stack, axis=0)
        if mode == "std":
            return np.nanstd(stack, axis=0, ddof=ddof)
        if mode == "var":
            return np.nanvar(stack, axis=0, ddof=ddof)
        raise RuntimeError("Unexpected aggregation mode branch")

    records: List[Dict[str, Any]] = []
    for gkey, gdf in df.groupby(group_cols, observed=True, dropna=False):
        if isinstance(gkey, tuple):
            row_dict = dict(zip(group_cols, gkey))
        else:
            row_dict = {group_cols[0]: gkey}
        any_value = False
        for col in value_cols:
            template = templates[col]
            vals = []
            for cell in gdf[col]:
                if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
                    continue
                vals.append(coerce_cell_to_array(cell, template, align=align, drop_empty=drop_empty))

            if len(vals) == 0:
                agg = coerce_cell_to_array(np.nan, template, align=align, drop_empty=drop_empty)
                value = rebuild_cell_from_array(agg, template)
            else:
                stack = np.stack(vals, axis=0)
                ddof = 1 if template.kind == "scalar" else 0
                agg = _agg_stack(stack, ddof=ddof)
                value = rebuild_cell_from_array(agg, template)
                any_value = True
            row_dict[col] = value
        if not any_value:
            continue
        if not drop_other_cols:
            extra_cols = [c for c in df.columns if c not in set(group_cols + value_cols)]
            first_row = gdf.iloc[0]
            for col in extra_cols:
                row_dict[col] = first_row[col]
        records.append(row_dict)

    out = pd.DataFrame.from_records(records)
    if drop_other_cols:
        if group_cols:
            out = out.dropna(subset=group_cols)
        out = out.dropna(subset=value_cols, how="all")
        return out.reset_index(drop=True)
    return out.dropna(subset=value_cols, how="all").reset_index(drop=True)


def subtract_reference(
    df: pd.DataFrame,
    ref_map: Dict[str, object] = {"Region": "SNr", "Lat": "Ipsi"},
    value_col: str = "Value",
    *,
    inplace: bool = False,
    on_missing: Literal["error", "skip"] = "error",
    return_ref: bool = False,
    require_same_labels: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """
    Global baseline subtraction for a column storing identically-shaped DataFrames.

    Steps
    -----
    1) Select reference rows matching `ref_map`.
    2) Compute the element-wise mean of their `value_col` matrices (NaN-safe).
    3) Subtract this mean matrix from every row's `value_col` matrix.

    Notes
    -----
    This function is intentionally strict:
    - `value_col` must contain pandas.DataFrame (or NA)
    - all non-missing entries must share the same shape
    - if `require_same_labels=True`, index/columns labels must also match
    """
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found.")

    if df.empty:
        out = df if inplace else df.copy()
        return (out, None) if return_ref else out

    if not all(isinstance(x, pd.DataFrame) or is_scalar_na(x) for x in df[value_col]):
        raise TypeError(f"All entries in '{value_col}' must be pandas.DataFrame (or NA).")

    # Pick the first non-missing DataFrame as the template
    first_df: Optional[pd.DataFrame] = None
    for v in df[value_col]:
        if isinstance(v, pd.DataFrame):
            first_df = v
            break
    if first_df is None:
        out = df if inplace else df.copy()
        return (out, None) if return_ref else out

    r, c = first_df.shape
    idx0, cols0 = first_df.index, first_df.columns

    for i, mini in enumerate(df[value_col]):
        if is_scalar_na(mini):
            continue
        assert isinstance(mini, pd.DataFrame)
        if mini.shape != (r, c):
            raise ValueError(f"Row {i} has mismatched shape: {mini.shape} vs {(r, c)}.")
        if require_same_labels and (not mini.index.equals(idx0) or not mini.columns.equals(cols0)):
            raise ValueError(f"Row {i} has mismatched index/columns labels.")

    # Build global reference mask
    mask = np.ones(len(df), dtype=bool)
    for col, val in ref_map.items():
        if col not in df.columns:
            raise KeyError(f"Reference column '{col}' not found in DataFrame.")
        mask &= (df[col] == val).to_numpy()

    if not np.any(mask):
        if on_missing == "skip":
            out = df if inplace else df.copy()
            return (out, None) if return_ref else out
        raise ValueError(f"No rows match reference filter {ref_map}.")

    mats: List[np.ndarray] = []
    for v in df[value_col]:
        if is_scalar_na(v):
            mats.append(np.full((r, c), np.nan, dtype=float))
        else:
            assert isinstance(v, pd.DataFrame)
            mats.append(v.to_numpy(dtype=float))
    arr = np.stack(mats, axis=0)

    ref_mean = np.nanmean(arr[mask, :, :], axis=0)
    corrected = arr - ref_mean

    new_vals = [pd.DataFrame(corrected[i], index=idx0, columns=cols0) for i in range(len(df))]

    if inplace:
        df[value_col] = new_vals
        out_df = df
    else:
        out_df = df.assign(**{value_col: new_vals})

    ref_df = pd.DataFrame(ref_mean, index=idx0, columns=cols0)
    return (out_df, ref_df) if return_ref else out_df
