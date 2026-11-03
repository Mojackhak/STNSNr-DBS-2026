# -*- coding: utf-8 -*-
"""
Element-wise transforms for a nested value column in a summary table.

This module provides `transform_df`, which applies a transform (log, dB, logit,
Fisher-z, ...) to each nested cell in `value_col` while preserving the original
cell structure (scalar / Series / DataFrame).

Important note:
This transform does NOT align Series/DataFrame labels across rows. Each cell is
transformed independently, preserving its own index/columns.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ...tabular.nested_value import cell_is_empty_or_all_nan
from ...utils.transforms import TransformMode, apply_transform_array


def transform_df(
    df: pd.DataFrame,
    *,
    value_col: str = "Value",
    mode: TransformMode | None = None,
    out_col: Optional[str] = None,
    eps: float = 1e-120,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """
    Apply an element-wise transform to `value_col` (scalar/Series/DataFrame).

    This function does NOT align labels (Series index / DataFrame index+columns)
    across rows. Each nested cell is transformed independently and its own
    index/columns are preserved.

    Transform modes:
      - "dB": 10*log10(x) with x clipped to >= eps
      - "log": log(x) with x clipped to >= eps
      - "fisherz": atanh(x) with x clipped to (-1+eps, 1-eps)
      - "fisherz_sqrt": atanh(sqrt(C)) with C clipped to [0, 1-eps)
      - "logit": log(x/(1-x)) with x clipped to (eps, 1-eps)
      - "asinh": log(x + sqrt(x^2 + 1))
      - "none": identity
    """
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found.")

    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be a positive number")

    if mode is None:
        return df.copy()

    def tf(arr: np.ndarray) -> np.ndarray:
        return apply_transform_array(arr, mode=mode, eps=eps)

    def coerce_series_to_float_array(s: pd.Series) -> np.ndarray:
        # Coerce non-numeric entries to NaN, preserving length and order.
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    def coerce_df_to_float_array(m: pd.DataFrame) -> np.ndarray:
        # Flatten -> to_numeric -> reshape, keeping original shape.
        flat_obj = m.to_numpy().ravel(order="C")
        flat_num = pd.to_numeric(pd.Series(flat_obj, copy=False), errors="coerce").to_numpy(dtype=float)
        return flat_num.reshape(m.shape)

    def transform_cell(cell: Any) -> Any:
        if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
            return np.nan

        if isinstance(cell, pd.Series):
            arr = coerce_series_to_float_array(cell)
            out_arr = tf(arr)
            # Preserve the original labels (index) and name.
            return pd.Series(out_arr, index=cell.index, name=cell.name)

        if isinstance(cell, pd.DataFrame):
            arr = coerce_df_to_float_array(cell)
            out_arr = tf(arr)
            # Preserve the original labels (index/columns).
            return pd.DataFrame(out_arr, index=cell.index, columns=cell.columns)

        # Scalar path (float/int/numpy scalar, etc.)
        arr0 = np.asarray(cell)
        if arr0.ndim != 0 and arr0.size != 1:
            raise TypeError(
                "Unsupported nested cell value. Expected scalar/pandas.Series/pandas.DataFrame; "
                f"got {type(cell)} with shape {getattr(arr0, 'shape', None)}."
            )
        arr = np.asarray(arr0, dtype=float).reshape(())
        out_arr = tf(arr)
        try:
            return float(np.asarray(out_arr).item())
        except Exception:
            return np.asarray(out_arr).item()

    out = df.copy()
    dest = value_col if out_col is None else out_col

    # IMPORTANT:
    # Do NOT use Series.apply here, because if the first transformed cell is a Series,
    # pandas may try to "expand" outputs into a DataFrame. We need an object column.
    transformed = [transform_cell(c) for c in out[value_col].tolist()]
    out[dest] = transformed
    return out
