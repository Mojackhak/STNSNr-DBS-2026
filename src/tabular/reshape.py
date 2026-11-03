# -*- coding: utf-8 -*-
"""
Reshaping / conversion helpers for summary tables.

Includes:
- parsing tuple-like columns for pair metadata
- splitting rows by pair columns
- converting a nested value column (scalar/Series/DataFrame) into a long-form table

The long-form conversion is especially useful before statistical modeling, plotting,
or exporting to R (tidy data).
"""

from __future__ import annotations

import ast
from typing import Any, List, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

from .nested_value import (
    cell_is_empty_or_all_nan,
    coerce_cell_to_array,
    infer_nested_template,
)


def convert_tuple_like_columns(df: pd.DataFrame, cols_to_convert: Sequence[str]) -> pd.DataFrame:
    """
    Convert columns whose values are tuple-like strings '(a, b)' into real tuples (a, b).

    Only converts values that:
      - are str
      - start with '(' and end with ')'
      - can be parsed by ast.literal_eval into a tuple of length 2
    """
    out = df.copy()

    def _parse_tuple_like(v: Any, col: str) -> Any:
        if isinstance(v, str) and v.strip().startswith("(") and v.strip().endswith(")"):
            try:
                parsed = ast.literal_eval(v)
            except Exception as e:
                raise ValueError(f'Cannot parse tuple-like string in column "{col}": {v!r}') from e
            if not (isinstance(parsed, tuple) and len(parsed) == 2):
                raise ValueError(f'Parsed value is not a len-2 tuple in column "{col}": {parsed!r}')
            return parsed
        return v

    for col in cols_to_convert:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found.")
        out[col] = out[col].apply(lambda v, c=col: _parse_tuple_like(v, c))

    return out


def split_rows_by_columns(
    df: pd.DataFrame,
    cols_to_split: Sequence[str],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Duplicate each row into 2 rows by splitting the assigned columns.

    Each selected column value must be either:
      - "a-b" (str)
      - (a, b) (tuple of length 2)

    The output has 2x rows; for each split column, row 1 gets `a` and row 2 gets `b`.
    If strict=False, invalid values are copied to both rows instead of raising.
    """
    if len(df) == 0:
        return df.copy()

    out_a = df.copy()
    out_b = df.copy()

    def _to_pair(v: Any, col: str) -> Tuple[Any, Any]:
        if isinstance(v, tuple):
            if len(v) != 2:
                if strict:
                    raise ValueError(f'Column "{col}" tuple must have length 2: {v!r}')
                return v, v
            return v[0], v[1]
        if isinstance(v, str):
            if "-" not in v:
                if strict:
                    raise ValueError(f'Column "{col}" string value must contain "-": {v!r}')
                return v, v
            a, b = v.split("-", maxsplit=1)
            return a, b
        if strict:
            raise TypeError(f'Column "{col}" must be str ("a-b") or tuple (a,b); got {type(v)}')
        return v, v

    for col in cols_to_split:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.")
        pairs = df[col].apply(lambda v, c=col: _to_pair(v, c))
        out_a[col] = pairs.apply(lambda t: t[0])
        out_b[col] = pairs.apply(lambda t: t[1])

    out = pd.concat([out_a, out_b], axis=0, ignore_index=True)
    out = out.reset_index(drop=True)
    return out


def explode_value_column(
    df: pd.DataFrame,
    *,
    value_col: str = "Value",
    value_name: str = "value",
    row_coord_name: str = "row_coord",
    col_coord_name: str = "col_coord",
    drop_missing_cells: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """
    Convert a nested `value_col` into a long-form table.

    - scalar cells -> one row per original row (no extra coords)
    - Series cells -> one row per series index element (adds `row_coord_name`)
    - DataFrame cells -> one row per cell (adds `row_coord_name` and `col_coord_name`)
    """
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found.")

    if df.empty:
        return df.copy()

    template = infer_nested_template(df[value_col], value_col=value_col, drop_empty=drop_empty)
    meta_cols = [c for c in df.columns if c != value_col]
    rows: List[pd.DataFrame] = []

    if template.kind == "scalar":
        out = df[meta_cols + [value_col]].rename(columns={value_col: value_name}).copy()
        return out

    if template.kind == "series":
        assert template.index is not None
        for _, row in df.iterrows():
            cell = row[value_col]
            if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
                if drop_missing_cells:
                    continue
                values = np.full((len(template.index),), np.nan, dtype=float)
            else:
                values = coerce_cell_to_array(cell, template, align="reindex", drop_empty=drop_empty)

            part = pd.DataFrame(
                {
                    **{c: row[c] for c in meta_cols},
                    row_coord_name: template.index,
                    value_name: values,
                }
            )
            rows.append(part)

        return pd.concat(rows, axis=0, ignore_index=True) if rows else df.iloc[0:0].copy()

    if template.kind == "dataframe":
        assert template.index is not None and template.columns is not None
        for _, row in df.iterrows():
            cell = row[value_col]
            if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
                if drop_missing_cells:
                    continue
                values = np.full((len(template.index), len(template.columns)), np.nan, dtype=float)
            else:
                values = coerce_cell_to_array(cell, template, align="reindex", drop_empty=drop_empty)

            m = pd.DataFrame(values, index=template.index, columns=template.columns)
            part = m.stack(dropna=False).reset_index()
            part.columns = [row_coord_name, col_coord_name, value_name]
            for c in meta_cols:
                part[c] = row[c]
            rows.append(part[meta_cols + [row_coord_name, col_coord_name, value_name]])

        return pd.concat(rows, axis=0, ignore_index=True) if rows else df.iloc[0:0].copy()

    raise RuntimeError(f"Unsupported template.kind={template.kind!r}")



def shift_last_dim(
    x: np.ndarray,
    shift: float,
    *,
    rounding: Literal["round", "floor", "ceil"] = "round",
) -> np.ndarray:
    """
    Circularly shift an n-D numpy array along the last axis by a fraction (0..1).

    Interpretation (matches your example):
      shift = 0.30 means rolling forward by 30% of the last-axis length:
        original [0 .. 70%) -> new [30% .. 100%)
        original [70% .. 100%) -> new [0 .. 30%)

    This is equivalent to: np.roll(x, shift=+k, axis=-1),
    where k = round/ceil/floor(shift * L).

    Parameters
    ----------
    x : np.ndarray
        Input n-D array.
    shift : float
        Fractional shift in [0, 1]. For example, 0.3 means 30% of the last axis.
    rounding : {"round", "floor", "ceil"}
        How to convert shift * L to an integer number of steps.

    Returns
    -------
    np.ndarray
        Shifted array (a new array; the input is not modified).
    """
    x = np.asanyarray(x)
    L = x.shape[-1]
    if L == 0:
        return x.copy()

    # Validate shift range
    if not (0.0 <= float(shift) <= 1.0):
        raise ValueError(f"`shift` must be in [0, 1], got {shift!r}")

    # Convert fraction to integer steps
    steps_f = float(shift) * L
    if rounding == "floor":
        steps = int(np.floor(steps_f))
    elif rounding == "ceil":
        steps = int(np.ceil(steps_f))
    else:  # "round"
        steps = int(np.rint(steps_f))

    # Positive steps move values toward larger indices with wrap-around
    return np.roll(x, shift=steps, axis=-1)


def shiftTFR(x: pd.Series | pd.DataFrame, shift: float = 0.5):
    """
    Input: a pandas Series or DataFrame (numeric).
    Output: a shifted *new* object. The input `x` is NOT modified, and the result
            is created to avoid sharing memory with `x` as much as possible.
    """
    if isinstance(x, pd.Series):
        v = x.to_numpy(copy=True)                      # copy underlying numeric data
        v2 = shift_last_dim(v, shift=shift)  # shift along the last axis
        v2 = np.array(v2, copy=True)                   # extra copy in case helper returns a view
        return pd.Series(v2, index=x.index, name=x.name)

    if isinstance(x, pd.DataFrame):
        v = x.to_numpy(copy=True)
        v2 = shift_last_dim(v, shift=shift)
        v2 = np.array(v2, copy=True)
        return pd.DataFrame(v2, index=x.index, columns=x.columns)

    raise TypeError(f"shiftTFR expects a pandas Series/DataFrame, got {type(x)}")
