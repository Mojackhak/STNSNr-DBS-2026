# -*- coding: utf-8 -*-
"""
Missing-value utilities for summary tables that use a nested value column.

The key function here, `drop_missing_value_rows`, implements a *strict* definition
of missingness suitable for QC (configurable via `mode`):
- scalar: NA or non-finite
- Series/DataFrame: NA/non-finite handled per mode:
    - "any": drop if any NA/non-finite in the cell
    - "all": drop only if all values are NA/non-finite
- optionally treat empty Series/DataFrame as missing
"""

from __future__ import annotations

from typing import List, Tuple, Union, Literal

import pandas as pd

from .nested_value import cell_has_any_nonfinite


def drop_missing_value_rows(
    df: pd.DataFrame,
    value_col: str = "Value",
    *,
    drop_empty: bool = True,
    mode: Literal["any", "all"] = "any",
    return_dropped_indices: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[int]]]:
    """
    Drop rows whose cell in `value_col` is considered missing (strict rule).

    Parameters
    ----------
    df:
        Input DataFrame.
    value_col:
        Column to evaluate.
    drop_empty:
        If True, empty Series/DataFrame cells are considered missing.
    mode:
        Missingness rule:
        - "any": drop if any element is NA/non-finite (default).
        - "all": drop only if all elements are NA/non-finite.
    return_dropped_indices:
        If True, also return a list of dropped row indices.

    Returns
    -------
    Filtered DataFrame (and optionally dropped indices).

    Raises
    ------
    KeyError
        If `value_col` is not a column in `df`.
    """
    if value_col not in df.columns:
        raise KeyError(f"value_col='{value_col}' not found in df.columns")

    keep_mask = ~df[value_col].apply(
        lambda c: cell_has_any_nonfinite(c, drop_empty=drop_empty, mode=mode)
    )
    dropped_idx = df.index[~keep_mask].tolist()
    out = df.loc[keep_mask]

    if return_dropped_indices:
        return out, dropped_idx
    return out
