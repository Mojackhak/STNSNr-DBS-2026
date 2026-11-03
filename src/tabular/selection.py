# -*- coding: utf-8 -*-
"""
Row-selection helpers for pandas DataFrames used as analysis-ready summary tables.

These helpers are intentionally conservative: they validate columns, remove unused
categories for categorical columns, drop NA rows, and reset the index.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Literal

import pandas as pd


def slice_df(
    df: pd.DataFrame,
    subset: Mapping[str, Sequence[Any]],
    mode: Literal["raise", "skip"] = "skip",
) -> pd.DataFrame:
    """
    Filter rows according to a subset dictionary and clean categorical columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    subset:
        Mapping: column name -> allowed values.

    Returns
    -------
    Filtered DataFrame with:
    - unused categories removed
    - NA rows dropped
    - index reset
    """
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    for col, allowed_vals in subset.items():
        if col not in out.columns:
            if mode == "raise":
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            elif mode == "skip":
                continue
            else:
                raise ValueError(f"Invalid mode: {mode}")
        mask &= out[col].isin(list(allowed_vals))

    out = out.loc[mask]

    def _drop_unused_categories(col: pd.Series) -> pd.Series:
        if isinstance(col.dtype, pd.CategoricalDtype):
            return col.cat.remove_unused_categories()
        return col

    out = out.apply(_drop_unused_categories, axis=0)
    out = out.reset_index(drop=True)
    return out
