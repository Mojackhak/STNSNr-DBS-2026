# -*- coding: utf-8 -*-
"""
stats.preproc

Preprocessing utilities for statistical analysis on summary tables.
"""

from .aggregate import summarize_df, subtract_reference
from .transform import transform_df
from .normalize import normalize_df, baseline_normalize
from .outliers import filter_df, filter_df_mcd_safe

__all__ = [
    "summarize_df",
    "subtract_reference",
    "transform_df",
    "normalize_df",
    "baseline_normalize",
    "filter_df",
    "filter_df_mcd_safe",
]
