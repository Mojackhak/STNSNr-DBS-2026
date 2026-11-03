# -*- coding: utf-8 -*-
"""
tabular

Utilities for working with pandas DataFrames as analysis-ready summary tables,
including support for a nested value column (scalar/Series/DataFrame).
"""

from .text import capitalize_df
from .missing import drop_missing_value_rows
from .selection import slice_df
from .reshape import convert_tuple_like_columns, split_rows_by_columns, explode_value_column
from .nested_value import NestedTemplate, infer_nested_template

__all__ = [
    "capitalize_df",
    "drop_missing_value_rows",
    "slice_df",
    "convert_tuple_like_columns",
    "split_rows_by_columns",
    "explode_value_column",
    # Advanced / internal-facing
    "NestedTemplate",
    "infer_nested_template",
]
