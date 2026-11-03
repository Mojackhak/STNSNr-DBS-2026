"""
Utilities for creating and enforcing lightweight, schema-driven pandas DataFrames.

This module provides:
- A simple `Schema` type (mapping column name -> Python type) used to declare the intended
  structure of tabular outputs.
- `make_typed_dataframe()` to create an empty (or preallocated) DataFrame with consistent
  pandas dtypes, using nullable dtypes where appropriate (e.g., Int64/boolean/string).
- `coerce_df_types()` to coerce an existing DataFrame to the declared schema with a clear
  error policy ("raise" vs "coerce") and optional creation of missing columns.

The goal is to make downstream analysis code more robust and reproducible by keeping
column presence and dtypes explicit, while remaining lightweight (no heavy validation
frameworks).
"""


from __future__ import annotations

from typing import Any, Literal, Sequence
import numpy as np
import pandas as pd


Schema = dict[str, type]
CoerceErrors = Literal["raise", "coerce"]


_PYTHON_TO_PANDAS_DTYPE: dict[type, Any] = {
    str: "string",
    int: "Int64",       # nullable integer
    float: "float64",
    bool: "boolean",    # nullable boolean
}

_TRUE_TOKENS = {"true", "t", "1", "yes", "y", "on"}
_FALSE_TOKENS = {"false", "f", "0", "no", "n", "off"}


def _validate_schema(schema: Schema) -> None:
    if not isinstance(schema, dict):
        raise TypeError(f"schema must be a dict[str, type], got {type(schema)!r}")
    for k, v in schema.items():
        if not isinstance(k, str):
            raise TypeError(f"schema keys must be str, got key {k!r} of type {type(k)!r}")
        if not isinstance(v, type):
            raise TypeError(f"schema values must be types, got {v!r} for column {k!r}")


def _dtype_for_python_type(py_type: type) -> Any:
    return _PYTHON_TO_PANDAS_DTYPE.get(py_type, "object")


def _missing_value_for_dtype(dtype: Any) -> Any:
    if dtype in ("string", "Int64", "boolean"):
        return pd.NA
    if dtype == "float64":
        return np.nan
    return None


def make_typed_dataframe(schema: Schema, n_rows: int = 0) -> pd.DataFrame:
    """
    Create a DataFrame whose columns and intended types are defined by `schema`.

    Args:
        schema: Mapping column_name -> Python type, e.g. {"start": int, "condition": str}.
        n_rows: If > 0, preallocate this many rows filled with missing values.

    Returns:
        A pandas DataFrame with the specified columns and pandas dtypes.
    """
    _validate_schema(schema)
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")

    dtypes: dict[str, Any] = {col: _dtype_for_python_type(py_t) for col, py_t in schema.items()}

    if n_rows == 0:
        return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})

    data: dict[str, pd.Series] = {}
    for col, dt in dtypes.items():
        fill = _missing_value_for_dtype(dt)
        data[col] = pd.Series([fill] * n_rows, dtype=dt)
    return pd.DataFrame(data)


def _coerce_bool(series: pd.Series, *, errors: CoerceErrors) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or str(series.dtype) == "boolean":
        return series.astype("boolean")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce" if errors == "coerce" else "raise")
        rounded = np.round(numeric.astype(float))
        is_true = np.isclose(rounded, 1.0, atol=0.0, rtol=0.0)
        is_false = np.isclose(rounded, 0.0, atol=0.0, rtol=0.0)

        out = pd.Series(pd.array([pd.NA] * len(series), dtype="boolean"), index=series.index)
        out[numeric.isna()] = pd.NA
        out[is_true] = True
        out[is_false] = False

        if errors == "raise":
            invalid = ~(series.isna() | pd.Series(is_true, index=series.index) | pd.Series(is_false, index=series.index))
            if invalid.any():
                bad = series[invalid].head(5).tolist()
                raise ValueError(f"Cannot coerce to bool; invalid values (showing up to 5): {bad}")
        return out

    s = series.astype("string").str.strip().str.lower()

    out = pd.Series(pd.array([pd.NA] * len(series), dtype="boolean"), index=series.index)
    is_true = s.isin(_TRUE_TOKENS)
    is_false = s.isin(_FALSE_TOKENS)

    out[is_true] = True
    out[is_false] = False
    out[series.isna()] = pd.NA

    if errors == "raise":
        invalid = ~(series.isna() | is_true | is_false)
        if invalid.any():
            bad = series[invalid].head(5).tolist()
            raise ValueError(f"Cannot coerce to bool; invalid values (showing up to 5): {bad}")

    return out


def coerce_df_types(
    df: pd.DataFrame,
    schema: Schema,
    *,
    errors: CoerceErrors = "coerce",
    add_missing_cols: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Coerce DataFrame column dtypes according to `schema` (col -> Python type).

    Args:
        df: Input DataFrame.
        schema: Mapping column_name -> Python type (str/int/float/bool supported explicitly).
        errors: "coerce" turns invalid parses into missing values; "raise" throws exceptions.
        add_missing_cols: If True, create schema columns that are missing in df (filled with missing values).
        inplace: If True, mutate df and return it; otherwise return a copy.

    Returns:
        DataFrame with coerced dtypes where possible.
    """
    _validate_schema(schema)
    if errors not in ("raise", "coerce"):
        raise ValueError(f"errors must be 'raise' or 'coerce', got {errors!r}")

    out = df if inplace else df.copy()

    for col, py_t in schema.items():
        target_dtype = _dtype_for_python_type(py_t)

        if col not in out.columns:
            if add_missing_cols:
                fill = _missing_value_for_dtype(target_dtype)
                out[col] = pd.Series([fill] * len(out), dtype=target_dtype)
            continue

        if py_t is str:
            out[col] = out[col].astype("string")

        elif py_t is float:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce" if errors == "coerce" else "raise",
            ).astype("float64")

        elif py_t is int:
            numeric = pd.to_numeric(
                out[col],
                errors="coerce" if errors == "coerce" else "raise",
            )

            rounded = np.round(numeric.astype(float))
            integer_like = np.isclose(numeric.astype(float), rounded, atol=1e-9, rtol=0.0)

            if errors == "raise":
                invalid = ~(numeric.isna() | pd.Series(integer_like, index=numeric.index))
                if invalid.any():
                    bad = out[col][invalid].head(5).tolist()
                    raise ValueError(f"Cannot coerce column {col!r} to int; invalid values (up to 5): {bad}")

            numeric_int = pd.Series(rounded, index=numeric.index)
            numeric_int = numeric_int.where(numeric.isna() | pd.Series(integer_like, index=numeric.index), np.nan)
            out[col] = numeric_int.astype("Int64")

        elif py_t is bool:
            out[col] = _coerce_bool(out[col], errors=errors)

        else:
            out[col] = out[col].astype(target_dtype)

    return out


# make sure your imports include:
# from typing import Any, Literal, Sequence

def normalize_df_categories(
    df: pd.DataFrame,
    df_category: dict[str, Sequence[Any]],
    *,
    ordered: bool = True,
    mode: Literal["drop", "append"] = "append",
    dropna: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Normalize DataFrame columns to categorical dtypes based on a category spec.

    This helper is useful when you maintain a unified summary table schema, but
    want specific columns to be treated as ordered (or unordered) categorical
    variables for downstream plotting / stats.

    For each (col_name -> categories) entry in ``df_category``:
        1) If ``col_name`` exists in ``df``:
            - Convert df[col_name] to a pandas CategoricalDtype with the given
              ``categories``.
            - If ``mode == 'drop'``: values not in ``categories`` become NA.
            - If ``mode == 'append'``: values not in ``categories`` are kept and
              appended to the category list after the declared categories.

    Then:
        - Drop rows containing NA in any normalized category column (optional).
        - Remove unused categories from all categorical columns.
        - Reset index.

    Args:
        df: Input DataFrame.
        df_category: Mapping column_name -> allowed categories (sequence).
        ordered: Whether created categorical dtypes should be ordered.
        mode:
            - "drop": out-of-spec values become NA (and will be removed if dropna=True).
            - "append": out-of-spec values are preserved as additional categories.
        dropna: If True, drop rows that contain NA in any normalized category column.
        inplace: If True, mutate ``df`` and return it; otherwise return a copy.

    Returns:
        A normalized DataFrame.

    Raises:
        ValueError: If ``mode`` is invalid.
    """
    if mode not in ("drop", "append"):
        raise ValueError(f"mode must be 'drop' or 'append', got {mode!r}")

    out = df if inplace else df.copy()

    normalized_cols: list[str] = []

    # Preserve order without relying on hashability.
    def _unique_preserve_order(values: Sequence[Any]) -> list[Any]:
        uniq: list[Any] = []
        for v in values:
            if v not in uniq:
                uniq.append(v)
        return uniq

    for col, base_categories in df_category.items():
        if col not in out.columns:
            continue

        base_list = _unique_preserve_order(list(base_categories))

        if mode == "append":
            observed = pd.Series(out[col]).dropna().unique().tolist()
            extras: list[Any] = []
            for v in observed:
                if v not in base_list:
                    extras.append(v)
            categories = base_list + extras
        else:
            categories = base_list

        dtype = pd.CategoricalDtype(categories=categories, ordered=ordered)
        out[col] = out[col].astype(dtype)
        normalized_cols.append(col)

    if dropna and normalized_cols:
        out = out.dropna(subset=normalized_cols)

    # Remove categories that no longer exist in the data.
    out = out.apply(
        lambda col: col.cat.remove_unused_categories()
        if isinstance(col.dtype, pd.CategoricalDtype)
        else col
    )

    out = out.reset_index(drop=True)
    return out
