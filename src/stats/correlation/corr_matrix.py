from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy import stats
import warnings


CorrelationMethod = Literal["spearman", "pearson"]
PAdjustMode = Literal[None, "All", "row", "col"]
PAdjustMethod = Literal["bh"]


@dataclass(frozen=True)
class CorrelationMatrixResult:
    """Container for correlation matrix outputs.

    Attributes
    ----------
    r : pd.DataFrame
        Correlation coefficient matrix. Rows correspond to df_y columns; columns correspond to df_x columns.
    p : pd.DataFrame
        P-value matrix after applying correction_mode (if any). If correction_mode is None, equals raw p-values.
    p_raw : pd.DataFrame
        Raw (uncorrected) p-value matrix.
    p_adj : pd.DataFrame | None
        Adjusted p-value matrix if correction_mode is not None; otherwise None.
    n : pd.DataFrame
        Pairwise valid sample size used for each correlation test.
    method : str
        Correlation method ("spearman" or "pearson").
    correction_mode : str | None
        Multiple-testing correction mode: None, "All", "row", or "col".
    correction_method : str | None
        Correction method, currently supports "bh" only.
    pair_col : str
        Column name used to pair samples between df_x and df_y.
    x_cols : list[str]
        Numeric scalar columns used from df_x (matrix columns).
    y_cols : list[str]
        Numeric scalar columns used from df_y (matrix rows).
    """

    r: pd.DataFrame
    p: pd.DataFrame
    p_raw: pd.DataFrame
    p_adj: pd.DataFrame | None
    n: pd.DataFrame
    method: str
    correction_mode: str | None
    correction_method: str | None
    pair_col: str
    x_cols: list[str]
    y_cols: list[str]


def compute_correlation_matrix(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    pair_col: str,
    method: CorrelationMethod = "spearman",
    correction_mode: PAdjustMode = None,
    correction_method: PAdjustMethod = "bh",
    min_n: int = 3,
) -> CorrelationMatrixResult:
    """Compute pairwise correlation matrix between numeric scalar columns of df_x and df_y.

    Parameters
    ----------
    df_x, df_y : pd.DataFrame
        Input dataframes. Both must contain pair_col for pairing the same sample.
    pair_col : str
        Column name used to pair samples between df_x and df_y (e.g., "ID").
    method : {"spearman", "pearson"}, default "spearman"
        Correlation method.
    correction_mode : {None, "All", "row", "col"}, default None
        Multiple testing correction scope:
        - None: no correction
        - "All": correct all p-values together
        - "row": correct p-values within each row (per df_y variable across df_x variables)
        - "col": correct p-values within each column (per df_x variable across df_y variables)
    correction_method : {"bh"}, default "bh"
        Multiple testing correction method. Currently only BH-FDR is implemented.
    min_n : int, default 3
        Minimum number of paired non-missing observations required to run a test.
        If fewer than min_n, r and p will be NaN.

    Returns
    -------
    CorrelationMatrixResult
        A dataclass containing r, p (possibly corrected), raw p, adjusted p, and n matrices.

    Notes
    -----
    - Non-scalar columns (excluding pair_col) are skipped with a warning.
    - Only numeric columns (including boolean) are tested.
    - Pairing requires unique, non-missing keys in pair_col for both df_x and df_y.
    """
    _validate_inputs(df_x=df_x, df_y=df_y, pair_col=pair_col, min_n=min_n)

    method_norm = _normalize_method(method)
    mode_norm = _normalize_mode(correction_mode)
    corr_method_norm = _normalize_correction_method(correction_method)

    x_cols = _get_usable_numeric_scalar_columns(df_x, pair_col=pair_col, df_label="df_x")
    y_cols = _get_usable_numeric_scalar_columns(df_y, pair_col=pair_col, df_label="df_y")

    if len(x_cols) == 0:
        raise ValueError("No usable numeric scalar columns found in df_x (excluding pair_col).")
    if len(y_cols) == 0:
        raise ValueError("No usable numeric scalar columns found in df_y (excluding pair_col).")

    x_aligned, y_aligned = _pair_and_align(df_x=df_x, df_y=df_y, pair_col=pair_col)

    # Preconvert to float arrays for speed/consistency (booleans become 0/1)
    x_arrays = {c: pd.to_numeric(x_aligned[c], errors="coerce").to_numpy(dtype=float) for c in x_cols}
    y_arrays = {c: pd.to_numeric(y_aligned[c], errors="coerce").to_numpy(dtype=float) for c in y_cols}

    r_arr = np.full((len(y_cols), len(x_cols)), np.nan, dtype=float)
    p_arr = np.full((len(y_cols), len(x_cols)), np.nan, dtype=float)
    n_arr = np.zeros((len(y_cols), len(x_cols)), dtype=int)

    for i, y_name in enumerate(y_cols):
        yv = y_arrays[y_name]
        for j, x_name in enumerate(x_cols):
            xv = x_arrays[x_name]

            r, p, n = _corr_test(xv, yv, method=method_norm, min_n=min_n)
            r_arr[i, j] = r
            p_arr[i, j] = p
            n_arr[i, j] = n

    r_df = pd.DataFrame(r_arr, index=y_cols, columns=x_cols)
    p_raw_df = pd.DataFrame(p_arr, index=y_cols, columns=x_cols)
    n_df = pd.DataFrame(n_arr, index=y_cols, columns=x_cols)

    p_adj_df: pd.DataFrame | None = None
    p_out_df = p_raw_df

    if mode_norm is not None:
        p_adj_df = adjust_pvalues_df(p_raw_df, mode=mode_norm, method=corr_method_norm)
        p_out_df = p_adj_df

    return CorrelationMatrixResult(
        r=r_df,
        p=p_out_df,
        p_raw=p_raw_df,
        p_adj=p_adj_df,
        n=n_df,
        method=method_norm,
        correction_mode=mode_norm,
        correction_method=corr_method_norm if mode_norm is not None else None,
        pair_col=pair_col,
        x_cols=x_cols,
        y_cols=y_cols,
    )


def adjust_pvalues_df(pvalues: pd.DataFrame, mode: Literal["All", "row", "col"], method: PAdjustMethod = "bh") -> pd.DataFrame:
    """Adjust p-values in a dataframe using BH-FDR under a given scope.

    Parameters
    ----------
    pvalues : pd.DataFrame
        Raw p-values (may contain NaN).
    mode : {"All", "row", "col"}
        Adjustment scope.
    method : {"bh"}, default "bh"
        Adjustment method.

    Returns
    -------
    pd.DataFrame
        Adjusted p-values with the same shape/index/columns as input.
    """
    method_norm = _normalize_correction_method(method)
    mode_norm = _normalize_mode(mode)
    if mode_norm is None:
        raise ValueError("mode must be one of: 'All', 'row', 'col'.")

    p = pvalues.to_numpy(dtype=float, copy=True)
    adj = np.full_like(p, np.nan, dtype=float)

    if method_norm != "bh":
        raise ValueError(f"Unsupported correction method: {method_norm}. Only 'bh' is supported.")

    if mode_norm == "All":
        adj = _bh_fdr(p)
    elif mode_norm == "row":
        for i in range(p.shape[0]):
            adj[i, :] = _bh_fdr(p[i, :])
    elif mode_norm == "col":
        for j in range(p.shape[1]):
            adj[:, j] = _bh_fdr(p[:, j])
    else:
        raise ValueError(f"Unsupported mode: {mode_norm}")

    return pd.DataFrame(adj, index=pvalues.index, columns=pvalues.columns)


def _validate_inputs(df_x: pd.DataFrame, df_y: pd.DataFrame, pair_col: str, min_n: int) -> None:
    if not isinstance(df_x, pd.DataFrame) or not isinstance(df_y, pd.DataFrame):
        raise TypeError("df_x and df_y must be pandas DataFrame objects.")
    if pair_col not in df_x.columns or pair_col not in df_y.columns:
        raise KeyError(f"pair_col='{pair_col}' must exist in both df_x and df_y.")
    if min_n < 2:
        raise ValueError("min_n must be >= 2.")

    if df_x[pair_col].isna().any():
        raise ValueError("df_x contains missing values in pair_col; pairing would be ambiguous.")
    if df_y[pair_col].isna().any():
        raise ValueError("df_y contains missing values in pair_col; pairing would be ambiguous.")

    if df_x[pair_col].duplicated().any():
        raise ValueError("df_x contains duplicate values in pair_col; pairing is ambiguous.")
    if df_y[pair_col].duplicated().any():
        raise ValueError("df_y contains duplicate values in pair_col; pairing is ambiguous.")


def _pair_and_align(df_x: pd.DataFrame, df_y: pd.DataFrame, pair_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df_x.set_index(pair_col, drop=True)
    y = df_y.set_index(pair_col, drop=True)

    common = x.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping pair_col values between df_x and df_y; cannot pair samples.")

    # Align both to the same index order
    common = common.sort_values()
    x_aligned = x.loc[common]
    y_aligned = y.loc[common]
    return x_aligned, y_aligned


def _get_usable_numeric_scalar_columns(df: pd.DataFrame, pair_col: str, df_label: str) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col == pair_col:
            continue

        s = df[col]

        if not _is_scalar_column(s):
            warnings.warn(
                f"{df_label}: column '{col}' contains non-scalar values; skipping this column.",
                UserWarning,
            )
            continue

        if is_numeric_dtype(s) or is_bool_dtype(s):
            cols.append(col)

    return cols


def _is_scalar_column(s: pd.Series) -> bool:
    # Numeric/bool columns are scalar by construction in normal pandas usage.
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return True

    # For object/mixed columns, explicitly check values (short-circuit on first non-scalar).
    # This is strict: if any non-null value is non-scalar (e.g., list/ndarray/dict), we skip.
    for v in s.dropna().to_numpy():
        if not pd.api.types.is_scalar(v):
            return False
    return True


def _normalize_method(method: str) -> CorrelationMethod:
    m = str(method).strip().lower()
    if m not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'.")
    return "spearman" if m == "spearman" else "pearson"


def _normalize_mode(mode: object) -> PAdjustMode:
    if mode is None:
        return None
    m = str(mode).strip()
    m_low = m.lower()
    if m_low == "none":
        return None
    if m_low == "all":
        return "All"
    if m_low == "row":
        return "row"
    if m_low == "col":
        return "col"
    raise ValueError("correction_mode must be one of: None, 'All', 'row', 'col'.")


def _normalize_correction_method(method: str) -> PAdjustMethod:
    m = str(method).strip().lower()
    if m in ("bh", "fdr_bh", "benjamini-hochberg", "benjamini_hochberg"):
        return "bh"
    raise ValueError("correction_method currently supports only BH-FDR: 'bh'.")


def _corr_test(x: np.ndarray, y: np.ndarray, method: CorrelationMethod, min_n: int) -> tuple[float, float, int]:
    # Pairwise finite mask: removes NaN and +/-inf
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    n = int(xv.size)

    if n < min_n:
        return np.nan, np.nan, n

    # Constant check: correlation undefined if one side has zero variance
    if np.nanstd(xv) == 0.0 or np.nanstd(yv) == 0.0:
        return np.nan, np.nan, n

    try:
        if method == "pearson":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = stats.pearsonr(xv, yv)
            r = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
            p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
            return r, p, n

        # Spearman
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.spearmanr(xv, yv)
        r = float(res.correlation) if hasattr(res, "correlation") else float(res[0])
        p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        return r, p, n

    except Exception:
        # Be conservative: do not crash the whole matrix due to one pair.
        return np.nan, np.nan, n


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment.

    Parameters
    ----------
    p : np.ndarray
        Raw p-values (any shape). NaNs are ignored and preserved.

    Returns
    -------
    np.ndarray
        Adjusted p-values with same shape.
    """
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    mask = np.isfinite(p)
    pv = p[mask]
    m = int(pv.size)
    if m == 0:
        return out

    order = np.argsort(pv)
    pv_sorted = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)

    q_sorted = pv_sorted * (m / ranks)
    # Ensure monotonicity from largest to smallest
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    qv = np.empty_like(q_sorted)
    qv[order] = q_sorted

    out[mask] = qv
    return out
