# -*- coding: utf-8 -*-
"""
lfpscope.stats.preproc.outliers

Group-wise sample-level outlier filtering for summary tables.

The input is a DataFrame where `value_col` can contain:
- scalar numeric values
- nested pandas.Series
- nested pandas.DataFrame

For outlier detection, each row's nested value is flattened into a feature vector.
The flattening is ONLY used for detection; the returned filtered DataFrame keeps the
original nested objects intact.

This module provides:
- filter_df: high-level dispatcher supporting multiple detectors
- filter_df_mcd_safe: a robust MCD backend with PCA stabilization for small-n / high-p
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ...tabular.nested_value import is_scalar_na as _is_scalar_na


TOL_TOTAL_VAR = 1e-12   # total variance threshold for PCA degeneracy
TOL_COV_REG = 1e-6      # ridge factor for empirical covariance fallback

OutlierMode = Literal["mad_z", "iqr", "mcd", "iforest", "lof", "pca_t2"]


def filter_df(
    df: pd.DataFrame,
    *,
    group_cols: Union[str, List[str]],
    value_col: Union[str, List[str]],
    mode: OutlierMode = "mcd",
    return_scores: bool = False,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Group by `group_cols` and perform outlier detection on `value_col` at the row level.

    If `value_col` is a list, filtering is applied sequentially in the given order:
    the output of each step becomes the input of the next step. In this case, the
    returned report includes a "steps" list with per-column reports.

    Supported methods:
      - "mad_z": Robust Z (median + MAD). kwargs: z_thr=3.5
      - "iqr": IQR rule. kwargs: iqr_k=3.0
      - "mcd": PCA-assisted robust Mahalanobis (safe backend)
      - "iforest": IsolationForest. kwargs: contamination='auto'
      - "lof": LocalOutlierFactor. kwargs: n_neighbors=20, contamination=0.05
      - "pca_t2": PCA + Hotelling's T² + SPE/Q. kwargs: alpha=0.99, pca_var=0.9

    Returns
    -------
    df_filtered, report
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if isinstance(value_col, str):
        value_cols = [value_col]
    else:
        value_cols = list(value_col)

    if not value_cols:
        raise ValueError("value_col must be a non-empty string or list of strings")

    _check_value_column(df, value_cols)

    if len(value_cols) == 1:
        return _filter_df_single(
            df,
            group_cols=group_cols,
            value_col=value_cols[0],
            mode=mode,
            return_scores=return_scores,
            random_state=random_state,
            **kwargs,
        )

    return _filter_df_multi_value_cols(
        df,
        group_cols=group_cols,
        value_cols=value_cols,
        mode=mode,
        return_scores=return_scores,
        random_state=random_state,
        **kwargs,
    )


def _filter_df_single(
    df: pd.DataFrame,
    *,
    group_cols: Union[str, List[str]],
    value_col: str,
    mode: OutlierMode = "mcd",
    return_scores: bool = False,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if not isinstance(value_col, str):
        raise TypeError("_filter_df_single expects value_col to be a string")

    _check_value_column(df, value_col)

    requested_mode: str = mode.lower()

    if requested_mode == "mcd":
        return filter_df_mcd_safe(
            df,
            group_cols=group_cols,
            value_col=value_col,
            alpha=kwargs.get("alpha", 0.975),
            support_fraction=kwargs.get("support_fraction", 0.75),
            pca=kwargs.get("pca", True),
            pca_max_components=kwargs.get("pca_max_components", 20),
            pca_explained_var=kwargs.get("pca_explained_var", 0.99),
            min_n=kwargs.get("min_n", 8),
            min_n_over_p=kwargs.get("min_n_over_p", 1.2),
            robust_scale=kwargs.get("robust_scale", True),
            relax_if_all_rejected=kwargs.get("relax_if_all_rejected", True),
            relax_alpha=kwargs.get("relax_alpha", 0.999),
            random_state=random_state,
            return_scores=return_scores,
        )

    valid_modes = {"mad_z", "iqr", "mcd", "iforest", "lof", "pca_t2"}
    if requested_mode not in valid_modes:
        raise ValueError(f"Unsupported outlier detection mode: {mode}. Valid: {sorted(valid_modes)}")

    kept_parts: List[pd.DataFrame] = []
    records: List[Dict[str, Any]] = []
    score_store: Dict[Any, float] = {}

    for gkey, gdf in df.groupby(group_cols, observed=True, dropna=False):
        idx = gdf.index
        X, _meta = _stack_features(gdf[value_col].tolist())

        if X.shape[0] == 0 or not np.isfinite(X).any():
            kept_parts.append(gdf)
            records.append(_mk_group_record(gkey, gdf, removed_idx=[], mode_used=requested_mode))
            continue

        X = _impute_nan_with_col_median(X)
        n = X.shape[0]

        mode_used = requested_mode
        try:
            keep_mask, scores = _detect_outliers_by_mode(
                X, mode=mode_used, random_state=random_state, **kwargs
            )
        except Exception as e:
            warnings.warn(
                f"Outlier detection failed for group {gkey} with mode '{mode_used}': {e}. "
                "Falling back to 'mad_z'.",
                RuntimeWarning,
            )
            mode_used = "mad_z"
            keep_mask, scores = _mad_z_default(X, z_thr=float(kwargs.get("z_thr", 3.5)))

        if keep_mask is None or len(keep_mask) != n:
            mode_used = "mad_z"
            keep_mask, scores = _mad_z_default(X, z_thr=float(kwargs.get("z_thr", 3.5)))

        kept_parts.append(gdf.loc[idx[keep_mask]])

        removed_idx = idx[~keep_mask].tolist()
        records.append(_mk_group_record(gkey, gdf, removed_idx=removed_idx, mode_used=mode_used))

        if return_scores and scores is not None:
            for i_row, s in zip(idx, scores):
                score_store[i_row] = float(s)

    df_filtered = pd.concat(kept_parts, axis=0).sort_index() if kept_parts else df.iloc[0:0]

    by_group_df = pd.DataFrame.from_records(records)
    overall_removed = int(sum(len(r["removed_index"]) for r in records))
    overall = {
        "n_groups": int(by_group_df.shape[0]),
        "n_rows": int(df.shape[0]),
        "n_removed": overall_removed,
        "pct_removed": float(overall_removed / max(1, df.shape[0])),
        "mode": requested_mode,
    }
    report: Dict[str, Any] = {
        "overall": overall,
        "by_group": by_group_df.sort_values("group_key"),
    }
    if return_scores:
        report["scores"] = score_store

    return df_filtered, report


def _filter_df_multi_value_cols(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    value_cols: List[str],
    mode: OutlierMode = "mcd",
    return_scores: bool = False,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    requested_mode: str = mode.lower()

    original_group_sizes: Dict[Tuple[Any, ...], int] = {}
    for gkey, gdf in df.groupby(group_cols, observed=True, dropna=False):
        gkey_t = gkey if isinstance(gkey, tuple) else (gkey,)
        original_group_sizes[gkey_t] = int(gdf.shape[0])

    df_current = df
    steps: List[Dict[str, Any]] = []
    removed_by_group: Dict[Tuple[Any, ...], set] = {}
    modes_by_group: Dict[Tuple[Any, ...], set] = {}

    for col in value_cols:
        df_current, report = _filter_df_single(
            df_current,
            group_cols=group_cols,
            value_col=col,
            mode=mode,
            return_scores=return_scores,
            random_state=random_state,
            **kwargs,
        )

        step_report = dict(report)
        step_report["value_col"] = col
        steps.append(step_report)

        by_group_df = report.get("by_group")
        if isinstance(by_group_df, pd.DataFrame) and not by_group_df.empty:
            for _, row in by_group_df.iterrows():
                gkey = row.get("group_key")
                if isinstance(gkey, tuple):
                    gkey_t = gkey
                elif isinstance(gkey, list):
                    gkey_t = tuple(gkey)
                else:
                    gkey_t = (gkey,)
                removed_idx = row.get("removed_index") or []
                if removed_idx:
                    removed_by_group.setdefault(gkey_t, set()).update(removed_idx)
                mode_used = row.get("mode_used")
                if mode_used is not None:
                    modes_by_group.setdefault(gkey_t, set()).add(mode_used)

    records: List[Dict[str, Any]] = []
    for gkey_t, n_rows in original_group_sizes.items():
        removed_idx = sorted(removed_by_group.get(gkey_t, set()))
        modes = modes_by_group.get(gkey_t, set())
        if len(modes) == 1:
            mode_used = next(iter(modes))
        elif len(modes) == 0:
            mode_used = requested_mode
        else:
            mode_used = "mixed"
        records.append(
            {
                "group_key": gkey_t,
                "n_rows": int(n_rows),
                "n_removed": int(len(removed_idx)),
                "pct_removed": (len(removed_idx) / n_rows) if n_rows else 0.0,
                "removed_index": removed_idx,
                "mode_used": mode_used,
            }
        )

    df_filtered = df_current
    by_group_df = pd.DataFrame.from_records(records)
    overall_removed = int(df.shape[0] - df_filtered.shape[0])
    overall = {
        "n_groups": int(len(original_group_sizes)),
        "n_rows": int(df.shape[0]),
        "n_removed": overall_removed,
        "pct_removed": float(overall_removed / max(1, df.shape[0])),
        "mode": requested_mode,
        "value_cols": list(value_cols),
    }
    report: Dict[str, Any] = {
        "overall": overall,
        "by_group": by_group_df.sort_values("group_key"),
        "steps": steps,
    }

    if return_scores:
        scores_by_step: Dict[str, Dict[Any, float]] = {}
        for step in steps:
            step_col = step.get("value_col")
            if isinstance(step_col, str):
                scores_by_step[step_col] = step.get("scores", {})
        report["scores"] = scores_by_step.get(value_cols[-1], {})
        report["scores_by_step"] = scores_by_step

    return df_filtered, report


def filter_df_mcd_safe(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    value_col: str = "Value",
    alpha: float = 0.975,
    support_fraction: Optional[float] = 0.75,
    pca: bool = True,
    pca_max_components: int = 20,
    pca_explained_var: Optional[float] = 0.99,
    min_n: int = 8,
    min_n_over_p: float = 1.2,
    robust_scale: bool = True,
    relax_if_all_rejected: bool = True,
    relax_alpha: float = 0.999,
    random_state: Optional[int] = None,
    return_scores: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    PCA-assisted, safe MCD outlier filtering per group.

    High-dimensional/small-n groups are stabilized by PCA (with whitening) before MCD.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if not isinstance(value_col, str):
        raise TypeError("filter_df_mcd_safe expects a single value_col string; use filter_df for lists")

    _check_value_column(df, value_col)

    MinCovDet, _, _, PCA = _require_sklearn_outliers()

    kept_parts: List[pd.DataFrame] = []
    records: List[Dict[str, Any]] = []
    score_store: Dict[Any, float] = {}

    for gkey, gdf in df.groupby(group_cols, observed=True, dropna=False):
        idx = gdf.index
        X, _meta = _stack_features(gdf[value_col].tolist())
        n, p_features = X.shape

        if n == 0 or p_features == 0 or not np.isfinite(X).any():
            kept_parts.append(gdf)
            records.append(_mk_group_record(gkey, gdf, removed_idx=[], mode_used="mcd"))
            continue

        X = _impute_nan_with_col_median(X)
        Xs = _robust_scale(X) if robust_scale else X

        total_var = float(np.nanvar(Xs, axis=0, ddof=1).sum())
        if (not np.isfinite(total_var)) or (total_var <= TOL_TOTAL_VAR):
            kept_parts.append(gdf)
            records.append(_mk_group_record(gkey, gdf, removed_idx=[], mode_used="mcd"))
            continue

        use_pca = bool(pca and (n < min_n or n < (min_n_over_p * p_features)))
        if use_pca:
            k_cap = max(1, min(n - 1, p_features, pca_max_components))
            pca_model = PCA(
                n_components=k_cap,
                svd_solver="full",
                whiten=True,
                random_state=random_state,
            )
            scores = pca_model.fit_transform(Xs)
            if pca_explained_var is not None and 0 < pca_explained_var < 1:
                cum = np.cumsum(pca_model.explained_variance_ratio_)
                k = int(np.searchsorted(cum, pca_explained_var) + 1)
                k = max(1, min(k, k_cap))
                Xr = scores[:, :k]
            else:
                Xr = scores
                k = Xr.shape[1]
        else:
            Xr = Xs
            k = Xr.shape[1]

        if n <= 2 or n <= k:
            kept_parts.append(gdf)
            records.append(_mk_group_record(gkey, gdf, removed_idx=[], mode_used="mcd"))
            continue

        h_min = int(np.ceil((n + k + 1) / 2.0))
        sf_min = min(1.0, max(0.5, h_min / float(n)))
        sf_try = min(1.0, max(support_fraction if support_fraction is not None else 0.75, sf_min))

        def _fit_mcd(sf_value: float) -> Tuple[np.ndarray, bool]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=RuntimeWarning)
                mcd_local = MinCovDet(support_fraction=sf_value, random_state=random_state).fit(Xr)
                md2_local = mcd_local.mahalanobis(Xr)
                warned = any("Determinant has increased" in str(ww.message) for ww in w)
            return md2_local, warned

        md2: Optional[np.ndarray] = None
        warned = False

        try:
            md2, warned = _fit_mcd(sf_try)
        except Exception:
            md2 = None
            warned = True

        if (md2 is None) or warned:
            sf_retry = min(1.0, max(sf_try, 0.99))
            try:
                md2, warned2 = _fit_mcd(sf_retry)
                warned = warned and warned2
            except Exception:
                md2 = None

        if md2 is None:
            mu = np.nanmean(Xr, axis=0)
            diffs = Xr - mu
            cov = np.cov(Xr, rowvar=False)
            cov = np.atleast_2d(cov)
            k_eff = cov.shape[0]
            tr = float(np.trace(cov))
            lam = (tr / max(1, k_eff))
            lam = TOL_COV_REG * (lam if np.isfinite(lam) and lam > 0 else 1.0)
            cov_reg = cov + lam * np.eye(k_eff)
            inv_cov = np.linalg.pinv(cov_reg)
            md2 = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)

        thr = chi2.ppf(alpha, df=k)
        keep_mask = md2 <= thr

        if relax_if_all_rejected and not keep_mask.any():
            thr_relax = chi2.ppf(relax_alpha, df=k)
            keep_mask = md2 <= thr_relax

        kept_parts.append(gdf.loc[idx[keep_mask]])
        removed_idx = idx[~keep_mask].tolist()
        records.append(_mk_group_record(gkey, gdf, removed_idx=removed_idx, mode_used="mcd"))

        if return_scores:
            for i_row, s in zip(idx, md2):
                score_store[i_row] = float(s)

    df_filtered = pd.concat(kept_parts, axis=0).sort_index() if kept_parts else df.iloc[0:0]

    by_group_df = pd.DataFrame.from_records(records)
    overall_removed = int(sum(len(r["removed_index"]) for r in records))
    overall = {
        "n_groups": int(by_group_df.shape[0]),
        "n_rows": int(df.shape[0]),
        "n_removed": overall_removed,
        "pct_removed": float(overall_removed / max(1, df.shape[0])),
        "mode": "mcd",
        "alpha": float(alpha),
        "support_fraction": support_fraction,
        "pca": bool(pca),
        "pca_max_components": int(pca_max_components),
        "pca_explained_var": pca_explained_var,
        "min_n": int(min_n),
        "min_n_over_p": float(min_n_over_p),
        "robust_scale": bool(robust_scale),
        "relax_if_all_rejected": bool(relax_if_all_rejected),
        "relax_alpha": float(relax_alpha),
    }
    report: Dict[str, Any] = {
        "overall": overall,
        "by_group": by_group_df.sort_values("group_key"),
    }
    if return_scores:
        report["scores"] = score_store

    return df_filtered, report


# -----------------------------------------------------------------------------
# Internal helpers (feature construction & detectors)
# -----------------------------------------------------------------------------

def _require_sklearn_outliers():
    """Import sklearn components lazily to keep non-ML parts usable without sklearn."""
    try:
        from sklearn.covariance import MinCovDet
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.decomposition import PCA
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for outlier detection modes 'mcd', 'iforest', 'lof', 'pca_t2'."
        ) from e
    return MinCovDet, IsolationForest, LocalOutlierFactor, PCA


def _check_value_column(df: pd.DataFrame, value_col: Union[str, List[str]]) -> None:
    if isinstance(value_col, str):
        if value_col not in df.columns:
            raise KeyError(f"value_col='{value_col}' is not in df.columns")
        return

    missing = [col for col in value_col if col not in df.columns]
    if missing:
        raise KeyError(f"value_col(s) missing from df.columns: {missing}")


def _value_kind(x: Any) -> str:
    """Return one of: 'scalar' | 'series' | 'dataframe' | 'ndarray1d' | 'ndarray2d' | 'unknown'."""
    if x is None:
        return "scalar"
    if isinstance(x, float) and np.isnan(x):
        return "scalar"
    if isinstance(x, (int, float, np.integer, np.floating, np.bool_)):
        return "scalar"
    if isinstance(x, pd.Series):
        return "series"
    if isinstance(x, pd.DataFrame):
        return "dataframe"
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return "ndarray1d"
        if x.ndim == 2:
            return "ndarray2d"
    return "unknown"


def _stack_features(values: List[Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stack a list of scalar/Series/DataFrame values into a feature matrix X (n × p).

    - Scalar -> p=1
    - Series / ndarray1d -> align to union index and stack
    - DataFrame / ndarray2d -> align to union (index, columns) and flatten row-major
    """
    n_total = len(values)
    non_missing = [v for v in values if _value_kind(v) != "unknown" and not _is_scalar_na(v)]
    if len(non_missing) == 0:
        X = np.full((n_total, 1), np.nan, dtype=float)
        return X, {"kind": "scalar"}

    kinds = {_value_kind(v) for v in non_missing}
    if kinds <= {"scalar"}:
        arr = [float(v) if not _is_scalar_na(v) else np.nan for v in values]
        X = np.asarray(arr, dtype=float).reshape(-1, 1)
        return X, {"kind": "scalar"}

    if kinds <= {"series", "ndarray1d"}:
        ser_list: List[pd.Series] = []
        for v in values:
            k = _value_kind(v)
            if k == "series":
                ser_list.append(v)
            elif k == "ndarray1d":
                ser_list.append(pd.Series(v))
            else:
                ser_list.append(pd.Series(dtype=float))

        try:
            union_idx: pd.Index = ser_list[0].index
            for s in ser_list[1:]:
                union_idx = union_idx.union(s.index)
        except Exception:
            max_len = max((len(s) for s in ser_list), default=0)
            union_idx = pd.RangeIndex(max_len)

        rows = [s.reindex(union_idx).to_numpy(dtype=float) for s in ser_list]
        X = np.vstack(rows) if rows else np.full((n_total, len(union_idx)), np.nan, dtype=float)
        return X, {"kind": "series", "index": union_idx}

    if kinds <= {"dataframe", "ndarray2d"}:
        df_list: List[pd.DataFrame] = []
        for v in values:
            k = _value_kind(v)
            if k == "dataframe":
                df_list.append(v)
            elif k == "ndarray2d":
                df_list.append(pd.DataFrame(v))
            else:
                df_list.append(pd.DataFrame())

        try:
            union_idx: pd.Index = df_list[0].index
            union_cols: pd.Index = df_list[0].columns
            for m in df_list[1:]:
                union_idx = union_idx.union(m.index)
                union_cols = union_cols.union(m.columns)
        except Exception:
            max_r = max((m.shape[0] for m in df_list), default=0)
            max_c = max((m.shape[1] for m in df_list), default=0)
            union_idx = pd.RangeIndex(max_r)
            union_cols = pd.RangeIndex(max_c)

        rows: List[np.ndarray] = []
        for m in df_list:
            aligned = m.reindex(index=union_idx, columns=union_cols)
            rows.append(aligned.to_numpy(dtype=float).ravel(order="C"))
        X = (
            np.vstack(rows)
            if rows
            else np.full((n_total, len(union_idx) * len(union_cols)), np.nan, dtype=float)
        )
        return X, {"kind": "dataframe", "index": union_idx, "columns": union_cols}

    raise TypeError(f"value_col contains incompatible types for outlier detection: {sorted(kinds)}")


def _impute_nan_with_col_median(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X = X.copy()
        X[inds] = np.take(med, inds[1])
    return X


def _robust_scale(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    scale = np.where(iqr == 0, 1.0, iqr)
    return (X - med) / scale


def _mad_z_default(X: np.ndarray, z_thr: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    abs_dev = np.abs(X - med)
    mad = np.nanmedian(abs_dev, axis=0)
    sigma = 1.4826 * mad

    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    sigma = np.where(sigma == 0, np.where(iqr == 0, 1.0, 0.7413 * iqr), sigma)

    z = abs_dev / sigma
    max_z = np.nanmax(z, axis=1)
    keep = max_z <= z_thr
    return keep, max_z


def _detect_outliers_by_mode(
    X: np.ndarray,
    *,
    mode: str,
    random_state: int = 0,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mode = mode.lower()
    n, p = X.shape

    if mode == "mad_z":
        z_thr = float(kwargs.get("z_thr", 3.5))
        return _mad_z_default(X, z_thr=z_thr)

    if mode == "iqr":
        k = float(kwargs.get("iqr_k", 3.0))
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        out_any = (X < lower) | (X > upper)
        out_row = np.any(out_any, axis=1)
        keep = ~out_row
        denom = np.where(iqr == 0, 1.0, iqr)
        score = np.nanmax(np.maximum((X - upper) / denom, (lower - X) / denom), axis=1)
        score = np.clip(score, 0, None)
        return keep, score

    if mode in {"iforest", "lof", "pca_t2", "mcd"}:
        MinCovDet, IsolationForest, LocalOutlierFactor, PCA = _require_sklearn_outliers()

    if mode == "mcd":
        alpha = float(kwargs.get("alpha", 0.975))
        Xs = _robust_scale(X) if bool(kwargs.get("robust_scale", True)) else X
        if n <= p + 1:
            return _mad_z_default(X, z_thr=float(kwargs.get("z_thr", 3.5)))
        mcd = MinCovDet(random_state=random_state).fit(Xs)
        mu = mcd.location_
        cov = mcd.covariance_
        try:
            vi = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            vi = np.linalg.pinv(cov)
        diff = Xs - mu
        mah2 = np.einsum("ij,jk,ik->i", diff, vi, diff)
        thr = chi2.ppf(alpha, df=p)
        keep = mah2 <= thr
        return keep, mah2

    if mode == "iforest":
        contamination = kwargs.get("contamination", "auto")
        clf = IsolationForest(contamination=contamination, random_state=random_state)
        pred = clf.fit_predict(X)  # 1=inlier, -1=outlier
        keep = pred == 1
        score = -clf.decision_function(X)  # larger = more anomalous
        return keep, score

    if mode == "lof":
        n_neighbors = int(kwargs.get("n_neighbors", 20))
        n_neighbors = max(2, min(n_neighbors, max(2, n - 1)))
        contamination = kwargs.get("contamination", 0.05)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        pred = lof.fit_predict(X)
        keep = pred == 1
        score = -lof.negative_outlier_factor_
        return keep, score

    if mode == "pca_t2":
        alpha = float(kwargs.get("alpha", 0.99))
        pca_var = float(kwargs.get("pca_var", 0.9))
        Xs = _robust_scale(X)
        k_max = min(n - 1, p) if n > 1 else 1
        if k_max <= 0:
            return _mad_z_default(X, z_thr=float(kwargs.get("z_thr", 3.5)))
        pca_model = PCA(n_components=k_max, random_state=random_state).fit(Xs)
        cum = np.cumsum(pca_model.explained_variance_ratio_)
        k = int(np.searchsorted(cum, pca_var) + 1)
        k = max(1, min(k, k_max))
        scores = pca_model.transform(Xs)
        lam = pca_model.explained_variance_

        t2 = np.sum((scores[:, :k] ** 2) / lam[:k], axis=1)
        t2_thr = chi2.ppf(alpha, df=k)

        x_hat = pca_model.inverse_transform(scores[:, :k])
        q = np.sum((Xs - x_hat) ** 2, axis=1)
        q_thr = float(np.quantile(q, alpha))

        keep = (t2 <= t2_thr) & (q <= q_thr)
        score = np.maximum(t2 / t2_thr, q / q_thr)
        return keep, score

    raise ValueError(f"Unknown outlier mode='{mode}'")


def _mk_group_record(
    gkey: Any,
    gdf: pd.DataFrame,
    removed_idx: List[Any],
    *,
    mode_used: str,
) -> Dict[str, Any]:
    n = int(gdf.shape[0])
    r = int(len(removed_idx))
    return {
        "group_key": gkey if isinstance(gkey, tuple) else (gkey,),
        "n_rows": n,
        "n_removed": r,
        "pct_removed": (r / n) if n else 0.0,
        "removed_index": removed_idx,
        "mode_used": mode_used,
    }
