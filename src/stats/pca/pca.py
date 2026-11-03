#%% Libraries
import os
# import glob
# import h5py
import math
from collections.abc import Mapping
from typing import Any, Literal, TypedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import dlc2kinematics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import itertools

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from plotly.offline import iplot
from ...utils.paths import get_files_list
from ...utils.fileops import create_subfolder

#%% Functions

ImputeStrategy = Literal["mean", "median"]


class PcaResult(TypedDict, total=False):
    df_scores: pd.DataFrame
    explained_variance_ratio: np.ndarray
    df_loadings: pd.DataFrame
    df_x_processed: pd.DataFrame
    feature_names: list[str]
    pc_columns: list[str]
    imputer_obj: SimpleImputer | None
    scaler_obj: StandardScaler | None
    pca_obj: PCA


class PcaImprovementProjection(TypedDict):
    g: pd.Series
    g_hat: pd.Series
    u: pd.Series
    u_hat: pd.Series


def perform_PCA(
    df: pd.DataFrame,
    n_components: int | float | None = None,  # None / int / float (0-1, variance threshold)
    impute: ImputeStrategy | None = None,  # "mean" | "median" | None
    scale: bool = True,  # True -> StandardScaler
    whiten: bool = False,
    random_state: int | None = 42,
    return_models: bool = False,  # True -> also return (imputer, scaler, pca)
) -> PcaResult:
    """
    Perform PCA on a wide-format DataFrame.

    Input contract (simplified):
      - df is wide format: rows = samples, columns = variables/features
      - df.index uniquely identifies each sample (can be MultiIndex, e.g., (ID, Phase))
      - df.columns are feature names
      - values should be numeric (missing values allowed)

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format input data.
    n_components : None | int | float
        Number of PCs to keep (int), or variance threshold (float in (0,1]).
        If None, keep all components.
    impute : {"mean", "median", None}
        Strategy to fill missing values. If None, no imputation is performed.
    scale : bool
        If True, standardize features to zero mean and unit variance before PCA.
    whiten : bool
        If True, whiten the components (see sklearn PCA docs).
    random_state : int
        Random seed for PCA (relevant for some SVD solvers).
    return_models : bool
        If True, also return fitted (imputer, scaler, pca) objects.

    Returns
    -------
    dict
        Dictionary containing:
        - "df_scores": PCA scores with the same index as input samples.
        - "explained_variance_ratio": explained variance ratio per component.
        - "df_loadings": PCA loadings (features x PCs).
        - "df_x_processed": processed matrix after imputation/scaling.
        - "feature_names": feature names used in PCA (order-consistent).
        - "pc_columns": principal component column names (PC1, PC2, ...).
        If return_models is True, also includes:
        - "imputer_obj", "scaler_obj", "pca_obj".
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    X: pd.DataFrame = df.copy()

    # 1) Force numeric (non-numeric becomes NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # 2) Drop all-NaN feature columns and all-NaN sample rows
    X = X.dropna(axis=1, how="all")
    X = X.dropna(axis=0, how="all")

    if X.shape[1] == 0:
        raise ValueError("No valid numeric feature columns available for PCA.")
    if X.shape[0] < 2:
        raise ValueError("PCA requires at least 2 valid samples.")

    # 3) Impute missing values
    imputer_obj: SimpleImputer | None = None
    X_arr: np.ndarray = X.to_numpy()
    if impute is not None:
        imputer_obj = SimpleImputer(strategy=impute)
        X_arr = imputer_obj.fit_transform(X_arr)

    # 4) Standardize features
    scaler_obj: StandardScaler | None = None
    if scale:
        scaler_obj = StandardScaler()
        X_arr = scaler_obj.fit_transform(X_arr)

    # 5) PCA
    pca_obj: PCA = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    scores: np.ndarray = pca_obj.fit_transform(X_arr)

    pc_cols: list[str] = [f"PC{i+1}" for i in range(scores.shape[1])]

    # Scores keep the original sample index
    df_scores: pd.DataFrame = pd.DataFrame(scores, index=X.index, columns=pc_cols)

    explained_variance_ratio: np.ndarray = pca_obj.explained_variance_ratio_

    # Loadings: features × PCs
    df_loadings: pd.DataFrame = pd.DataFrame(
        pca_obj.components_.T,
        index=X.columns,
        columns=pc_cols
    )

    # Processed X (after impute + optional scaling)
    df_x_processed: pd.DataFrame = pd.DataFrame(X_arr, index=X.index, columns=X.columns)

    result: PcaResult = {
        "df_scores": df_scores,
        "explained_variance_ratio": explained_variance_ratio,
        "df_loadings": df_loadings,
        "df_x_processed": df_x_processed,
        "feature_names": list(X.columns),
        "pc_columns": pc_cols,
    }

    if return_models:
        result["imputer_obj"] = imputer_obj
        result["scaler_obj"] = scaler_obj
        result["pca_obj"] = pca_obj

    return result


def project_improvement_vector_to_pcs(
    fit_result: PcaResult | dict[str, Any],
    feature_directions: Mapping[str, int | float] | pd.Series,
    n_components: int | Literal["all"] = 2,
) -> PcaImprovementProjection:
    """
    Map a feature-level improvement direction vector into top-k PC space.

    Given feature directions `g_j in {+1, -1}`:
      - `g_hat = g / ||g||`
      - `u = W_k^T g_hat`, where `W_k` is the (features x k) loading matrix
      - `u_hat = u / ||u||`

    Parameters
    ----------
    fit_result : PcaResult | dict[str, Any]
        Output from `perform_PCA`, must contain key "df_loadings".
    feature_directions : Mapping[str, int | float] | pd.Series
        Feature -> direction mapping (+1 means higher is better, -1 means lower is better).
        The mapping must cover all features used in PCA exactly.
    n_components : int | "all"
        Number of PCs to use. "all" uses all available PCs.

    Returns
    -------
    PcaImprovementProjection
        Dictionary with:
        - "g": original feature direction vector.
        - "g_hat": normalized feature direction vector.
        - "u": projected PC-space vector.
        - "u_hat": normalized projected PC-space vector (target output).
    """
    if not isinstance(fit_result, dict):
        raise TypeError("fit_result must be a dict-like object returned by perform_PCA.")
    if "df_loadings" not in fit_result:
        raise KeyError("fit_result must include key 'df_loadings'.")

    df_loadings = fit_result["df_loadings"]
    if not isinstance(df_loadings, pd.DataFrame):
        raise TypeError("fit_result['df_loadings'] must be a pandas DataFrame.")
    if df_loadings.empty:
        raise ValueError("fit_result['df_loadings'] is empty.")

    if isinstance(n_components, str):
        if n_components != "all":
            raise ValueError("n_components must be an integer >= 1 or 'all'.")
        k = df_loadings.shape[1]
    else:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be an integer >= 1 or 'all'.")
        if n_components > df_loadings.shape[1]:
            raise ValueError(
                f"n_components={n_components} exceeds available PCs ({df_loadings.shape[1]})."
            )
        k = n_components

    if isinstance(feature_directions, pd.Series):
        dir_series = feature_directions.copy()
    elif isinstance(feature_directions, Mapping):
        dir_series = pd.Series(dict(feature_directions))
    else:
        raise TypeError("feature_directions must be a mapping or pandas Series.")

    if dir_series.empty:
        raise ValueError("feature_directions cannot be empty.")
    if dir_series.index.has_duplicates:
        raise ValueError("feature_directions contains duplicate feature names.")

    dir_series = dir_series.apply(pd.to_numeric, errors="coerce")
    if dir_series.isna().any():
        bad_features = dir_series.index[dir_series.isna()].tolist()
        raise ValueError(f"feature_directions has non-numeric direction values for: {bad_features}.")

    extra_features = [feat for feat in dir_series.index if feat not in df_loadings.index]
    if extra_features:
        raise ValueError(
            f"feature_directions includes features not present in PCA loadings: {extra_features}."
        )

    g = dir_series.reindex(df_loadings.index)
    missing_features = g.index[g.isna()].tolist()
    if missing_features:
        raise ValueError(
            "feature_directions must cover all PCA features. "
            f"Missing directions for: {missing_features}."
        )

    invalid_values = g[~g.isin([-1.0, 1.0])]
    if not invalid_values.empty:
        invalid_feats = invalid_values.index.tolist()
        invalid_vals = invalid_values.tolist()
        raise ValueError(
            "feature_directions values must be exactly +1 or -1. "
            f"Invalid pairs: {list(zip(invalid_feats, invalid_vals))}."
        )

    g_arr = g.to_numpy(dtype=float)
    g_norm = float(np.linalg.norm(g_arr))
    if g_norm == 0.0:
        raise ValueError("Direction vector g has zero norm; cannot normalize.")
    g_hat_arr = g_arr / g_norm

    pc_cols = list(df_loadings.columns[:k])
    w_k = df_loadings.loc[:, pc_cols].to_numpy(dtype=float)  # features x k

    u_arr = w_k.T @ g_hat_arr
    u_norm = float(np.linalg.norm(u_arr))
    if u_norm == 0.0:
        raise ValueError("Projected vector u has zero norm; cannot normalize.")
    u_hat_arr = u_arr / u_norm

    g_series = pd.Series(g_arr, index=df_loadings.index, name="g")
    g_hat_series = pd.Series(g_hat_arr, index=df_loadings.index, name="g_hat")
    u_series = pd.Series(u_arr, index=pc_cols, name="u")
    u_hat_series = pd.Series(u_hat_arr, index=pc_cols, name="u_hat")

    return {
        "g": g_series,
        "g_hat": g_hat_series,
        "u": u_series,
        "u_hat": u_hat_series,
    }


def get_improvement_pc_unit_vector(
    fit_result: PcaResult | dict[str, Any],
    feature_directions: Mapping[str, int | float] | pd.Series,
    n_components: int | Literal["all"] = 2,
) -> pd.Series:
    """
    Convenience wrapper returning only the unit vector in PC space (`u_hat`).
    """
    projection = project_improvement_vector_to_pcs(
        fit_result=fit_result,
        feature_directions=feature_directions,
        n_components=n_components,
    )
    return projection["u_hat"]


def extract_top_features(
    loadings: pd.DataFrame,
    n_components: int | Literal["all"] = 3,
    top_n: int | Literal["all"] = 10,
) -> pd.DataFrame:
    ''' Extract the top N features for each principal component based on the loadings.

    Parameters:
    - loadings: The loadings DataFrame from PCA.
    - n_components: The number of principal components to extract top features for. Default is 3.
    - top_n: The number of top features to extract for each principal component. Default is 5.
    '''
    # Extract the top N features for each principal component
    top_features: dict[str, list[str]] = {}

    if n_components == 'all':
        n_components = len(loadings.columns)
    
    extract_col: list[str] = [f'PC{i+1}' for i in range(n_components)]

    if top_n == 'all':
        top_n = len(loadings)

    #for col in loadings.columns:
    for col in extract_col:
        top_features[col] = loadings[col].abs().sort_values(ascending=False).head(top_n).index.tolist()
    
    # Create a DataFrame with the top features
    df_top_features: pd.DataFrame = pd.DataFrame(top_features)

    return df_top_features



def get_dist(
    df: pd.DataFrame,
    match: str | list[str] | tuple[str, ...],
    baseline: dict[str, Any] | None = None,
    ncomponents: int = 2,
) -> pd.DataFrame:
    """
    Compute distances between matched samples in PCA space.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing PC coordinates and matching columns.
    match : str | list[str] | tuple[str, ...]
        Column(s) used to group samples before distance computation.
    baseline : dict[str, Any] | None
        If None, compute all pairwise distances within each match-group.
        If dict, select one baseline sample per group using exact equality
        on the provided columns, then compute distances from other samples
        to that baseline.
    ncomponents : int
        Number of principal components used to define the space
        (PC1..PC{ncomponents}).

    Returns
    -------
    pd.DataFrame
        Distance table with columns:
        - match columns
        - sample_ref: tuple of non-match/non-PC values for reference row
        - sample_cmp: tuple of non-match/non-PC values for compared row
        - distance: Euclidean distance
        - mode: {"pairwise", "to_baseline"}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if isinstance(ncomponents, bool) or not isinstance(ncomponents, int) or ncomponents < 1:
        raise ValueError("ncomponents must be an integer >= 1.")

    if isinstance(match, str):
        match_cols: list[str] = [match]
    else:
        match_cols = list(match)
    if len(match_cols) == 0:
        raise ValueError("match must contain at least one column name.")

    missing_match_cols = [col for col in match_cols if col not in df.columns]
    if missing_match_cols:
        raise ValueError(f"match columns not found in df: {missing_match_cols}.")

    pc_cols: list[str] = [f"PC{i+1}" for i in range(ncomponents)]
    missing_pc_cols = [col for col in pc_cols if col not in df.columns]
    if missing_pc_cols:
        raise ValueError(
            f"Required PC columns are missing: {missing_pc_cols}. "
            f"Please ensure df has columns {pc_cols}."
        )
    if df[pc_cols].isna().any().any():
        raise ValueError(f"NaN values found in {pc_cols}. Please clean or impute before calling get_dist.")

    if baseline is not None and not isinstance(baseline, dict):
        raise TypeError("baseline must be a dict[str, Any] or None.")
    if baseline is not None:
        missing_baseline_cols = [col for col in baseline if col not in df.columns]
        if missing_baseline_cols:
            raise ValueError(f"baseline columns not found in df: {missing_baseline_cols}.")

    def _is_pc_column(col: Any) -> bool:
        return isinstance(col, str) and col.startswith("PC") and col[2:].isdigit()

    all_pc_cols = [col for col in df.columns if _is_pc_column(col)]
    excluded_cols = set(match_cols) | set(pc_cols) | set(all_pc_cols)
    sample_meta_cols = [col for col in df.columns if col not in excluded_cols]

    records: list[dict[str, Any]] = []
    grouped = df.groupby(match_cols, observed=True, dropna=False)

    for group_key, df_group in grouped:
        if len(match_cols) == 1:
            group_key_tuple = (group_key,)
        else:
            group_key_tuple = tuple(group_key)
        group_info = {col: value for col, value in zip(match_cols, group_key_tuple)}
        if sample_meta_cols:
            sample_meta_map = {
                idx: tuple(values)
                for idx, values in zip(
                    df_group.index,
                    df_group[sample_meta_cols].itertuples(index=False, name=None),
                )
            }
        else:
            sample_meta_map = {idx: tuple() for idx in df_group.index}

        if baseline is None:
            if len(df_group) < 2:
                continue
            idx_values = list(df_group.index)
            coords = df_group[pc_cols].to_numpy(dtype=float)
            for idx_i, idx_j in itertools.combinations(range(len(df_group)), 2):
                distance_value = float(np.linalg.norm(coords[idx_i] - coords[idx_j]))
                records.append(
                    {
                        **group_info,
                        "sample_ref": sample_meta_map[idx_values[idx_i]],
                        "sample_cmp": sample_meta_map[idx_values[idx_j]],
                        "distance": distance_value,
                        "mode": "pairwise",
                    }
                )
            continue

        baseline_mask = np.ones(len(df_group), dtype=bool)
        for col, value in baseline.items():
            baseline_mask &= df_group[col].eq(value).to_numpy()

        df_baseline = df_group.loc[baseline_mask]
        if df_baseline.empty:
            raise ValueError(
                f"No baseline sample found for group {group_info} with baseline={baseline}."
            )
        if len(df_baseline) > 1:
            raise ValueError(
                f"Multiple baseline samples found for group {group_info} with baseline={baseline}. "
                "Expected exactly one."
            )

        baseline_idx = df_baseline.index[0]
        baseline_coord = df_baseline.iloc[0][pc_cols].to_numpy(dtype=float)
        df_others = df_group.drop(index=baseline_idx)
        for sample_idx, sample_row in df_others.iterrows():
            sample_coord = sample_row[pc_cols].to_numpy(dtype=float)
            distance_value = float(np.linalg.norm(sample_coord - baseline_coord))
            records.append(
                {
                    **group_info,
                    "sample_ref": sample_meta_map[baseline_idx],
                    "sample_cmp": sample_meta_map[sample_idx],
                    "distance": distance_value,
                    "mode": "to_baseline",
                }
            )

    output_cols = [*match_cols, "sample_ref", "sample_cmp", "distance", "mode"]
    return pd.DataFrame.from_records(records, columns=output_cols)


def get_signed_dist(
    df: pd.DataFrame,
    match: str | list[str] | tuple[str, ...],
    vector: pd.Series | np.ndarray | list[float] | tuple[float, ...],
    baseline: dict[str, Any] | None = None,
    ncomponents: int = 2,
) -> pd.DataFrame:
    """
    Compute signed projected distances between matched samples in PCA space.

    This function follows the same grouping/baseline logic as `get_dist`, but
    replaces Euclidean distance with the scalar projection of direction vector
    onto a target PC-space vector.

    For each pair (ref -> cmp), define:
      - delta = cmp - ref  (in PC1..PC{ncomponents})
      - signed_distance = dot(delta, vector_hat)

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing PC coordinates and matching columns.
    match : str | list[str] | tuple[str, ...]
        Column(s) used to group samples before signed distance computation.
    vector : pd.Series | np.ndarray | list[float] | tuple[float, ...]
        Projection direction in PC space. Recommended to pass output from
        `get_improvement_pc_unit_vector`. If not unit-length, it will be normalized.
        If pd.Series is provided, it should be indexed by PC names (e.g., PC1, PC2).
    baseline : dict[str, Any] | None
        If None, compute all pairwise signed projections within each match-group.
        If dict, select one baseline sample per group using exact equality
        on the provided columns, then compute signed projections from baseline
        to each other sample in the group.
    ncomponents : int
        Number of principal components used to define the space
        (PC1..PC{ncomponents}).

    Returns
    -------
    pd.DataFrame
        Signed-distance table with columns:
        - match columns
        - sample_ref: tuple of non-match/non-PC values for reference row
        - sample_cmp: tuple of non-match/non-PC values for compared row
        - distance: scalar projection on the input vector
        - mode: {"pairwise", "to_baseline"}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if isinstance(ncomponents, bool) or not isinstance(ncomponents, int) or ncomponents < 1:
        raise ValueError("ncomponents must be an integer >= 1.")

    if isinstance(match, str):
        match_cols: list[str] = [match]
    else:
        match_cols = list(match)
    if len(match_cols) == 0:
        raise ValueError("match must contain at least one column name.")

    missing_match_cols = [col for col in match_cols if col not in df.columns]
    if missing_match_cols:
        raise ValueError(f"match columns not found in df: {missing_match_cols}.")

    pc_cols: list[str] = [f"PC{i+1}" for i in range(ncomponents)]
    missing_pc_cols = [col for col in pc_cols if col not in df.columns]
    if missing_pc_cols:
        raise ValueError(
            f"Required PC columns are missing: {missing_pc_cols}. "
            f"Please ensure df has columns {pc_cols}."
        )
    if df[pc_cols].isna().any().any():
        raise ValueError(
            f"NaN values found in {pc_cols}. Please clean or impute before calling get_signed_distance."
        )

    if baseline is not None and not isinstance(baseline, dict):
        raise TypeError("baseline must be a dict[str, Any] or None.")
    if baseline is not None:
        missing_baseline_cols = [col for col in baseline if col not in df.columns]
        if missing_baseline_cols:
            raise ValueError(f"baseline columns not found in df: {missing_baseline_cols}.")

    if isinstance(vector, pd.Series):
        vec_series = vector.copy()
        missing_vec_pc = [col for col in pc_cols if col not in vec_series.index]
        if missing_vec_pc:
            raise ValueError(
                f"vector is missing required PC entries: {missing_vec_pc}. "
                f"Expected at least {pc_cols}."
            )
        vec_arr = vec_series.reindex(pc_cols).to_numpy(dtype=float)
    else:
        vec_arr = np.asarray(vector, dtype=float)
        if vec_arr.ndim != 1:
            raise ValueError("vector must be one-dimensional.")
        if vec_arr.shape[0] != ncomponents:
            raise ValueError(
                f"vector length ({vec_arr.shape[0]}) must equal ncomponents ({ncomponents})."
            )

    if np.isnan(vec_arr).any():
        raise ValueError("vector contains NaN values.")
    vec_norm = float(np.linalg.norm(vec_arr))
    if vec_norm == 0.0:
        raise ValueError("vector has zero norm; cannot normalize.")
    vec_hat = vec_arr / vec_norm

    def _is_pc_column(col: Any) -> bool:
        return isinstance(col, str) and col.startswith("PC") and col[2:].isdigit()

    all_pc_cols = [col for col in df.columns if _is_pc_column(col)]
    excluded_cols = set(match_cols) | set(pc_cols) | set(all_pc_cols)
    sample_meta_cols = [col for col in df.columns if col not in excluded_cols]

    records: list[dict[str, Any]] = []
    grouped = df.groupby(match_cols, observed=True, dropna=False)

    for group_key, df_group in grouped:
        if len(match_cols) == 1:
            group_key_tuple = (group_key,)
        else:
            group_key_tuple = tuple(group_key)
        group_info = {col: value for col, value in zip(match_cols, group_key_tuple)}

        if sample_meta_cols:
            sample_meta_map = {
                idx: tuple(values)
                for idx, values in zip(
                    df_group.index,
                    df_group[sample_meta_cols].itertuples(index=False, name=None),
                )
            }
        else:
            sample_meta_map = {idx: tuple() for idx in df_group.index}

        if baseline is None:
            if len(df_group) < 2:
                continue
            idx_values = list(df_group.index)
            coords = df_group[pc_cols].to_numpy(dtype=float)
            for idx_i, idx_j in itertools.combinations(range(len(df_group)), 2):
                delta = coords[idx_j] - coords[idx_i]  # ref -> cmp
                signed_distance_value = float(np.dot(delta, vec_hat))
                records.append(
                    {
                        **group_info,
                        "sample_ref": sample_meta_map[idx_values[idx_i]],
                        "sample_cmp": sample_meta_map[idx_values[idx_j]],
                        "distance": signed_distance_value,
                        "mode": "pairwise",
                    }
                )
            continue

        baseline_mask = np.ones(len(df_group), dtype=bool)
        for col, value in baseline.items():
            baseline_mask &= df_group[col].eq(value).to_numpy()

        df_baseline = df_group.loc[baseline_mask]
        if df_baseline.empty:
            raise ValueError(
                f"No baseline sample found for group {group_info} with baseline={baseline}."
            )
        if len(df_baseline) > 1:
            raise ValueError(
                f"Multiple baseline samples found for group {group_info} with baseline={baseline}. "
                "Expected exactly one."
            )

        baseline_idx = df_baseline.index[0]
        baseline_coord = df_baseline.iloc[0][pc_cols].to_numpy(dtype=float)
        df_others = df_group.drop(index=baseline_idx)
        for sample_idx, sample_row in df_others.iterrows():
            sample_coord = sample_row[pc_cols].to_numpy(dtype=float)
            delta = sample_coord - baseline_coord  # baseline -> sample
            signed_distance_value = float(np.dot(delta, vec_hat))
            records.append(
                {
                    **group_info,
                    "sample_ref": sample_meta_map[baseline_idx],
                    "sample_cmp": sample_meta_map[sample_idx],
                    "distance": signed_distance_value,
                    "mode": "to_baseline",
                }
            )

    output_cols = [*match_cols, "sample_ref", "sample_cmp", "distance", "mode"]
    return pd.DataFrame.from_records(records, columns=output_cols)


# %%
