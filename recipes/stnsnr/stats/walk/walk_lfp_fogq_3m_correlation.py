from __future__ import annotations

import os

from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stats.correlation.corr_matrix import CorrelationMatrixResult, compute_correlation_matrix
from recipes.stnsnr.stats.walk.walk_lfp_gait_pca_3m_correlation import (
    BAND_COL,
    BAND_ORDER,
    CONNECTIVITY_FEATURE_ORDER,
    CONNECTIVITY_REGION,
    CONNECTIVITY_SPECS,
    LOCAL_BAND_SPECS,
    LOCAL_FEATURE_ORDER_BY_REGION,
    LOCAL_REGIONS,
    PAIR_COL,
    PARAM_ORDER,
    PARAMS_INPUT_PATH,
    POSE_COL,
    POSE_ORDER,
    REGION_COL,
    VALUE_COL,
    WALK_CONTEXT,
    WALK_DIR as SOURCE_WALK_DIR,
    load_normalized_scalar_table,
)


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
WALK_DIR = PROJECT / "summary" / "stats" / "walk"
OUTPUT_DIR = WALK_DIR / "corr" / "walk_lfp_fogq_3m"
FOGQ_INPUT_PATH = PROJECT / "summary" / "cohort" / "subj" / "subj_delta_effect.xlsx"

FOGQ_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_fogq_3m_subject_wide.xlsx"
FOGQ_SUBSET_OUTPUT_PATH = OUTPUT_DIR / "walk_fogq_3m_subset_for_corr.xlsx"
STN_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_single_stn_local_subject_normalized_wide_by_pose.xlsx"
SNR_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_single_snr_local_subject_normalized_wide_by_pose.xlsx"
CONNECTIVITY_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_single_connectivity_subject_normalized_wide_by_pose.xlsx"
STN_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "walk_single_stn_local_fogq_3m_corr_by_pose.xlsx"
SNR_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "walk_single_snr_local_fogq_3m_corr_by_pose.xlsx"
CONNECTIVITY_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "walk_single_connectivity_fogq_3m_corr_by_pose.xlsx"

FOGQ_COLUMNS = [
    "FOG-Q Δ% (STN, 3m)",
    "ΔFOG-Q Δ% (+SNr, 3m)",
]


def load_fogq_3m_table(input_path: str | Path = FOGQ_INPUT_PATH) -> pd.DataFrame:
    """Load the established 3-month STN and incremental SNr FOGQ effects."""
    input_path = Path(input_path)
    fogq = pd.read_excel(input_path)
    _validate_required_columns(
        fogq,
        required_columns=[PAIR_COL, *FOGQ_COLUMNS],
        table_name=str(input_path),
    )

    fogq = _normalize_pair_column(
        fogq.loc[:, [PAIR_COL, *FOGQ_COLUMNS]].copy(),
        pair_col=PAIR_COL,
    )
    if fogq[PAIR_COL].duplicated().any():
        raise ValueError(f"{input_path} contains duplicate IDs.")

    for column in FOGQ_COLUMNS:
        fogq[column] = pd.to_numeric(fogq[column], errors="coerce")

    return fogq.sort_values(PAIR_COL).reset_index(drop=True)


def load_local_wide_by_pose(region: str) -> dict[str, pd.DataFrame]:
    """Load local normalized walking LFP pose effects for one region."""
    if region not in LOCAL_REGIONS:
        raise ValueError(f"Unsupported local region: {region!r}. Expected one of {LOCAL_REGIONS}.")

    tables_by_pose: dict[str, pd.DataFrame] = {}
    for pose in POSE_ORDER:
        feature_rows: list[pd.DataFrame] = []
        for spec in LOCAL_BAND_SPECS:
            input_path = _mounted_source_path(spec.input_path)
            scalar = load_normalized_scalar_table(input_path)
            filtered = _filter_walk_lfp_rows(
                scalar,
                region=region,
                pose=pose,
                required_bands=BAND_ORDER,
                table_name=str(input_path),
            )
            filtered["feature"] = [
                f"{region}__{band}__{spec.key}"
                for band in filtered[BAND_COL].astype(str)
            ]
            feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        params_input_path = _mounted_source_path(PARAMS_INPUT_PATH)
        params = load_normalized_scalar_table(params_input_path)
        params_filtered = _filter_walk_lfp_rows(
            params,
            region=region,
            pose=pose,
            required_bands=PARAM_ORDER,
            table_name=str(params_input_path),
        )
        params_filtered["feature"] = [
            f"{region}__params__{parameter}"
            for parameter in params_filtered[BAND_COL].astype(str)
        ]
        feature_rows.append(params_filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        tables_by_pose[pose] = _pivot_feature_rows(
            feature_rows,
            ordered_features=LOCAL_FEATURE_ORDER_BY_REGION[region],
            table_name=f"{region} local normalized walk LFP table for {pose}",
        )

    return tables_by_pose


def load_connectivity_wide_by_pose() -> dict[str, pd.DataFrame]:
    """Load normalized walking connectivity pose effects."""
    tables_by_pose: dict[str, pd.DataFrame] = {}
    for pose in POSE_ORDER:
        feature_rows: list[pd.DataFrame] = []
        for spec in CONNECTIVITY_SPECS:
            input_path = _mounted_source_path(spec.input_path)
            scalar = load_normalized_scalar_table(input_path)
            filtered = _filter_walk_lfp_rows(
                scalar,
                region=CONNECTIVITY_REGION,
                pose=pose,
                required_bands=BAND_ORDER,
                table_name=str(input_path),
            )
            filtered["feature"] = [
                f"{CONNECTIVITY_REGION}__{band}__{spec.key}"
                for band in filtered[BAND_COL].astype(str)
            ]
            feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        tables_by_pose[pose] = _pivot_feature_rows(
            feature_rows,
            ordered_features=CONNECTIVITY_FEATURE_ORDER,
            table_name=f"connectivity normalized walk LFP table for {pose}",
        )

    return tables_by_pose


def filter_to_common_ids(
    x_wide: pd.DataFrame,
    fogq: pd.DataFrame,
    pair_col: str = PAIR_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter walking LFP and FOGQ tables to shared IDs in the same order."""
    x_wide = _normalize_pair_column(x_wide.copy(), pair_col=pair_col)
    fogq = _normalize_pair_column(fogq.copy(), pair_col=pair_col)

    if x_wide[pair_col].duplicated().any():
        raise ValueError("LFP wide table contains duplicate IDs.")
    if fogq[pair_col].duplicated().any():
        raise ValueError("FOGQ table contains duplicate IDs.")

    common_ids = sorted(set(x_wide[pair_col]).intersection(fogq[pair_col]))
    if not common_ids:
        raise ValueError("No shared IDs were found between walk LFP features and FOGQ table.")

    x_common = (
        x_wide.loc[x_wide[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )
    fogq_common = (
        fogq.loc[fogq[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )

    if x_common[pair_col].tolist() != fogq_common[pair_col].tolist():
        raise ValueError("Filtered LFP and FOGQ tables do not share the same ordered IDs.")

    return x_common, fogq_common


def compute_walk_fogq_correlation(
    x_common: pd.DataFrame,
    fogq_common: pd.DataFrame,
) -> CorrelationMatrixResult:
    """Compute globally FDR-adjusted Spearman correlations."""
    return compute_correlation_matrix(
        df_x=x_common,
        df_y=fogq_common,
        pair_col=PAIR_COL,
        method="spearman",
        correction_mode="All",
        correction_method="bh",
        min_n=3,
    )


def export_pose_workbook(
    pose_tables: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    """Export one subject-wide workbook with one sheet per walking pose."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for pose in POSE_ORDER:
            pose_tables[pose].to_excel(writer, sheet_name=pose, index=False)


def export_fogq_subset_workbook(
    fogq_subsets: dict[str, pd.DataFrame],
    output_path: str | Path = FOGQ_SUBSET_OUTPUT_PATH,
) -> None:
    """Export the FOGQ subsets used for every analysis group and pose."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, fogq_subset in fogq_subsets.items():
            fogq_subset.to_excel(writer, sheet_name=sheet_name, index=False)


def export_correlation_workbook_by_pose(
    correlation_results: dict[str, CorrelationMatrixResult],
    output_path: str | Path,
) -> None:
    """Export pose-specific correlation matrices to a multi-sheet workbook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for pose in POSE_ORDER:
            result = correlation_results[pose]
            result.r.to_excel(writer, sheet_name=f"{pose}_r")
            result.p.to_excel(writer, sheet_name=f"{pose}_p")
            result.p_raw.to_excel(writer, sheet_name=f"{pose}_p_raw")
            if result.p_adj is None:
                raise ValueError(f"{pose}: expected adjusted p-values, but p_adj is None.")
            result.p_adj.to_excel(writer, sheet_name=f"{pose}_p_adj")
            result.n.to_excel(writer, sheet_name=f"{pose}_n")


def run_workflow() -> dict[str, dict[str, CorrelationMatrixResult]]:
    """Run the complete walking LFP-FOGQ 3-month correlation workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fogq = load_fogq_3m_table(FOGQ_INPUT_PATH)
    fogq.to_excel(FOGQ_WIDE_OUTPUT_PATH, index=False)

    wide_tables = {
        "stn_local": load_local_wide_by_pose("STN"),
        "snr_local": load_local_wide_by_pose("SNr"),
        "connectivity": load_connectivity_wide_by_pose(),
    }
    output_specs = {
        "stn_local": {
            "wide_path": STN_LOCAL_WIDE_OUTPUT_PATH,
            "corr_path": STN_LOCAL_CORRELATION_OUTPUT_PATH,
            "features": LOCAL_FEATURE_ORDER_BY_REGION["STN"],
        },
        "snr_local": {
            "wide_path": SNR_LOCAL_WIDE_OUTPUT_PATH,
            "corr_path": SNR_LOCAL_CORRELATION_OUTPUT_PATH,
            "features": LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
        },
        "connectivity": {
            "wide_path": CONNECTIVITY_WIDE_OUTPUT_PATH,
            "corr_path": CONNECTIVITY_CORRELATION_OUTPUT_PATH,
            "features": CONNECTIVITY_FEATURE_ORDER,
        },
    }

    results: dict[str, dict[str, CorrelationMatrixResult]] = {}
    fogq_subsets: dict[str, pd.DataFrame] = {}
    for group_name, pose_tables in wide_tables.items():
        output_spec = output_specs[group_name]
        pose_common_tables: dict[str, pd.DataFrame] = {}
        pose_results: dict[str, CorrelationMatrixResult] = {}

        for pose in POSE_ORDER:
            x_common, fogq_common = filter_to_common_ids(pose_tables[pose], fogq)
            pose_common_tables[pose] = x_common
            fogq_subsets[f"{group_name}_{pose}"] = fogq_common

            correlation_result = compute_walk_fogq_correlation(x_common, fogq_common)
            _validate_outputs(
                group_name=group_name,
                pose=pose,
                x_common=x_common,
                fogq_common=fogq_common,
                correlation_result=correlation_result,
                expected_features=output_spec["features"],
            )
            pose_results[pose] = correlation_result

        export_pose_workbook(pose_common_tables, output_spec["wide_path"])
        export_correlation_workbook_by_pose(pose_results, output_spec["corr_path"])
        results[group_name] = pose_results

    export_fogq_subset_workbook(fogq_subsets, FOGQ_SUBSET_OUTPUT_PATH)
    return results


def _mounted_source_path(source_path: str | Path) -> Path:
    source_path = Path(source_path)
    return WALK_DIR / source_path.relative_to(SOURCE_WALK_DIR)


def _filter_walk_lfp_rows(
    scalar: pd.DataFrame,
    *,
    region: str,
    pose: str,
    required_bands: list[str],
    table_name: str,
) -> pd.DataFrame:
    filtered = scalar.loc[scalar[REGION_COL] == region].copy()
    for column, expected in WALK_CONTEXT.items():
        filtered = filtered.loc[filtered[column] == expected].copy()
    filtered = filtered.loc[filtered[POSE_COL] == pose].copy()

    if filtered.empty:
        raise ValueError(
            f"{table_name} has no rows for Region={region!r}, Pose={pose!r}, "
            f"and context {WALK_CONTEXT}."
        )

    available_bands = set(filtered[BAND_COL].astype(str))
    missing_bands = [band for band in required_bands if band not in available_bands]
    unexpected_bands = sorted(available_bands - set(required_bands))
    if missing_bands:
        raise ValueError(
            f"{table_name} is missing required bands for {region} {pose}: {missing_bands}."
        )
    if unexpected_bands:
        raise ValueError(
            f"{table_name} has unexpected bands for {region} {pose}: {unexpected_bands}."
        )

    return filtered.loc[filtered[BAND_COL].isin(required_bands)].copy()


def _pivot_feature_rows(
    feature_rows: list[pd.DataFrame],
    ordered_features: list[str],
    table_name: str,
) -> pd.DataFrame:
    if not feature_rows:
        raise ValueError(f"{table_name} has no feature rows to pivot.")

    long = pd.concat(feature_rows, ignore_index=True)
    duplicate_mask = long.duplicated([PAIR_COL, "feature"], keep=False)
    if duplicate_mask.any():
        examples = (
            long.loc[duplicate_mask, [PAIR_COL, "feature"]]
            .drop_duplicates()
            .sort_values([PAIR_COL, "feature"])
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            f"{table_name} contains duplicate '{PAIR_COL}' + 'feature' combinations. "
            f"Examples: {examples}"
        )

    wide = long.pivot(index=PAIR_COL, columns="feature", values=VALUE_COL)
    missing_features = [feature for feature in ordered_features if feature not in wide.columns]
    if missing_features:
        raise ValueError(f"{table_name} is missing expected features: {missing_features}.")

    wide = wide.loc[:, ordered_features].sort_index(axis=0).reset_index()
    wide.columns.name = None
    return wide


def _validate_outputs(
    *,
    group_name: str,
    pose: str,
    x_common: pd.DataFrame,
    fogq_common: pd.DataFrame,
    correlation_result: CorrelationMatrixResult,
    expected_features: list[str],
) -> None:
    if x_common.columns.tolist()[1:] != expected_features:
        raise ValueError(f"{group_name} {pose}: x-axis features do not match the configured order.")
    if fogq_common.columns.tolist()[1:] != FOGQ_COLUMNS:
        raise ValueError(f"{group_name} {pose}: FOGQ columns do not match the configured order.")

    expected_shape = (len(FOGQ_COLUMNS), len(expected_features))
    if correlation_result.r.shape != expected_shape:
        raise ValueError(
            f"{group_name} {pose}: expected correlation matrix shape {expected_shape}; "
            f"got {correlation_result.r.shape}."
        )

    matrices = {
        "p": correlation_result.p,
        "p_raw": correlation_result.p_raw,
        "n": correlation_result.n,
    }
    if correlation_result.p_adj is None:
        raise ValueError(f"{group_name} {pose}: expected globally adjusted p-values.")
    matrices["p_adj"] = correlation_result.p_adj

    for name, matrix in matrices.items():
        index_aligned = correlation_result.r.index.equals(matrix.index)
        columns_aligned = correlation_result.r.columns.equals(matrix.columns)
        if not index_aligned or not columns_aligned:
            raise ValueError(f"{group_name} {pose}: matrix '{name}' is not aligned with r.")


def _normalize_pair_column(df: pd.DataFrame, pair_col: str) -> pd.DataFrame:
    _validate_required_columns(df, required_columns=[pair_col], table_name="input table")
    if df[pair_col].isna().any():
        raise ValueError(f"Column '{pair_col}' contains missing values.")

    normalized = df.copy()
    normalized[pair_col] = normalized[pair_col].astype(str)
    return normalized


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(missing_columns)}."
        )


if __name__ == "__main__":
    workflow_results = run_workflow()
    print(f"Saved FOGQ 3-month table: {FOGQ_WIDE_OUTPUT_PATH}")
    print(f"Saved FOGQ subset workbook: {FOGQ_SUBSET_OUTPUT_PATH}")
    print(f"Saved STN local wide workbook: {STN_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved SNr local wide workbook: {SNR_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved connectivity wide workbook: {CONNECTIVITY_WIDE_OUTPUT_PATH}")
    print(f"Saved STN local correlation workbook: {STN_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved SNr local correlation workbook: {SNR_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved connectivity correlation workbook: {CONNECTIVITY_CORRELATION_OUTPUT_PATH}")
    for group, pose_results in workflow_results.items():
        for pose_name, result in pose_results.items():
            print(f"{group} {pose_name}: correlation r shape {result.r.shape}")
