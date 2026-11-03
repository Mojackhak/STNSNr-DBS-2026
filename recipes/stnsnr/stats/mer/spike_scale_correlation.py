from __future__ import annotations

import os

from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipes.stnsnr.stats.clinic.subject_effect_tables import BACKUP_SCALE_COLUMN_NAMES
from src.stats.correlation.corr_matrix import CorrelationMatrixResult, compute_correlation_matrix


PAIR_COL = "ID"
VALUE_COL = "Value"

PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
SPIKE_DIR = PROJECT / "summary" / "stats" / "mer" / "spike"
SPIKE_INPUT_PATH = SPIKE_DIR / "spike_subject_raw.pkl"
SCALE_INPUT_PATH = PROJECT / "summary" / "cohort" / "subj" / "subj_scale.xlsx"
OUTPUT_DIR = SPIKE_DIR / "corr"

CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "mer_spike_scale_corr.xlsx"
SPIKE_WIDE_OUTPUT_PATH = OUTPUT_DIR / "mer_spike_subject_wide.xlsx"
SCALE_SUBSET_OUTPUT_PATH = OUTPUT_DIR / "subj_scale_subset_for_corr.xlsx"

SPIKE_PARAM_ORDER = [
    "Firing_rate",
    "Burst_index",
    "LvR",
    "Peak_trough_ratio",
    "Peak_to_valley",
    "Half_width",
    "Repolarization_slope",
    "Recovery_slope",
]

SPIKE_X_LABELS = [
    "Firing rate",
    "Burst index",
    "LvR",
    "Peak trough ratio",
    "Peak to valley",
    "Half width",
    "Repolarization slope",
    "Recovery slope",
]

Y_COLUMNS = [
    "Age",
    "BMI",
    "Education years",
    "Disease duration",
    "LCT improvement rate",
    "MDS-UPDRS I",
    "MDS-UPDRS II",
    "MDS-UPDRS III (Med-OFF)",
    "MDS-UPDRS III axial (Med-OFF)",
    "MDS-UPDRS III rigidity (Med-OFF)",
    "MDS-UPDRS III bradykinesia (Med-OFF)",
    "MDS-UPDRS III rest tremor (Med-OFF)",
    "MDS-UPDRS III action tremor (Med-OFF)",
    "MDS-UPDRS IV",
    "FOG-Q",
    "MoCA",
    "MMSE",
    "BDI",
    "MADRS",
    "HAMD",
    "HAMA",
    "MAES",
    "ESS",
    "RBDSQ",
    "RBDSQ-HK",
    "SCOPA-AUT",
    "SDQ",
    "EAT-10",
    "DSFS",
    "CCCS",
    "FSS",
    "KPPS",
    "ODQ",
    "ADL",
    "SE-ADL",
    "PDQ-39",
]


def load_spike_wide(input_path: str | Path = SPIKE_INPUT_PATH) -> pd.DataFrame:
    """Load MER spike subject data and pivot to one row per subject."""
    spike = pd.read_pickle(input_path)
    _validate_required_columns(
        spike,
        required_columns=[PAIR_COL, "Region", "Label", "Param", VALUE_COL],
        table_name=str(input_path),
    )

    prepared = spike.loc[:, [PAIR_COL, "Region", "Label", "Param", VALUE_COL]].copy()
    prepared = _normalize_pair_column(prepared, pair_col=PAIR_COL)
    prepared[VALUE_COL] = pd.to_numeric(prepared[VALUE_COL], errors="coerce")

    missing_params = sorted(set(SPIKE_PARAM_ORDER) - set(prepared["Param"].astype(str)))
    if missing_params:
        raise ValueError(f"Spike table is missing required Param values: {missing_params}.")

    unexpected_params = sorted(set(prepared["Param"].astype(str)) - set(SPIKE_PARAM_ORDER))
    if unexpected_params:
        raise ValueError(f"Spike table contains unexpected Param values: {unexpected_params}.")

    region_label = prepared[["Region", "Label"]].drop_duplicates()
    if len(region_label) != 1:
        rows = region_label.to_dict("records")
        raise ValueError(f"Expected one Region+Label combination; found {rows}.")

    prepared["feature"] = prepared[["Region", "Label", "Param"]].astype(str).agg("__".join, axis=1)
    _raise_on_duplicate_pair_feature(prepared, feature_col="feature", table_name="spike table")

    feature_prefix = "__".join(region_label.iloc[0].astype(str).tolist())
    ordered_features = [f"{feature_prefix}__{param}" for param in SPIKE_PARAM_ORDER]

    spike_wide = prepared.pivot(index=PAIR_COL, columns="feature", values=VALUE_COL)
    missing_features = [feature for feature in ordered_features if feature not in spike_wide.columns]
    if missing_features:
        raise ValueError(f"Spike wide table is missing expected features: {missing_features}.")

    spike_wide = spike_wide.loc[:, ordered_features].sort_index(axis=0).reset_index()
    spike_wide.columns.name = None
    return spike_wide


def load_scale_subset(input_path: str | Path = SCALE_INPUT_PATH) -> pd.DataFrame:
    """Load the clinical scale workbook and keep the planned correlation variables."""
    scale = pd.read_excel(input_path).rename(columns=BACKUP_SCALE_COLUMN_NAMES)
    keep_columns = [PAIR_COL, *Y_COLUMNS]
    _validate_required_columns(scale, required_columns=keep_columns, table_name=str(input_path))

    excluded = [col for col in scale.columns if "Med-ON" in str(col) or str(col) == "QUIP-RS"]
    leaked = [col for col in keep_columns if col in excluded]
    if leaked:
        raise ValueError(f"Excluded columns are present in the retained column list: {leaked}.")

    scale_subset = scale.loc[:, keep_columns].copy()
    scale_subset = _normalize_pair_column(scale_subset, pair_col=PAIR_COL)
    return scale_subset


def filter_to_common_ids(
    spike_wide: pd.DataFrame,
    scale_subset: pd.DataFrame,
    pair_col: str = PAIR_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter spike and scale tables to their shared IDs in the same order."""
    spike_wide = _normalize_pair_column(spike_wide.copy(), pair_col=pair_col)
    scale_subset = _normalize_pair_column(scale_subset.copy(), pair_col=pair_col)

    if spike_wide[pair_col].duplicated().any():
        raise ValueError("Spike wide table contains duplicate IDs.")
    if scale_subset[pair_col].duplicated().any():
        raise ValueError("Scale subset table contains duplicate IDs.")

    common_ids = sorted(set(spike_wide[pair_col]).intersection(scale_subset[pair_col]))
    if not common_ids:
        raise ValueError("No shared IDs were found between spike and scale tables.")

    spike_common = (
        spike_wide[spike_wide[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )
    scale_common = (
        scale_subset[scale_subset[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )

    if spike_common[pair_col].tolist() != scale_common[pair_col].tolist():
        raise ValueError("Filtered spike and scale tables do not share the same ordered IDs.")

    return spike_common, scale_common


def compute_spike_scale_correlation(
    spike_common: pd.DataFrame,
    scale_common: pd.DataFrame,
) -> CorrelationMatrixResult:
    """Compute globally FDR-adjusted Spearman correlations."""
    return compute_correlation_matrix(
        df_x=spike_common,
        df_y=scale_common,
        pair_col=PAIR_COL,
        method="spearman",
        correction_mode="All",
        correction_method="bh",
    )


def export_correlation_workbook(
    correlation_result: CorrelationMatrixResult,
    output_path: str | Path = CORRELATION_OUTPUT_PATH,
) -> None:
    """Export correlation matrices to a multi-sheet Excel workbook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        correlation_result.r.to_excel(writer, sheet_name="r")
        correlation_result.p.to_excel(writer, sheet_name="p")
        correlation_result.p_raw.to_excel(writer, sheet_name="p_raw")
        if correlation_result.p_adj is not None:
            correlation_result.p_adj.to_excel(writer, sheet_name="p_adj")
        correlation_result.n.to_excel(writer, sheet_name="n")


def run_workflow() -> CorrelationMatrixResult:
    """Run the full MER spike-scale correlation workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spike_wide = load_spike_wide(SPIKE_INPUT_PATH)
    scale_subset = load_scale_subset(SCALE_INPUT_PATH)
    spike_common, scale_common = filter_to_common_ids(spike_wide, scale_subset)

    spike_common.to_excel(SPIKE_WIDE_OUTPUT_PATH, index=False)
    scale_common.to_excel(SCALE_SUBSET_OUTPUT_PATH, index=False)

    correlation_result = compute_spike_scale_correlation(spike_common, scale_common)
    export_correlation_workbook(correlation_result, CORRELATION_OUTPUT_PATH)

    _validate_outputs(spike_common, scale_common, correlation_result)
    return correlation_result


def _validate_outputs(
    spike_common: pd.DataFrame,
    scale_common: pd.DataFrame,
    correlation_result: CorrelationMatrixResult,
) -> None:
    expected_x = spike_common.columns.tolist()[1:]
    expected_y = scale_common.columns.tolist()[1:]

    if expected_y != Y_COLUMNS:
        raise ValueError("Scale y-axis columns do not match the configured order.")
    if len(expected_x) != len(SPIKE_PARAM_ORDER):
        raise ValueError(f"Expected {len(SPIKE_PARAM_ORDER)} spike features; found {len(expected_x)}.")
    if any("Med-ON" in col for col in expected_y):
        raise ValueError("Scale y-axis columns contain Med-ON variables.")
    if "QUIP-RS" in expected_y:
        raise ValueError("Scale y-axis columns contain QUIP-RS.")

    expected_shape = (len(Y_COLUMNS), len(SPIKE_PARAM_ORDER))
    if correlation_result.r.shape != expected_shape:
        raise ValueError(f"Expected correlation matrix shape {expected_shape}; got {correlation_result.r.shape}.")

    for name, matrix in {
        "p": correlation_result.p,
        "p_raw": correlation_result.p_raw,
        "n": correlation_result.n,
    }.items():
        if not correlation_result.r.index.equals(matrix.index) or not correlation_result.r.columns.equals(matrix.columns):
            raise ValueError(f"Matrix '{name}' is not aligned with r.")

    if correlation_result.p_adj is None:
        raise ValueError("Expected globally adjusted p-values, but p_adj is None.")
    if not correlation_result.r.index.equals(correlation_result.p_adj.index) or not correlation_result.r.columns.equals(correlation_result.p_adj.columns):
        raise ValueError("Matrix 'p_adj' is not aligned with r.")


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


def _raise_on_duplicate_pair_feature(
    df: pd.DataFrame,
    feature_col: str,
    table_name: str,
) -> None:
    duplicate_mask = df.duplicated([PAIR_COL, feature_col], keep=False)
    if not duplicate_mask.any():
        return

    duplicate_rows = (
        df.loc[duplicate_mask, [PAIR_COL, feature_col]]
        .drop_duplicates()
        .sort_values([PAIR_COL, feature_col])
    )
    examples = duplicate_rows.head(5).to_dict("records")
    raise ValueError(
        f"{table_name} contains duplicate '{PAIR_COL}' + '{feature_col}' combinations. "
        f"Examples: {examples}"
    )


if __name__ == "__main__":
    result = run_workflow()
    print(f"Saved spike wide table: {SPIKE_WIDE_OUTPUT_PATH}")
    print(f"Saved scale subset table: {SCALE_SUBSET_OUTPUT_PATH}")
    print(f"Saved correlation workbook: {CORRELATION_OUTPUT_PATH}")
    print(f"Correlation r shape: {result.r.shape}")
