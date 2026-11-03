from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stats.correlation.corr_matrix import CorrelationMatrixResult
from recipes.stnsnr.stats.rest.rest_scale_correlation import (
    CONNECTIVITY_FEATURE_ORDER,
    LOCAL_FEATURE_ORDER_BY_REGION,
    OUTPUT_DIR,
    PAIR_COL,
    PROJECT,
    compute_rest_scale_correlation,
    export_correlation_workbook,
    export_scale_subset_workbook,
    filter_to_common_ids,
    load_connectivity_wide,
    load_local_wide,
)


SCALE_COL = "Scale"
REGION_COL = "Region"
VALUE_COL = "Value"

EFFICACY_INPUT_PATH = (
    PROJECT / "summary" / "stats" / "clinic" / "coords" / "scale" / "scale_subject.xlsx"
)

STN_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "rest_stn_local_efficacy_corr.xlsx"
SNR_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "rest_snr_local_delta_efficacy_corr.xlsx"
CONNECTIVITY_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "rest_connectivity_delta_efficacy_corr.xlsx"

STN_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "rest_stn_local_efficacy_subject_wide.xlsx"
SNR_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "rest_snr_local_delta_efficacy_subject_wide.xlsx"
CONNECTIVITY_WIDE_OUTPUT_PATH = OUTPUT_DIR / "rest_connectivity_delta_efficacy_subject_wide.xlsx"
EFFICACY_SUBSET_OUTPUT_PATH = OUTPUT_DIR / "scale_efficacy_subset_for_rest_corr.xlsx"

STN_EFFICACY_COLUMNS = [
    "UPDRS-III Δ% (STN, 3m)",
    "UPDRS-III axial Δ% (STN, 3m)",
    "FOG-Q Δ% (STN, 3m)",
    "MADRS Δ% (STN, 3m)",
    "KPPS Δ% (STN, 3m)",
    "ADL Δ% (STN, 3m)",
    "SE-ADL Δ% (STN, 3m)",
    "PDQ-39 Δ% (STN, 3m)",
]

DELTA_EFFICACY_COLUMNS = [
    "ΔUPDRS-III Δ% (+SNr, 3m)",
    "ΔUPDRS-III axial Δ% (+SNr, 3m)",
    "ΔFOG-Q Δ% (+SNr, 3m)",
    "ΔMADRS Δ% (+SNr, 3m)",
    "ΔKPPS Δ% (+SNr, 3m)",
    "ΔADL Δ% (+SNr, 3m)",
    "ΔSE-ADL Δ% (+SNr, 3m)",
    "ΔPDQ-39 Δ% (+SNr, 3m)",
]


def load_efficacy_subset(
    input_path: str | Path = EFFICACY_INPUT_PATH,
    efficacy_columns: list[str] | None = None,
    expected_region: str | None = None,
) -> pd.DataFrame:
    """Load selected clinical efficacy variables and pivot to subject-wide format."""
    if efficacy_columns is None:
        efficacy_columns = STN_EFFICACY_COLUMNS

    efficacy = pd.read_excel(input_path)
    _validate_required_columns(
        efficacy,
        required_columns=[PAIR_COL, REGION_COL, SCALE_COL, VALUE_COL],
        table_name=str(input_path),
    )
    efficacy = efficacy.loc[:, [PAIR_COL, REGION_COL, SCALE_COL, VALUE_COL]].copy()
    efficacy[PAIR_COL] = _normalize_pair_series(efficacy[PAIR_COL], pair_col=PAIR_COL)
    efficacy[REGION_COL] = efficacy[REGION_COL].astype(str)
    efficacy[SCALE_COL] = efficacy[SCALE_COL].astype(str)
    efficacy[VALUE_COL] = pd.to_numeric(efficacy[VALUE_COL], errors="coerce")

    _raise_on_duplicate_id_scale(efficacy, table_name=str(input_path))

    available_scales = set(efficacy[SCALE_COL])
    missing_scales = [scale for scale in efficacy_columns if scale not in available_scales]
    if missing_scales:
        raise ValueError(f"{input_path} is missing required efficacy scales: {missing_scales}.")

    selected = efficacy.loc[efficacy[SCALE_COL].isin(efficacy_columns)].copy()
    if expected_region is not None:
        observed_regions = sorted(set(selected[REGION_COL].dropna().astype(str)))
        if observed_regions != [expected_region]:
            raise ValueError(
                f"Selected efficacy rows have unexpected regions: {observed_regions}. "
                f"Expected only {expected_region!r}."
            )

    _validate_efficacy_columns(efficacy_columns)

    wide = selected.pivot(index=PAIR_COL, columns=SCALE_COL, values=VALUE_COL)
    missing_columns = [column for column in efficacy_columns if column not in wide.columns]
    if missing_columns:
        raise ValueError(f"Pivoted efficacy table is missing expected columns: {missing_columns}.")

    wide = wide.loc[:, efficacy_columns].sort_index(axis=0).reset_index()
    wide.columns.name = None
    return wide


def run_workflow() -> dict[str, CorrelationMatrixResult]:
    """Run the rest LFP feature-efficacy correlation workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stn_efficacy = load_efficacy_subset(
        EFFICACY_INPUT_PATH,
        efficacy_columns=STN_EFFICACY_COLUMNS,
        expected_region="STN",
    )
    delta_efficacy = load_efficacy_subset(
        EFFICACY_INPUT_PATH,
        efficacy_columns=DELTA_EFFICACY_COLUMNS,
        expected_region="SNr",
    )

    wide_tables = {
        "stn_local": load_local_wide("STN"),
        "snr_local": load_local_wide("SNr"),
        "connectivity": load_connectivity_wide(),
    }

    output_specs = {
        "stn_local": {
            "efficacy": stn_efficacy,
            "wide_path": STN_LOCAL_WIDE_OUTPUT_PATH,
            "corr_path": STN_LOCAL_CORRELATION_OUTPUT_PATH,
            "features": LOCAL_FEATURE_ORDER_BY_REGION["STN"],
            "efficacy_columns": STN_EFFICACY_COLUMNS,
        },
        "snr_local": {
            "efficacy": delta_efficacy,
            "wide_path": SNR_LOCAL_WIDE_OUTPUT_PATH,
            "corr_path": SNR_LOCAL_CORRELATION_OUTPUT_PATH,
            "features": LOCAL_FEATURE_ORDER_BY_REGION["SNr"],
            "efficacy_columns": DELTA_EFFICACY_COLUMNS,
        },
        "connectivity": {
            "efficacy": delta_efficacy,
            "wide_path": CONNECTIVITY_WIDE_OUTPUT_PATH,
            "corr_path": CONNECTIVITY_CORRELATION_OUTPUT_PATH,
            "features": CONNECTIVITY_FEATURE_ORDER,
            "efficacy_columns": DELTA_EFFICACY_COLUMNS,
        },
    }

    results: dict[str, CorrelationMatrixResult] = {}
    efficacy_subsets: dict[str, pd.DataFrame] = {}

    for group_name, x_wide in wide_tables.items():
        output_spec = output_specs[group_name]
        x_common, efficacy_common = filter_to_common_ids(x_wide, output_spec["efficacy"])

        x_common.to_excel(output_spec["wide_path"], index=False)
        efficacy_subsets[group_name] = efficacy_common

        correlation_result = compute_rest_scale_correlation(x_common, efficacy_common)
        export_correlation_workbook(correlation_result, output_spec["corr_path"])
        _validate_outputs(
            group_name=group_name,
            x_common=x_common,
            efficacy_common=efficacy_common,
            correlation_result=correlation_result,
            expected_features=output_spec["features"],
            expected_efficacy_columns=output_spec["efficacy_columns"],
        )
        results[group_name] = correlation_result

    export_scale_subset_workbook(efficacy_subsets, EFFICACY_SUBSET_OUTPUT_PATH)
    return results


def _validate_outputs(
    group_name: str,
    x_common: pd.DataFrame,
    efficacy_common: pd.DataFrame,
    correlation_result: CorrelationMatrixResult,
    expected_features: list[str],
    expected_efficacy_columns: list[str],
) -> None:
    expected_x = x_common.columns.tolist()[1:]
    expected_y = efficacy_common.columns.tolist()[1:]

    if expected_x != expected_features:
        raise ValueError(f"{group_name}: x-axis features do not match the configured order.")
    if expected_y != expected_efficacy_columns:
        raise ValueError(f"{group_name}: efficacy y-axis columns do not match the configured order.")
    if any("immed" in col for col in expected_y):
        raise ValueError(f"{group_name}: efficacy y-axis columns contain immediate variables.")

    is_delta_group = group_name in {"snr_local", "connectivity"}
    if is_delta_group and any(not col.startswith("Δ") for col in expected_y):
        raise ValueError(f"{group_name}: expected only Δ efficacy variables.")
    if not is_delta_group and any(col.startswith("Δ") for col in expected_y):
        raise ValueError(f"{group_name}: expected only non-Δ STN efficacy variables.")

    expected_shape = (len(expected_efficacy_columns), len(expected_features))
    if correlation_result.r.shape != expected_shape:
        raise ValueError(
            f"{group_name}: expected correlation matrix shape {expected_shape}; "
            f"got {correlation_result.r.shape}."
        )

    for name, matrix in {
        "p": correlation_result.p,
        "p_raw": correlation_result.p_raw,
        "n": correlation_result.n,
    }.items():
        if not correlation_result.r.index.equals(matrix.index) or not correlation_result.r.columns.equals(matrix.columns):
            raise ValueError(f"{group_name}: matrix '{name}' is not aligned with r.")

    if correlation_result.p_adj is None:
        raise ValueError(f"{group_name}: expected globally adjusted p-values, but p_adj is None.")
    if not correlation_result.r.index.equals(correlation_result.p_adj.index) or not correlation_result.r.columns.equals(correlation_result.p_adj.columns):
        raise ValueError(f"{group_name}: matrix 'p_adj' is not aligned with r.")


def _validate_efficacy_columns(efficacy_columns: list[str]) -> None:
    if any("immed" in column for column in efficacy_columns):
        raise ValueError(f"Efficacy columns must exclude immediate variables: {efficacy_columns}.")

    all_delta = all(column.startswith("Δ") for column in efficacy_columns)
    any_delta = any(column.startswith("Δ") for column in efficacy_columns)
    if any_delta and not all_delta:
        raise ValueError("Efficacy column list must not mix Δ and non-Δ variables.")


def _normalize_pair_series(series: pd.Series, pair_col: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"Column '{pair_col}' contains missing values.")
    return series.astype(str)


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


def _raise_on_duplicate_id_scale(df: pd.DataFrame, table_name: str) -> None:
    duplicate_mask = df.duplicated([PAIR_COL, SCALE_COL], keep=False)
    if not duplicate_mask.any():
        return

    duplicate_rows = (
        df.loc[duplicate_mask, [PAIR_COL, SCALE_COL]]
        .drop_duplicates()
        .sort_values([PAIR_COL, SCALE_COL])
    )
    examples = duplicate_rows.head(5).to_dict("records")
    raise ValueError(
        f"{table_name} contains duplicate '{PAIR_COL}' + '{SCALE_COL}' combinations. "
        f"Examples: {examples}"
    )


if __name__ == "__main__":
    workflow_results = run_workflow()
    print(f"Saved STN local wide table: {STN_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved SNr local wide table: {SNR_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved connectivity wide table: {CONNECTIVITY_WIDE_OUTPUT_PATH}")
    print(f"Saved efficacy subset workbook: {EFFICACY_SUBSET_OUTPUT_PATH}")
    print(f"Saved STN local correlation workbook: {STN_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved SNr local correlation workbook: {SNR_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved connectivity correlation workbook: {CONNECTIVITY_CORRELATION_OUTPUT_PATH}")
    for name, result in workflow_results.items():
        print(f"{name}: correlation r shape {result.r.shape}")
