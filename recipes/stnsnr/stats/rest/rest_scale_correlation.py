from __future__ import annotations

import os

from dataclasses import dataclass
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
REGION_COL = "Region"
BAND_COL = "Band"
VALUE_COL = "Value"

PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
REST_DIR = PROJECT / "summary" / "stats" / "rest"
SCALE_INPUT_PATH = PROJECT / "summary" / "cohort" / "subj" / "subj_scale.xlsx"
OUTPUT_DIR = REST_DIR / "corr"

STN_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "rest_stn_local_scale_corr.xlsx"
SNR_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "rest_snr_local_scale_corr.xlsx"
CONNECTIVITY_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "rest_connectivity_scale_corr.xlsx"

STN_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "rest_stn_local_subject_wide.xlsx"
SNR_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "rest_snr_local_subject_wide.xlsx"
CONNECTIVITY_WIDE_OUTPUT_PATH = OUTPUT_DIR / "rest_connectivity_subject_wide.xlsx"
SCALE_SUBSET_OUTPUT_PATH = OUTPUT_DIR / "subj_scale_subset_for_rest_corr.xlsx"

REST_CONTEXT = {
    "Pose": "Sit",
    "Med": "Off",
    "Task": "Single",
    "Phase": "All",
}

BAND_ORDER = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta_low",
    "Beta_high",
    "Gamma",
]

PARAM_ORDER = [
    "Exponent",
    "Offset",
]

LOCAL_REGIONS = [
    "STN",
    "SNr",
]

CONNECTIVITY_REGION = "SNr-STN"


@dataclass(frozen=True)
class ScalarFeatureSpec:
    key: str
    input_path: Path


LOCAL_BAND_SPECS = [
    ScalarFeatureSpec(
        key="raw",
        input_path=REST_DIR / "tfr" / "raw" / "raw-mean-scalar" / "raw-mean-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="period",
        input_path=REST_DIR / "tfr" / "period" / "period-mean-scalar" / "period-mean-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="burst_occupation",
        input_path=REST_DIR / "burst" / "burst" / "burst-occupation-scalar" / "burst-occupation-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="burst_rate",
        input_path=REST_DIR / "burst" / "burst" / "burst-rate-scalar" / "burst-rate-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="burst_duration",
        input_path=REST_DIR / "burst" / "burst" / "burst-duration-scalar" / "burst-duration-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="burst_mean",
        input_path=REST_DIR / "burst" / "burst" / "burst-mean-scalar" / "burst-mean-scalar_subject_raw.pkl",
    ),
]

PARAMS_INPUT_PATH = REST_DIR / "params" / "params" / "params-mean-scalar" / "params-mean-scalar_subject_raw.pkl"

CONNECTIVITY_SPECS = [
    ScalarFeatureSpec(
        key="coh",
        input_path=REST_DIR / "conn" / "coh" / "coh-mean-scalar" / "coh-mean-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="wpli",
        input_path=REST_DIR / "conn" / "wpli" / "wpli-mean-scalar" / "wpli-mean-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="ciplv",
        input_path=REST_DIR / "conn" / "ciplv" / "ciplv-mean-scalar" / "ciplv-mean-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="gc",
        input_path=REST_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar" / "delta_net_gc-mean-scalar_subject_raw.pkl",
    ),
    ScalarFeatureSpec(
        key="psi",
        input_path=REST_DIR / "psi" / "psi" / "psi-mean-scalar" / "psi-mean-scalar_subject_raw.pkl",
    ),
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


def _local_band_feature_key(region: str, band: str, metric_key: str) -> str:
    return f"{region}__{band}__{metric_key}"


def _local_param_feature_key(region: str, param_name: str) -> str:
    return f"{region}__params__{param_name}"


def _connectivity_feature_key(band: str, metric_key: str) -> str:
    return f"{CONNECTIVITY_REGION}__{band}__{metric_key}"


def make_local_feature_order(region: str) -> list[str]:
    """Return the band-first local feature order for a single region."""
    return [
        _local_band_feature_key(region, band, spec.key)
        for band in BAND_ORDER
        for spec in LOCAL_BAND_SPECS
    ] + [_local_param_feature_key(region, param) for param in PARAM_ORDER]


def make_connectivity_feature_order() -> list[str]:
    """Return the band-first connectivity feature order."""
    return [
        _connectivity_feature_key(band, spec.key)
        for band in BAND_ORDER
        for spec in CONNECTIVITY_SPECS
    ]


LOCAL_FEATURE_ORDER_BY_REGION = {
    region: make_local_feature_order(region)
    for region in LOCAL_REGIONS
}
CONNECTIVITY_FEATURE_ORDER = make_connectivity_feature_order()


def load_local_wide(region: str) -> pd.DataFrame:
    """Load local scalar features for one region and pivot to subject-wide format."""
    if region not in LOCAL_REGIONS:
        raise ValueError(f"Unsupported local region: {region!r}. Expected one of {LOCAL_REGIONS}.")

    feature_rows = []
    for spec in LOCAL_BAND_SPECS:
        scalar = load_scalar_table(spec.input_path)
        filtered = _filter_region_and_bands(
            scalar,
            region=region,
            required_bands=BAND_ORDER,
            table_name=str(spec.input_path),
        )
        filtered["feature"] = [
            _local_band_feature_key(region, band, spec.key)
            for band in filtered[BAND_COL].astype(str)
        ]
        feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

    params = load_scalar_table(PARAMS_INPUT_PATH)
    params_filtered = _filter_region_and_bands(
        params,
        region=region,
        required_bands=PARAM_ORDER,
        table_name=str(PARAMS_INPUT_PATH),
    )
    params_filtered["feature"] = [
        _local_param_feature_key(region, band)
        for band in params_filtered[BAND_COL].astype(str)
    ]
    feature_rows.append(params_filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

    return _pivot_feature_rows(
        feature_rows,
        ordered_features=LOCAL_FEATURE_ORDER_BY_REGION[region],
        table_name=f"{region} local scalar table",
    )


def load_connectivity_wide() -> pd.DataFrame:
    """Load connectivity scalar features and pivot to subject-wide format."""
    feature_rows = []
    for spec in CONNECTIVITY_SPECS:
        scalar = load_scalar_table(spec.input_path)
        filtered = _filter_region_and_bands(
            scalar,
            region=CONNECTIVITY_REGION,
            required_bands=BAND_ORDER,
            table_name=str(spec.input_path),
        )
        filtered["feature"] = [
            _connectivity_feature_key(band, spec.key)
            for band in filtered[BAND_COL].astype(str)
        ]
        feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

    return _pivot_feature_rows(
        feature_rows,
        ordered_features=CONNECTIVITY_FEATURE_ORDER,
        table_name="connectivity scalar table",
    )


def load_scalar_table(input_path: str | Path) -> pd.DataFrame:
    """Load and validate one long-form resting-state scalar table."""
    input_path = Path(input_path)
    scalar = pd.read_pickle(input_path)
    if not isinstance(scalar, pd.DataFrame):
        raise TypeError(f"{input_path} did not contain a pandas DataFrame.")

    _validate_required_columns(
        scalar,
        required_columns=[PAIR_COL, REGION_COL, BAND_COL, VALUE_COL, *REST_CONTEXT.keys()],
        table_name=str(input_path),
    )
    _validate_rest_context(scalar, table_name=str(input_path))

    prepared = scalar.loc[:, [PAIR_COL, REGION_COL, BAND_COL, VALUE_COL, *REST_CONTEXT.keys()]].copy()
    prepared = _normalize_pair_column(prepared, pair_col=PAIR_COL)
    prepared[REGION_COL] = prepared[REGION_COL].astype(str)
    prepared[BAND_COL] = prepared[BAND_COL].astype(str)
    prepared[VALUE_COL] = pd.to_numeric(prepared[VALUE_COL], errors="coerce")
    return prepared


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
    x_wide: pd.DataFrame,
    scale_subset: pd.DataFrame,
    pair_col: str = PAIR_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter x and scale tables to their shared IDs in the same order."""
    x_wide = _normalize_pair_column(x_wide.copy(), pair_col=pair_col)
    scale_subset = _normalize_pair_column(scale_subset.copy(), pair_col=pair_col)

    if x_wide[pair_col].duplicated().any():
        raise ValueError("Feature wide table contains duplicate IDs.")
    if scale_subset[pair_col].duplicated().any():
        raise ValueError("Scale subset table contains duplicate IDs.")

    common_ids = sorted(set(x_wide[pair_col]).intersection(scale_subset[pair_col]))
    if not common_ids:
        raise ValueError("No shared IDs were found between rest features and scale table.")

    x_common = (
        x_wide[x_wide[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )
    scale_common = (
        scale_subset[scale_subset[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )

    if x_common[pair_col].tolist() != scale_common[pair_col].tolist():
        raise ValueError("Filtered feature and scale tables do not share the same ordered IDs.")

    return x_common, scale_common


def compute_rest_scale_correlation(
    x_common: pd.DataFrame,
    scale_common: pd.DataFrame,
) -> CorrelationMatrixResult:
    """Compute globally FDR-adjusted Spearman correlations."""
    return compute_correlation_matrix(
        df_x=x_common,
        df_y=scale_common,
        pair_col=PAIR_COL,
        method="spearman",
        correction_mode="All",
        correction_method="bh",
    )


def export_correlation_workbook(
    correlation_result: CorrelationMatrixResult,
    output_path: str | Path,
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


def export_scale_subset_workbook(
    scale_subsets: dict[str, pd.DataFrame],
    output_path: str | Path = SCALE_SUBSET_OUTPUT_PATH,
) -> None:
    """Export per-analysis scale subsets to one workbook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, scale_subset in scale_subsets.items():
            scale_subset.to_excel(writer, sheet_name=sheet_name, index=False)


def run_workflow() -> dict[str, CorrelationMatrixResult]:
    """Run the full rest scalar-scale correlation workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scale_subset = load_scale_subset(SCALE_INPUT_PATH)
    wide_tables = {
        "stn_local": load_local_wide("STN"),
        "snr_local": load_local_wide("SNr"),
        "connectivity": load_connectivity_wide(),
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

    results: dict[str, CorrelationMatrixResult] = {}
    scale_subsets: dict[str, pd.DataFrame] = {}

    for group_name, x_wide in wide_tables.items():
        output_spec = output_specs[group_name]
        x_common, scale_common = filter_to_common_ids(x_wide, scale_subset)

        x_common.to_excel(output_spec["wide_path"], index=False)
        scale_subsets[group_name] = scale_common

        correlation_result = compute_rest_scale_correlation(x_common, scale_common)
        export_correlation_workbook(correlation_result, output_spec["corr_path"])
        _validate_outputs(
            group_name=group_name,
            x_common=x_common,
            scale_common=scale_common,
            correlation_result=correlation_result,
            expected_features=output_spec["features"],
        )
        results[group_name] = correlation_result

    export_scale_subset_workbook(scale_subsets, SCALE_SUBSET_OUTPUT_PATH)
    return results


def _filter_region_and_bands(
    scalar: pd.DataFrame,
    region: str,
    required_bands: list[str],
    table_name: str,
) -> pd.DataFrame:
    filtered = scalar.loc[scalar[REGION_COL] == region].copy()
    if filtered.empty:
        raise ValueError(f"{table_name} has no rows for Region={region!r}.")

    available_bands = set(filtered[BAND_COL].astype(str))
    required = set(required_bands)
    missing_bands = [band for band in required_bands if band not in available_bands]
    unexpected_bands = sorted(available_bands - required)
    if missing_bands:
        raise ValueError(f"{table_name} is missing required bands for {region}: {missing_bands}.")
    if unexpected_bands:
        raise ValueError(f"{table_name} has unexpected bands for {region}: {unexpected_bands}.")

    return filtered.loc[filtered[BAND_COL].isin(required_bands)].copy()


def _pivot_feature_rows(
    feature_rows: list[pd.DataFrame],
    ordered_features: list[str],
    table_name: str,
) -> pd.DataFrame:
    if not feature_rows:
        raise ValueError(f"{table_name} has no feature rows to pivot.")

    long = pd.concat(feature_rows, ignore_index=True)
    _raise_on_duplicate_pair_feature(long, feature_col="feature", table_name=table_name)

    wide = long.pivot(index=PAIR_COL, columns="feature", values=VALUE_COL)
    missing_features = [feature for feature in ordered_features if feature not in wide.columns]
    if missing_features:
        raise ValueError(f"{table_name} is missing expected features: {missing_features}.")

    wide = wide.loc[:, ordered_features].sort_index(axis=0).reset_index()
    wide.columns.name = None
    return wide


def _validate_outputs(
    group_name: str,
    x_common: pd.DataFrame,
    scale_common: pd.DataFrame,
    correlation_result: CorrelationMatrixResult,
    expected_features: list[str],
) -> None:
    expected_x = x_common.columns.tolist()[1:]
    expected_y = scale_common.columns.tolist()[1:]

    if expected_x != expected_features:
        raise ValueError(f"{group_name}: x-axis features do not match the configured order.")
    if expected_y != Y_COLUMNS:
        raise ValueError(f"{group_name}: scale y-axis columns do not match the configured order.")
    if any("Med-ON" in col for col in expected_y):
        raise ValueError(f"{group_name}: scale y-axis columns contain Med-ON variables.")
    if "QUIP-RS" in expected_y:
        raise ValueError(f"{group_name}: scale y-axis columns contain QUIP-RS.")

    expected_shape = (len(Y_COLUMNS), len(expected_features))
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


def _validate_rest_context(df: pd.DataFrame, table_name: str) -> None:
    for column, expected in REST_CONTEXT.items():
        observed = sorted(set(df[column].dropna().astype(str)))
        if observed != [expected]:
            raise ValueError(
                f"{table_name} has unexpected {column} values: {observed}. "
                f"Expected only {expected!r}."
            )


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
    workflow_results = run_workflow()
    print(f"Saved STN local wide table: {STN_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved SNr local wide table: {SNR_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved connectivity wide table: {CONNECTIVITY_WIDE_OUTPUT_PATH}")
    print(f"Saved scale subset workbook: {SCALE_SUBSET_OUTPUT_PATH}")
    print(f"Saved STN local correlation workbook: {STN_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved SNr local correlation workbook: {SNR_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved connectivity correlation workbook: {CONNECTIVITY_CORRELATION_OUTPUT_PATH}")
    for name, result in workflow_results.items():
        print(f"{name}: correlation r shape {result.r.shape}")
