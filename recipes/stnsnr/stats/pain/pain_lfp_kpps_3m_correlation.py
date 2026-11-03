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

from src.stats.correlation.corr_matrix import CorrelationMatrixResult, compute_correlation_matrix


PAIR_COL = "ID"
REGION_COL = "Region"
BAND_COL = "Band"
PHASE_COL = "Phase"
VALUE_COL = "Value"

PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
PAIN_DIR = PROJECT / "summary" / "stats" / "pain"
OUTPUT_DIR = PAIN_DIR / "corr" / "pain_lfp_kpps_3m"
PREOP_SCALE_INPUT_PATH = PROJECT / "summary" / "cohort" / "subj" / "subj_scale.xlsx"
DERIVED_KPPS_INPUT_PATH = PROJECT / "summary" / "stats" / "clinic" / "correlation" / "pain" / "scale_subject.xlsx"

KPPS_WIDE_OUTPUT_PATH = OUTPUT_DIR / "pain_kpps_3m_subject_wide.xlsx"
KPPS_SUBSET_OUTPUT_PATH = OUTPUT_DIR / "pain_kpps_3m_subset_for_corr.xlsx"
STN_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "pain_stn_local_subject_normalized_wide_by_phase.xlsx"
SNR_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "pain_snr_local_subject_normalized_wide_by_phase.xlsx"
CONNECTIVITY_WIDE_OUTPUT_PATH = OUTPUT_DIR / "pain_connectivity_subject_normalized_wide_by_phase.xlsx"
STN_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "pain_stn_local_kpps_3m_corr_by_phase.xlsx"
SNR_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "pain_snr_local_kpps_3m_corr_by_phase.xlsx"
CONNECTIVITY_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "pain_connectivity_kpps_3m_corr_by_phase.xlsx"

PHASE_ORDER = [
    "ONS",
    "SUS",
    "OFS",
    "PST",
]

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

KPPS_COLUMNS = [
    "Pre-op KPPS",
    "KPPS Δ% (STN, 3m)",
    "ΔKPPS Δ% (+SNr, 3m)",
]

DERIVED_KPPS_COLUMNS = [
    "KPPS Δ% (STN, 3m)",
    "ΔKPPS Δ% (+SNr, 3m)",
]


@dataclass(frozen=True)
class ScalarFeatureSpec:
    key: str
    input_path: Path


LOCAL_BAND_SPECS = [
    ScalarFeatureSpec(
        key="raw",
        input_path=PAIN_DIR / "tfr" / "raw" / "raw-mean-scalar" / "raw-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="period",
        input_path=PAIN_DIR / "tfr" / "period" / "period-mean-scalar" / "period-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_occupation",
        input_path=PAIN_DIR
        / "burst"
        / "burst"
        / "burst-occupation-scalar"
        / "burst-occupation-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_rate",
        input_path=PAIN_DIR / "burst" / "burst" / "burst-rate-scalar" / "burst-rate-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_duration",
        input_path=PAIN_DIR
        / "burst"
        / "burst"
        / "burst-duration-scalar"
        / "burst-duration-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_mean",
        input_path=PAIN_DIR / "burst" / "burst" / "burst-mean-scalar" / "burst-mean-scalar_subject_normalized.csv",
    ),
]

PARAMS_INPUT_PATH = (
    PAIN_DIR
    / "params"
    / "params"
    / "params-mean-scalar"
    / "params-mean-scalar_subject_normalized.csv"
)

CONNECTIVITY_SPECS = [
    ScalarFeatureSpec(
        key="coh",
        input_path=PAIN_DIR / "conn" / "coh" / "coh-mean-scalar" / "coh-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="wpli",
        input_path=PAIN_DIR / "conn" / "wpli" / "wpli-mean-scalar" / "wpli-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="ciplv",
        input_path=PAIN_DIR / "conn" / "ciplv" / "ciplv-mean-scalar" / "ciplv-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="delta_net_gc",
        input_path=PAIN_DIR
        / "gc"
        / "delta_net_gc"
        / "delta_net_gc-mean-scalar"
        / "delta_net_gc-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="psi",
        input_path=PAIN_DIR / "psi" / "psi" / "psi-mean-scalar" / "psi-mean-scalar_subject_normalized.csv",
    ),
]


def _local_band_feature_key(region: str, band: str, metric_key: str) -> str:
    return f"{region}__{band}__{metric_key}"


def _local_param_feature_key(region: str, param_name: str) -> str:
    return f"{region}__params__{param_name}"


def _connectivity_feature_key(band: str, metric_key: str) -> str:
    return f"{CONNECTIVITY_REGION}__{band}__{metric_key}"


def make_local_feature_order(region: str) -> list[str]:
    """Return the band-first local pain LFP feature order."""
    return [
        _local_band_feature_key(region, band, spec.key)
        for band in BAND_ORDER
        for spec in LOCAL_BAND_SPECS
    ] + [_local_param_feature_key(region, param) for param in PARAM_ORDER]


def make_connectivity_feature_order() -> list[str]:
    """Return the band-first connectivity pain LFP feature order."""
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


def load_kpps_3m_table(
    preop_input_path: str | Path = PREOP_SCALE_INPUT_PATH,
    derived_input_path: str | Path = DERIVED_KPPS_INPUT_PATH,
) -> pd.DataFrame:
    """Load preoperative KPPS and existing derived 3-month KPPS improvement variables."""
    preop_input_path = Path(preop_input_path)
    derived_input_path = Path(derived_input_path)

    preop = pd.read_excel(preop_input_path)
    _validate_required_columns(
        preop,
        required_columns=[PAIR_COL, "KPPS"],
        table_name=str(preop_input_path),
    )
    preop = _normalize_pair_column(preop.loc[:, [PAIR_COL, "KPPS"]].copy(), pair_col=PAIR_COL)
    preop = preop.rename(columns={"KPPS": "Pre-op KPPS"})
    preop["Pre-op KPPS"] = pd.to_numeric(preop["Pre-op KPPS"], errors="coerce")
    _raise_on_duplicate_pairs(preop, table_name=str(preop_input_path))

    derived = pd.read_excel(derived_input_path)
    _validate_required_columns(
        derived,
        required_columns=[PAIR_COL, "Scale", VALUE_COL],
        table_name=str(derived_input_path),
    )
    derived = _normalize_pair_column(derived.loc[:, [PAIR_COL, "Scale", VALUE_COL]].copy(), pair_col=PAIR_COL)
    derived["Scale"] = derived["Scale"].astype(str)
    derived[VALUE_COL] = pd.to_numeric(derived[VALUE_COL], errors="coerce")
    derived = derived.loc[derived["Scale"].isin(DERIVED_KPPS_COLUMNS)].copy()
    if derived.empty:
        raise ValueError(f"{derived_input_path} has no rows for {DERIVED_KPPS_COLUMNS}.")

    _raise_on_duplicate_pair_feature(derived, feature_col="Scale", table_name=str(derived_input_path))
    wide_derived = derived.pivot(index=PAIR_COL, columns="Scale", values=VALUE_COL)
    missing_scales = [scale for scale in DERIVED_KPPS_COLUMNS if scale not in wide_derived.columns]
    if missing_scales:
        raise ValueError(f"{derived_input_path} is missing derived KPPS variables: {missing_scales}.")

    wide_derived = wide_derived.loc[:, DERIVED_KPPS_COLUMNS].sort_index(axis=0).reset_index()
    wide_derived.columns.name = None

    out = preop.merge(wide_derived, on=PAIR_COL, how="outer")
    out = out.loc[:, [PAIR_COL, *KPPS_COLUMNS]].sort_values(PAIR_COL).reset_index(drop=True)
    return out


def load_local_wide_by_phase(region: str) -> dict[str, pd.DataFrame]:
    """Load local normalized pain LFP effects for one region by phase."""
    if region not in LOCAL_REGIONS:
        raise ValueError(f"Unsupported local region: {region!r}. Expected one of {LOCAL_REGIONS}.")

    tables_by_phase: dict[str, pd.DataFrame] = {}
    for phase in PHASE_ORDER:
        feature_rows = []
        for spec in LOCAL_BAND_SPECS:
            scalar = load_normalized_scalar_table(spec.input_path)
            filtered = _filter_pain_lfp_rows(
                scalar,
                region=region,
                phase=phase,
                required_bands=BAND_ORDER,
                table_name=str(spec.input_path),
            )
            filtered["feature"] = [
                _local_band_feature_key(region, band, spec.key)
                for band in filtered[BAND_COL].astype(str)
            ]
            feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        params = load_normalized_scalar_table(PARAMS_INPUT_PATH)
        params_filtered = _filter_pain_lfp_rows(
            params,
            region=region,
            phase=phase,
            required_bands=PARAM_ORDER,
            table_name=str(PARAMS_INPUT_PATH),
        )
        params_filtered["feature"] = [
            _local_param_feature_key(region, band)
            for band in params_filtered[BAND_COL].astype(str)
        ]
        feature_rows.append(params_filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        tables_by_phase[phase] = _pivot_feature_rows(
            feature_rows,
            ordered_features=LOCAL_FEATURE_ORDER_BY_REGION[region],
            table_name=f"{region} local normalized pain LFP table for {phase}",
        )

    return tables_by_phase


def load_connectivity_wide_by_phase() -> dict[str, pd.DataFrame]:
    """Load normalized pain connectivity effects by phase."""
    tables_by_phase: dict[str, pd.DataFrame] = {}
    for phase in PHASE_ORDER:
        feature_rows = []
        for spec in CONNECTIVITY_SPECS:
            scalar = load_normalized_scalar_table(spec.input_path)
            filtered = _filter_pain_lfp_rows(
                scalar,
                region=CONNECTIVITY_REGION,
                phase=phase,
                required_bands=BAND_ORDER,
                table_name=str(spec.input_path),
            )
            filtered["feature"] = [
                _connectivity_feature_key(band, spec.key)
                for band in filtered[BAND_COL].astype(str)
            ]
            feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        tables_by_phase[phase] = _pivot_feature_rows(
            feature_rows,
            ordered_features=CONNECTIVITY_FEATURE_ORDER,
            table_name=f"connectivity normalized pain LFP table for {phase}",
        )

    return tables_by_phase


def load_normalized_scalar_table(input_path: str | Path) -> pd.DataFrame:
    """Load and validate one normalized pain scalar table."""
    input_path = Path(input_path)
    scalar = pd.read_csv(input_path)

    required_columns = [
        PAIR_COL,
        REGION_COL,
        PHASE_COL,
        BAND_COL,
        VALUE_COL,
    ]
    _validate_required_columns(scalar, required_columns=required_columns, table_name=str(input_path))

    prepared = scalar.loc[:, required_columns].copy()
    prepared = _normalize_pair_column(prepared, pair_col=PAIR_COL)
    for column in [REGION_COL, PHASE_COL, BAND_COL]:
        prepared[column] = prepared[column].astype(str)
    prepared[VALUE_COL] = pd.to_numeric(prepared[VALUE_COL], errors="coerce")
    return prepared


def filter_to_common_ids(
    x_wide: pd.DataFrame,
    kpps: pd.DataFrame,
    pair_col: str = PAIR_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter LFP and KPPS tables to shared IDs in the same order."""
    x_wide = _normalize_pair_column(x_wide.copy(), pair_col=pair_col)
    kpps = _normalize_pair_column(kpps.copy(), pair_col=pair_col)

    if x_wide[pair_col].duplicated().any():
        raise ValueError("LFP wide table contains duplicate IDs.")
    if kpps[pair_col].duplicated().any():
        raise ValueError("KPPS table contains duplicate IDs.")

    common_ids = sorted(set(x_wide[pair_col]).intersection(kpps[pair_col]))
    if not common_ids:
        raise ValueError("No shared IDs were found between pain LFP features and KPPS table.")

    x_common = (
        x_wide[x_wide[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )
    kpps_common = (
        kpps[kpps[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )

    if x_common[pair_col].tolist() != kpps_common[pair_col].tolist():
        raise ValueError("Filtered LFP and KPPS tables do not share the same ordered IDs.")

    return x_common, kpps_common


def compute_pain_kpps_correlation(
    x_common: pd.DataFrame,
    kpps_common: pd.DataFrame,
) -> CorrelationMatrixResult:
    """Compute globally FDR-adjusted Spearman correlations."""
    return compute_correlation_matrix(
        df_x=x_common,
        df_y=kpps_common,
        pair_col=PAIR_COL,
        method="spearman",
        correction_mode="All",
        correction_method="bh",
    )


def export_phase_workbook(
    phase_tables: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    """Export one workbook with one sheet per phase."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for phase in PHASE_ORDER:
            phase_tables[phase].to_excel(writer, sheet_name=phase, index=False)


def export_scale_subset_workbook(
    scale_subsets: dict[str, pd.DataFrame],
    output_path: str | Path = KPPS_SUBSET_OUTPUT_PATH,
) -> None:
    """Export KPPS subsets used for each analysis group and phase."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, scale_subset in scale_subsets.items():
            scale_subset.to_excel(writer, sheet_name=sheet_name, index=False)


def export_correlation_workbook_by_phase(
    correlation_results: dict[str, CorrelationMatrixResult],
    output_path: str | Path,
) -> None:
    """Export phase-specific correlation matrices to a multi-sheet workbook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for phase in PHASE_ORDER:
            result = correlation_results[phase]
            result.r.to_excel(writer, sheet_name=f"{phase}_r")
            result.p.to_excel(writer, sheet_name=f"{phase}_p")
            result.p_raw.to_excel(writer, sheet_name=f"{phase}_p_raw")
            if result.p_adj is None:
                raise ValueError(f"{phase}: expected adjusted p-values, but p_adj is None.")
            result.p_adj.to_excel(writer, sheet_name=f"{phase}_p_adj")
            result.n.to_excel(writer, sheet_name=f"{phase}_n")


def run_workflow() -> dict[str, dict[str, CorrelationMatrixResult]]:
    """Run the full pain LFP-KPPS 3-month correlation workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kpps = load_kpps_3m_table(PREOP_SCALE_INPUT_PATH, DERIVED_KPPS_INPUT_PATH)
    kpps.to_excel(KPPS_WIDE_OUTPUT_PATH, index=False)

    wide_tables = {
        "stn_local": load_local_wide_by_phase("STN"),
        "snr_local": load_local_wide_by_phase("SNr"),
        "connectivity": load_connectivity_wide_by_phase(),
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
    scale_subsets: dict[str, pd.DataFrame] = {}

    for group_name, phase_tables in wide_tables.items():
        output_spec = output_specs[group_name]
        phase_common_tables: dict[str, pd.DataFrame] = {}
        phase_results: dict[str, CorrelationMatrixResult] = {}

        for phase in PHASE_ORDER:
            x_common, kpps_common = filter_to_common_ids(phase_tables[phase], kpps)
            phase_common_tables[phase] = x_common
            scale_subsets[f"{group_name}_{phase}"] = kpps_common

            correlation_result = compute_pain_kpps_correlation(x_common, kpps_common)
            _validate_outputs(
                group_name=group_name,
                phase=phase,
                x_common=x_common,
                kpps_common=kpps_common,
                correlation_result=correlation_result,
                expected_features=output_spec["features"],
            )
            phase_results[phase] = correlation_result

        export_phase_workbook(phase_common_tables, output_spec["wide_path"])
        export_correlation_workbook_by_phase(phase_results, output_spec["corr_path"])
        results[group_name] = phase_results

    export_scale_subset_workbook(scale_subsets, KPPS_SUBSET_OUTPUT_PATH)
    return results


def _filter_pain_lfp_rows(
    scalar: pd.DataFrame,
    *,
    region: str,
    phase: str,
    required_bands: list[str],
    table_name: str,
) -> pd.DataFrame:
    filtered = scalar.loc[
        (scalar[REGION_COL] == region)
        & (scalar[PHASE_COL] == phase)
    ].copy()

    if filtered.empty:
        raise ValueError(f"{table_name} has no rows for Region={region!r} and Phase={phase!r}.")

    available_bands = set(filtered[BAND_COL].astype(str))
    required = set(required_bands)
    missing_bands = [band for band in required_bands if band not in available_bands]
    unexpected_bands = sorted(available_bands - required)
    if missing_bands:
        raise ValueError(
            f"{table_name} is missing required bands for {region} {phase}: {missing_bands}."
        )
    if unexpected_bands:
        raise ValueError(
            f"{table_name} has unexpected bands for {region} {phase}: {unexpected_bands}."
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
    _raise_on_duplicate_pair_feature(long, feature_col="feature", table_name=table_name)

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
    phase: str,
    x_common: pd.DataFrame,
    kpps_common: pd.DataFrame,
    correlation_result: CorrelationMatrixResult,
    expected_features: list[str],
) -> None:
    expected_x = x_common.columns.tolist()[1:]
    expected_y = kpps_common.columns.tolist()[1:]

    if expected_x != expected_features:
        raise ValueError(f"{group_name} {phase}: x-axis features do not match the configured order.")
    if expected_y != KPPS_COLUMNS:
        raise ValueError(f"{group_name} {phase}: KPPS y-axis columns do not match the configured order.")

    expected_shape = (len(KPPS_COLUMNS), len(expected_features))
    if correlation_result.r.shape != expected_shape:
        raise ValueError(
            f"{group_name} {phase}: expected correlation matrix shape {expected_shape}; "
            f"got {correlation_result.r.shape}."
        )

    for name, matrix in {
        "p": correlation_result.p,
        "p_raw": correlation_result.p_raw,
        "n": correlation_result.n,
    }.items():
        if not correlation_result.r.index.equals(matrix.index) or not correlation_result.r.columns.equals(matrix.columns):
            raise ValueError(f"{group_name} {phase}: matrix {name!r} is not aligned with r.")

    if correlation_result.p_adj is None:
        raise ValueError(f"{group_name} {phase}: expected globally adjusted p-values, but p_adj is None.")
    if not correlation_result.r.index.equals(correlation_result.p_adj.index) or not correlation_result.r.columns.equals(correlation_result.p_adj.columns):
        raise ValueError(f"{group_name} {phase}: matrix 'p_adj' is not aligned with r.")


def _normalize_pair_column(df: pd.DataFrame, pair_col: str) -> pd.DataFrame:
    _validate_required_columns(df, required_columns=[pair_col], table_name="input table")
    if df[pair_col].isna().any():
        raise ValueError(f"Column {pair_col!r} contains missing values.")

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


def _raise_on_duplicate_pairs(
    df: pd.DataFrame,
    table_name: str,
    pair_col: str = PAIR_COL,
) -> None:
    duplicate_mask = df.duplicated(pair_col, keep=False)
    if not duplicate_mask.any():
        return

    duplicate_ids = sorted(df.loc[duplicate_mask, pair_col].astype(str).unique().tolist())
    raise ValueError(f"{table_name} has duplicate IDs: {duplicate_ids[:5]}.")


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
    examples = duplicate_rows.head(10).to_dict("records")
    raise ValueError(f"{table_name} contains duplicate ID-feature rows: {examples}.")


if __name__ == "__main__":
    run_workflow()
