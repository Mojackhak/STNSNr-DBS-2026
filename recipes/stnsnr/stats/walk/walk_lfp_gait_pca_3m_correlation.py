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
POSE_COL = "Pose"
MED_COL = "Med"
TASK_COL = "Task"
PHASE_COL = "Phase"
VALUE_COL = "Value"

PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
WALK_DIR = PROJECT / "summary" / "stats" / "walk"
OUTPUT_DIR = WALK_DIR / "corr" / "walk_lfp_gait_pca_3m"
GAIT_PCA_INPUT_PATH = PROJECT / "summary" / "stats" / "clinic" / "pca" / "gait" / "dist.xlsx"

GAIT_PCA_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_gait_pca_3m_distance_subject_wide.xlsx"
GAIT_PCA_SUBSET_OUTPUT_PATH = OUTPUT_DIR / "walk_gait_pca_3m_distance_subset_for_corr.xlsx"
STN_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_single_stn_local_subject_normalized_wide_by_pose.xlsx"
SNR_LOCAL_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_single_snr_local_subject_normalized_wide_by_pose.xlsx"
CONNECTIVITY_WIDE_OUTPUT_PATH = OUTPUT_DIR / "walk_single_connectivity_subject_normalized_wide_by_pose.xlsx"
STN_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "walk_single_stn_local_gait_pca_3m_corr_by_pose.xlsx"
SNR_LOCAL_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "walk_single_snr_local_gait_pca_3m_corr_by_pose.xlsx"
CONNECTIVITY_CORRELATION_OUTPUT_PATH = OUTPUT_DIR / "walk_single_connectivity_gait_pca_3m_corr_by_pose.xlsx"

POSE_ORDER = ["Stand", "Walk", "Turn"]
WALK_CONTEXT = {
    MED_COL: "Off",
    TASK_COL: "Single",
    PHASE_COL: "All",
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

GAIT_PCA_COLUMNS = [
    "ΔGait PCA STN vs Pre-op",
    "ΔGait PCA STN+SNr vs Pre-op",
    "ΔGait PCA STN+SNr vs STN",
]

GAIT_PCA_PROTOCOLS = ["STN", "STN+SNr"]
GAIT_PCA_PHASE = "3 m"


@dataclass(frozen=True)
class ScalarFeatureSpec:
    key: str
    input_path: Path


LOCAL_BAND_SPECS = [
    ScalarFeatureSpec(
        key="raw",
        input_path=WALK_DIR / "tfr" / "raw" / "raw-mean-scalar" / "raw-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="period",
        input_path=WALK_DIR / "tfr" / "period" / "period-mean-scalar" / "period-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_occupation",
        input_path=WALK_DIR
        / "burst"
        / "burst"
        / "burst-occupation-scalar"
        / "burst-occupation-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_rate",
        input_path=WALK_DIR / "burst" / "burst" / "burst-rate-scalar" / "burst-rate-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_duration",
        input_path=WALK_DIR
        / "burst"
        / "burst"
        / "burst-duration-scalar"
        / "burst-duration-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="burst_mean",
        input_path=WALK_DIR / "burst" / "burst" / "burst-mean-scalar" / "burst-mean-scalar_subject_normalized.csv",
    ),
]

PARAMS_INPUT_PATH = (
    WALK_DIR
    / "params"
    / "params"
    / "params-mean-scalar"
    / "params-mean-scalar_subject_normalized.csv"
)

CONNECTIVITY_SPECS = [
    ScalarFeatureSpec(
        key="coh",
        input_path=WALK_DIR / "conn" / "coh" / "coh-mean-scalar" / "coh-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="wpli",
        input_path=WALK_DIR / "conn" / "wpli" / "wpli-mean-scalar" / "wpli-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="ciplv",
        input_path=WALK_DIR / "conn" / "ciplv" / "ciplv-mean-scalar" / "ciplv-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="delta_net_gc",
        input_path=WALK_DIR
        / "gc"
        / "delta_net_gc"
        / "delta_net_gc-mean-scalar"
        / "delta_net_gc-mean-scalar_subject_normalized.csv",
    ),
    ScalarFeatureSpec(
        key="psi",
        input_path=WALK_DIR / "psi" / "psi" / "psi-mean-scalar" / "psi-mean-scalar_subject_normalized.csv",
    ),
]


def _local_band_feature_key(region: str, band: str, metric_key: str) -> str:
    return f"{region}__{band}__{metric_key}"


def _local_param_feature_key(region: str, param_name: str) -> str:
    return f"{region}__params__{param_name}"


def _connectivity_feature_key(band: str, metric_key: str) -> str:
    return f"{CONNECTIVITY_REGION}__{band}__{metric_key}"


def make_local_feature_order(region: str) -> list[str]:
    """Return the band-first local walk LFP feature order."""
    return [
        _local_band_feature_key(region, band, spec.key)
        for band in BAND_ORDER
        for spec in LOCAL_BAND_SPECS
    ] + [_local_param_feature_key(region, param) for param in PARAM_ORDER]


def make_connectivity_feature_order() -> list[str]:
    """Return the band-first connectivity walk LFP feature order."""
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


def load_gait_pca_3m_distance(input_path: str | Path = GAIT_PCA_INPUT_PATH) -> pd.DataFrame:
    """Load 3-month gait PCA distance effects and derive the SNr increment."""
    input_path = Path(input_path)
    gait = pd.read_excel(input_path)
    _validate_required_columns(
        gait,
        required_columns=[PAIR_COL, "Protocol", "Phase", VALUE_COL],
        table_name=str(input_path),
    )

    gait = gait.loc[gait["Phase"].astype(str) == GAIT_PCA_PHASE].copy()
    if gait.empty:
        raise ValueError(f"{input_path} has no rows with Phase={GAIT_PCA_PHASE!r}.")

    gait = _normalize_pair_column(gait, pair_col=PAIR_COL)
    gait["Protocol"] = gait["Protocol"].astype(str)
    gait[VALUE_COL] = pd.to_numeric(gait[VALUE_COL], errors="coerce")

    unexpected_protocols = sorted(set(gait["Protocol"]) - set(GAIT_PCA_PROTOCOLS))
    if unexpected_protocols:
        raise ValueError(
            f"{input_path} has unexpected 3-month protocols: {unexpected_protocols}."
        )

    duplicate_mask = gait.duplicated([PAIR_COL, "Protocol"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = (
            gait.loc[duplicate_mask, [PAIR_COL, "Protocol"]]
            .drop_duplicates()
            .sort_values([PAIR_COL, "Protocol"])
            .to_dict("records")
        )
        raise ValueError(f"{input_path} has duplicate 3-month gait rows: {duplicate_rows[:5]}.")

    protocol_sets = (
        gait.groupby(PAIR_COL, observed=True)["Protocol"]
        .agg(lambda values: set(values.tolist()))
        .reset_index(name="protocols")
    )
    expected_protocols = set(GAIT_PCA_PROTOCOLS)
    incomplete = protocol_sets[protocol_sets["protocols"] != expected_protocols]
    if not incomplete.empty:
        examples = incomplete.head(5).to_dict("records")
        raise ValueError(
            "Each subject must have both 3-month gait PCA protocols. "
            f"Incomplete examples: {examples}."
        )

    wide = gait.pivot(index=PAIR_COL, columns="Protocol", values=VALUE_COL)
    wide = wide.loc[:, GAIT_PCA_PROTOCOLS].sort_index(axis=0).reset_index()
    wide.columns.name = None

    out = pd.DataFrame({PAIR_COL: wide[PAIR_COL].astype(str)})
    out["ΔGait PCA STN vs Pre-op"] = wide["STN"]
    out["ΔGait PCA STN+SNr vs Pre-op"] = wide["STN+SNr"]
    out["ΔGait PCA STN+SNr vs STN"] = wide["STN+SNr"] - wide["STN"]
    return out.loc[:, [PAIR_COL, *GAIT_PCA_COLUMNS]]


def load_local_wide_by_pose(region: str) -> dict[str, pd.DataFrame]:
    """Load local normalized walk LFP effects for one region by pose."""
    if region not in LOCAL_REGIONS:
        raise ValueError(f"Unsupported local region: {region!r}. Expected one of {LOCAL_REGIONS}.")

    tables_by_pose: dict[str, pd.DataFrame] = {}
    for pose in POSE_ORDER:
        feature_rows = []
        for spec in LOCAL_BAND_SPECS:
            scalar = load_normalized_scalar_table(spec.input_path)
            filtered = _filter_walk_lfp_rows(
                scalar,
                region=region,
                pose=pose,
                required_bands=BAND_ORDER,
                table_name=str(spec.input_path),
            )
            filtered["feature"] = [
                _local_band_feature_key(region, band, spec.key)
                for band in filtered[BAND_COL].astype(str)
            ]
            feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        params = load_normalized_scalar_table(PARAMS_INPUT_PATH)
        params_filtered = _filter_walk_lfp_rows(
            params,
            region=region,
            pose=pose,
            required_bands=PARAM_ORDER,
            table_name=str(PARAMS_INPUT_PATH),
        )
        params_filtered["feature"] = [
            _local_param_feature_key(region, band)
            for band in params_filtered[BAND_COL].astype(str)
        ]
        feature_rows.append(params_filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        tables_by_pose[pose] = _pivot_feature_rows(
            feature_rows,
            ordered_features=LOCAL_FEATURE_ORDER_BY_REGION[region],
            table_name=f"{region} local normalized walk LFP table for {pose}",
        )

    return tables_by_pose


def load_connectivity_wide_by_pose() -> dict[str, pd.DataFrame]:
    """Load normalized walk connectivity effects by pose."""
    tables_by_pose: dict[str, pd.DataFrame] = {}
    for pose in POSE_ORDER:
        feature_rows = []
        for spec in CONNECTIVITY_SPECS:
            scalar = load_normalized_scalar_table(spec.input_path)
            filtered = _filter_walk_lfp_rows(
                scalar,
                region=CONNECTIVITY_REGION,
                pose=pose,
                required_bands=BAND_ORDER,
                table_name=str(spec.input_path),
            )
            filtered["feature"] = [
                _connectivity_feature_key(band, spec.key)
                for band in filtered[BAND_COL].astype(str)
            ]
            feature_rows.append(filtered.loc[:, [PAIR_COL, "feature", VALUE_COL]])

        tables_by_pose[pose] = _pivot_feature_rows(
            feature_rows,
            ordered_features=CONNECTIVITY_FEATURE_ORDER,
            table_name=f"connectivity normalized walk LFP table for {pose}",
        )

    return tables_by_pose


def load_normalized_scalar_table(input_path: str | Path) -> pd.DataFrame:
    """Load and validate one normalized walk scalar table."""
    input_path = Path(input_path)
    scalar = pd.read_csv(input_path)

    required_columns = [
        PAIR_COL,
        REGION_COL,
        BAND_COL,
        POSE_COL,
        MED_COL,
        TASK_COL,
        PHASE_COL,
        VALUE_COL,
    ]
    _validate_required_columns(scalar, required_columns=required_columns, table_name=str(input_path))

    prepared = scalar.loc[:, required_columns].copy()
    prepared = _normalize_pair_column(prepared, pair_col=PAIR_COL)
    for column in [REGION_COL, BAND_COL, POSE_COL, MED_COL, TASK_COL, PHASE_COL]:
        prepared[column] = prepared[column].astype(str)
    prepared[VALUE_COL] = pd.to_numeric(prepared[VALUE_COL], errors="coerce")
    return prepared


def filter_to_common_ids(
    x_wide: pd.DataFrame,
    gait_pca: pd.DataFrame,
    pair_col: str = PAIR_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter LFP and gait PCA tables to shared IDs in the same order."""
    x_wide = _normalize_pair_column(x_wide.copy(), pair_col=pair_col)
    gait_pca = _normalize_pair_column(gait_pca.copy(), pair_col=pair_col)

    if x_wide[pair_col].duplicated().any():
        raise ValueError("LFP wide table contains duplicate IDs.")
    if gait_pca[pair_col].duplicated().any():
        raise ValueError("Gait PCA table contains duplicate IDs.")

    common_ids = sorted(set(x_wide[pair_col]).intersection(gait_pca[pair_col]))
    if not common_ids:
        raise ValueError("No shared IDs were found between walk LFP features and gait PCA table.")

    x_common = (
        x_wide[x_wide[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )
    gait_common = (
        gait_pca[gait_pca[pair_col].isin(common_ids)]
        .sort_values(pair_col)
        .reset_index(drop=True)
    )

    if x_common[pair_col].tolist() != gait_common[pair_col].tolist():
        raise ValueError("Filtered LFP and gait PCA tables do not share the same ordered IDs.")

    return x_common, gait_common


def compute_walk_gait_pca_correlation(
    x_common: pd.DataFrame,
    gait_common: pd.DataFrame,
) -> CorrelationMatrixResult:
    """Compute globally FDR-adjusted Spearman correlations."""
    return compute_correlation_matrix(
        df_x=x_common,
        df_y=gait_common,
        pair_col=PAIR_COL,
        method="spearman",
        correction_mode="All",
        correction_method="bh",
    )


def export_pose_workbook(
    pose_tables: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    """Export one workbook with one sheet per pose."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for pose in POSE_ORDER:
            pose_tables[pose].to_excel(writer, sheet_name=pose, index=False)


def export_scale_subset_workbook(
    scale_subsets: dict[str, pd.DataFrame],
    output_path: str | Path = GAIT_PCA_SUBSET_OUTPUT_PATH,
) -> None:
    """Export gait PCA subsets used for each analysis group and pose."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, scale_subset in scale_subsets.items():
            scale_subset.to_excel(writer, sheet_name=sheet_name, index=False)


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
    """Run the full walk LFP-gait PCA 3-month correlation workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gait_pca = load_gait_pca_3m_distance(GAIT_PCA_INPUT_PATH)
    gait_pca.to_excel(GAIT_PCA_WIDE_OUTPUT_PATH, index=False)

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
    scale_subsets: dict[str, pd.DataFrame] = {}

    for group_name, pose_tables in wide_tables.items():
        output_spec = output_specs[group_name]
        pose_common_tables: dict[str, pd.DataFrame] = {}
        pose_results: dict[str, CorrelationMatrixResult] = {}

        for pose in POSE_ORDER:
            x_common, gait_common = filter_to_common_ids(pose_tables[pose], gait_pca)
            pose_common_tables[pose] = x_common
            scale_subsets[f"{group_name}_{pose}"] = gait_common

            correlation_result = compute_walk_gait_pca_correlation(x_common, gait_common)
            _validate_outputs(
                group_name=group_name,
                pose=pose,
                x_common=x_common,
                gait_common=gait_common,
                correlation_result=correlation_result,
                expected_features=output_spec["features"],
            )
            pose_results[pose] = correlation_result

        export_pose_workbook(pose_common_tables, output_spec["wide_path"])
        export_correlation_workbook_by_pose(pose_results, output_spec["corr_path"])
        results[group_name] = pose_results

    export_scale_subset_workbook(scale_subsets, GAIT_PCA_SUBSET_OUTPUT_PATH)
    return results


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
    required = set(required_bands)
    missing_bands = [band for band in required_bands if band not in available_bands]
    unexpected_bands = sorted(available_bands - required)
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
    pose: str,
    x_common: pd.DataFrame,
    gait_common: pd.DataFrame,
    correlation_result: CorrelationMatrixResult,
    expected_features: list[str],
) -> None:
    expected_x = x_common.columns.tolist()[1:]
    expected_y = gait_common.columns.tolist()[1:]

    if expected_x != expected_features:
        raise ValueError(f"{group_name} {pose}: x-axis features do not match the configured order.")
    if expected_y != GAIT_PCA_COLUMNS:
        raise ValueError(f"{group_name} {pose}: gait PCA y-axis columns do not match the configured order.")

    expected_shape = (len(GAIT_PCA_COLUMNS), len(expected_features))
    if correlation_result.r.shape != expected_shape:
        raise ValueError(
            f"{group_name} {pose}: expected correlation matrix shape {expected_shape}; "
            f"got {correlation_result.r.shape}."
        )

    for name, matrix in {
        "p": correlation_result.p,
        "p_raw": correlation_result.p_raw,
        "n": correlation_result.n,
    }.items():
        if not correlation_result.r.index.equals(matrix.index) or not correlation_result.r.columns.equals(matrix.columns):
            raise ValueError(f"{group_name} {pose}: matrix '{name}' is not aligned with r.")

    if correlation_result.p_adj is None:
        raise ValueError(f"{group_name} {pose}: expected globally adjusted p-values, but p_adj is None.")
    if not correlation_result.r.index.equals(correlation_result.p_adj.index) or not correlation_result.r.columns.equals(correlation_result.p_adj.columns):
        raise ValueError(f"{group_name} {pose}: matrix 'p_adj' is not aligned with r.")


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
    print(f"Saved gait PCA 3-month table: {GAIT_PCA_WIDE_OUTPUT_PATH}")
    print(f"Saved gait PCA subset workbook: {GAIT_PCA_SUBSET_OUTPUT_PATH}")
    print(f"Saved STN local wide workbook: {STN_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved SNr local wide workbook: {SNR_LOCAL_WIDE_OUTPUT_PATH}")
    print(f"Saved connectivity wide workbook: {CONNECTIVITY_WIDE_OUTPUT_PATH}")
    print(f"Saved STN local correlation workbook: {STN_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved SNr local correlation workbook: {SNR_LOCAL_CORRELATION_OUTPUT_PATH}")
    print(f"Saved connectivity correlation workbook: {CONNECTIVITY_CORRELATION_OUTPUT_PATH}")
    for group, pose_results in workflow_results.items():
        for pose_name, result in pose_results.items():
            print(f"{group} {pose_name}: correlation r shape {result.r.shape}")
