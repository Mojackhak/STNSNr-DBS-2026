from __future__ import annotations

import os

import csv
from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
WALK_DIR = PROJECT / "summary" / "stats" / "walk"
OUTPUT_DIR = WALK_DIR / "corr"
SIGNED_LOGP_OUTPUT_PATH = OUTPUT_DIR / "walk_task_dual_vs_single_signed_logp_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_TUKEY_FILE = "Task-Pose-Region_tukey.csv"
CONNECTIVITY_TUKEY_FILE = "Task-Pose_tukey.csv"
SOURCE_CONTRAST = "Single - Dual"
COMPARISON_TEXT = "Dual vs Single"
POSITIVE_DIRECTION_LABEL = "Dual > Single"
NEGATIVE_DIRECTION_LABEL = "Dual < Single"
VALUE_LIMIT = 3.0

POSES = ("Sit", "Stand", "Walk", "Turn")
BANDS = ("Delta", "Theta", "Alpha", "Beta_low", "Beta_high", "Gamma")
BAND_LABELS = {
    "Delta": "δ",
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    base_dir: Path


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label_top: str
    metrics: tuple[MetricSpec, ...]
    tukey_file: str
    region: str | None


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


LOCAL_METRICS = (
    MetricSpec("raw", "Total power", WALK_DIR / "tfr" / "raw" / "raw-mean-scalar"),
    MetricSpec("period", "Periodic power", WALK_DIR / "tfr" / "period" / "period-mean-scalar"),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        WALK_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-rate",
        r"$\operatorname{arcsinh}$ burst rate",
        WALK_DIR / "burst" / "burst" / "burst-rate-scalar",
    ),
    MetricSpec(
        "burst-duration",
        r"$\operatorname{ln}$ burst duration",
        WALK_DIR / "burst" / "burst" / "burst-duration-scalar",
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        WALK_DIR / "burst" / "burst" / "burst-mean-scalar",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        WALK_DIR / "conn" / "coh" / "coh-mean-scalar",
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        WALK_DIR / "conn" / "wpli" / "wpli-mean-scalar",
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        WALK_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        WALK_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "PSI", WALK_DIR / "psi" / "psi" / "psi-mean-scalar"),
)

GROUPS = (
    GroupSpec(
        key="stn_local",
        label_top="STN",
        metrics=LOCAL_METRICS,
        tukey_file=LOCAL_TUKEY_FILE,
        region="STN",
    ),
    GroupSpec(
        key="snr_local",
        label_top="SNr",
        metrics=LOCAL_METRICS,
        tukey_file=LOCAL_TUKEY_FILE,
        region="SNr",
    ),
    GroupSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        metrics=CONNECTIVITY_METRICS,
        tukey_file=CONNECTIVITY_TUKEY_FILE,
        region=None,
    ),
)

PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)
PARAMETER_ROOT = WALK_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL_TOP = "Params"

EXPECTED_ROW_COUNT = (
    2 * len(LOCAL_METRICS) * len(BANDS) * len(POSES)
    + len(CONNECTIVITY_METRICS) * len(BANDS) * len(POSES)
    + len(PARAMETERS) * len(PARAMETER_REGIONS) * len(POSES)
)


def clean_value(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.upper() in {"", "NA", "NAN", "NONE"} else text


def parse_float(value: object) -> float:
    text = clean_value(value)
    if not text:
        return math.nan
    try:
        parsed = float(text)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def p_to_star(p_value: float) -> str:
    if not math.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def signed_logp(estimate: float, p_value: float) -> float:
    if not math.isfinite(estimate) or not math.isfinite(p_value) or p_value <= 0:
        return math.nan
    if estimate == 0:
        return 0.0
    return math.copysign(-math.log10(p_value), estimate)


def source_path(metric: MetricSpec, band: str, tukey_file: str) -> Path:
    return metric.base_dir / band / MODEL_DIR / tukey_file


def parameter_source_path(parameter: ParameterSpec) -> Path:
    return PARAMETER_ROOT / parameter.key / MODEL_DIR / LOCAL_TUKEY_FILE


def load_task_pose_row(path: Path, *, pose: str, region: str | None) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Tukey CSV: {path}")

    matches: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if clean_value(row.get("contrast")) != SOURCE_CONTRAST:
                continue
            if clean_value(row.get("Pose")) != pose:
                continue
            if region is not None and clean_value(row.get("Region")) != region:
                continue
            matches.append(row)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {SOURCE_CONTRAST!r} row in {path} "
            f"for pose={pose!r}, region={region!r}; got {len(matches)}."
        )
    return matches[0]


def make_signed_logp_record(
    *,
    group_key: str,
    label_top: str,
    metric_key: str,
    metric_label: str,
    band: str,
    band_label: str,
    parameter: str,
    parameter_label: str,
    region: str,
    pose: str,
    source_csv: Path,
    row: dict[str, str],
) -> dict[str, object]:
    source_estimate = parse_float(row.get("estimate"))
    estimate_dual_minus_single = -source_estimate if math.isfinite(source_estimate) else math.nan
    p_value = parse_float(row.get("p_tukey"))
    value = signed_logp(estimate_dual_minus_single, p_value)
    value_display = float(np.clip(value, -VALUE_LIMIT, VALUE_LIMIT)) if math.isfinite(value) else math.nan

    return {
        "group": group_key,
        "label_top": label_top,
        "metric": metric_key,
        "metric_label": metric_label,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "region": region,
        "pose": pose,
        "comparison": COMPARISON_TEXT,
        "positive_direction": POSITIVE_DIRECTION_LABEL,
        "negative_direction": NEGATIVE_DIRECTION_LABEL,
        "source_task_first": "Single",
        "source_task_second": "Dual",
        "plot_task_first": "Dual",
        "plot_task_second": "Single",
        "source_model": MODEL_DIR,
        "source_contrast": clean_value(row.get("contrast")),
        "source_estimate_single_minus_dual": source_estimate,
        "estimate_dual_minus_single": estimate_dual_minus_single,
        "p_tukey": p_value,
        "signed_logp": value,
        "signed_logp_display": value_display,
        "star": p_to_star(p_value),
        "source_csv": str(source_csv),
    }


def build_group_records(group: GroupSpec) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for metric in group.metrics:
        for band in BANDS:
            path = source_path(metric, band, group.tukey_file)
            for pose in POSES:
                row = load_task_pose_row(path, pose=pose, region=group.region)
                records.append(
                    make_signed_logp_record(
                        group_key=group.key,
                        label_top=group.label_top,
                        metric_key=metric.key,
                        metric_label=metric.label,
                        band=band,
                        band_label=BAND_LABELS[band],
                        parameter="",
                        parameter_label="",
                        region=group.region or "",
                        pose=pose,
                        source_csv=path,
                        row=row,
                    )
                )
    return records


def build_parameter_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for parameter in PARAMETERS:
        path = parameter_source_path(parameter)
        for region in PARAMETER_REGIONS:
            for pose in POSES:
                row = load_task_pose_row(path, pose=pose, region=region)
                records.append(
                    make_signed_logp_record(
                        group_key=PARAMETER_GROUP_KEY,
                        label_top=PARAMETER_LABEL_TOP,
                        metric_key="params",
                        metric_label=region,
                        band="",
                        band_label="",
                        parameter=parameter.key,
                        parameter_label=parameter.label,
                        region=region,
                        pose=pose,
                        source_csv=path,
                        row=row,
                    )
                )
    return records


def build_signed_logp_table() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group in GROUPS:
        records.extend(build_group_records(group))
    records.extend(build_parameter_records())

    table = pd.DataFrame.from_records(records)
    validate_signed_logp_table(table)
    return table


def validate_signed_logp_table(table: pd.DataFrame) -> None:
    if len(table) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} rows; got {len(table)}.")

    required_columns = [
        "group",
        "metric",
        "band",
        "parameter",
        "region",
        "pose",
        "source_model",
        "source_contrast",
        "source_estimate_single_minus_dual",
        "estimate_dual_minus_single",
        "p_tukey",
        "signed_logp",
        "signed_logp_display",
        "source_csv",
    ]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Signed-logP table is missing columns: {missing_columns}")

    bad_models = table.loc[table["source_model"] != MODEL_DIR, "source_csv"].tolist()
    if bad_models:
        raise ValueError(f"Found non-{MODEL_DIR} source rows: {bad_models[:5]}")

    bad_contrasts = table.loc[table["source_contrast"] != SOURCE_CONTRAST, "source_contrast"].tolist()
    if bad_contrasts:
        raise ValueError(f"Found non-{SOURCE_CONTRAST!r} source rows: {bad_contrasts[:5]}")

    bad_poses = table.loc[~table["pose"].isin(POSES), ["pose", "source_csv"]].to_dict("records")
    if bad_poses:
        raise ValueError(f"Found unexpected poses: {bad_poses[:5]}")

    bad_paths = [
        source_csv
        for source_csv in table["source_csv"].astype(str)
        if f"/{MODEL_DIR}/" not in source_csv
    ]
    if bad_paths:
        raise ValueError(f"Found source paths outside {MODEL_DIR}: {bad_paths[:5]}")

    numeric_columns = [
        "source_estimate_single_minus_dual",
        "estimate_dual_minus_single",
        "p_tukey",
        "signed_logp",
        "signed_logp_display",
    ]
    for column in numeric_columns:
        invalid_count = int(pd.to_numeric(table[column], errors="coerce").isna().sum())
        if invalid_count:
            raise ValueError(f"Column {column!r} contains {invalid_count} missing values.")

    source_estimate = pd.to_numeric(table["source_estimate_single_minus_dual"], errors="coerce")
    flipped_estimate = pd.to_numeric(table["estimate_dual_minus_single"], errors="coerce")
    if not np.allclose(flipped_estimate.to_numpy(), -source_estimate.to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError("Found rows where Dual-minus-Single estimate does not equal -source estimate.")

    if ((table["signed_logp_display"] < -VALUE_LIMIT) | (table["signed_logp_display"] > VALUE_LIMIT)).any():
        raise ValueError(f"Found display values outside [-{VALUE_LIMIT}, {VALUE_LIMIT}].")

    duplicate_columns = ["group", "region", "metric", "band", "parameter", "pose"]
    duplicate_mask = table.duplicated(duplicate_columns, keep=False)
    if duplicate_mask.any():
        examples = table.loc[duplicate_mask, duplicate_columns].head(5).to_dict("records")
        raise ValueError(f"Found duplicate signed-logP cells: {examples}")


def export_signed_logp_table(
    table: pd.DataFrame,
    output_path: str | Path = SIGNED_LOGP_OUTPUT_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path


def run_workflow() -> pd.DataFrame:
    table = build_signed_logp_table()
    export_signed_logp_table(table, SIGNED_LOGP_OUTPUT_PATH)
    return table


def validate_output_file(
    output_path: str | Path = SIGNED_LOGP_OUTPUT_PATH,
    expected_rows: int = EXPECTED_ROW_COUNT,
) -> None:
    output_path = Path(output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {output_path}")
    table = pd.read_csv(output_path)
    validate_signed_logp_table(table)
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} output rows; got {len(table)}.")


def iter_expected_source_paths() -> Iterable[Path]:
    for group in GROUPS:
        for metric in group.metrics:
            for band in BANDS:
                yield source_path(metric, band, group.tukey_file)
    for parameter in PARAMETERS:
        yield parameter_source_path(parameter)


def validate_expected_source_paths() -> None:
    missing = [str(path) for path in set(iter_expected_source_paths()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source Tukey CSV files: {missing[:10]}")


if __name__ == "__main__":
    validate_expected_source_paths()
    signed_logp_table = run_workflow()
    validate_output_file(SIGNED_LOGP_OUTPUT_PATH)
    print(f"Saved {len(signed_logp_table)} rows to {SIGNED_LOGP_OUTPUT_PATH}")

