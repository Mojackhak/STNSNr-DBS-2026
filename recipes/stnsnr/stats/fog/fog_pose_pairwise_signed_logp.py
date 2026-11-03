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
FOG_DIR = PROJECT / "summary" / "stats" / "fog"
OUTPUT_DIR = FOG_DIR / "corr" / "pose_pairwise"
SIGNED_LOGP_OUTPUT_PATH = OUTPUT_DIR / "fog_pose_pairwise_signed_logp_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_TUKEY_FILE = "Pose-Region_tukey.csv"
CONNECTIVITY_TUKEY_FILE = "Pose_tukey.csv"
VALUE_LIMIT = 3.0

POSE_CONTRASTS = (
    ("Stand", "Walk"),
    ("Stand", "FoG"),
    ("Walk", "FoG"),
)
SOURCE_CONTRASTS = tuple(f"{first} - {second}" for first, second in POSE_CONTRASTS)
DISPLAY_COMPARISONS = tuple(f"{first} vs {second}" for first, second in POSE_CONTRASTS)

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
    MetricSpec("raw", "Total power", FOG_DIR / "tfr" / "raw" / "raw-mean-scalar"),
    MetricSpec("period", "Periodic power", FOG_DIR / "tfr" / "period" / "period-mean-scalar"),
    MetricSpec(
        "aperiod",
        "Aperiodic power",
        FOG_DIR / "tfr" / "aperiod" / "aperiod-mean-scalar",
    ),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        FOG_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-rate",
        r"$\operatorname{arcsinh}$ burst rate",
        FOG_DIR / "burst" / "burst" / "burst-rate-scalar",
    ),
    MetricSpec(
        "burst-duration",
        r"$\operatorname{ln}$ burst duration",
        FOG_DIR / "burst" / "burst" / "burst-duration-scalar",
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        FOG_DIR / "burst" / "burst" / "burst-mean-scalar",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        FOG_DIR / "conn" / "coh" / "coh-mean-scalar",
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        FOG_DIR / "conn" / "wpli" / "wpli-mean-scalar",
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        FOG_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        FOG_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "PSI", FOG_DIR / "psi" / "psi" / "psi-mean-scalar"),
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
PARAMETER_ROOT = FOG_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL_TOP = "Params"

EXPECTED_ROW_COUNT = (
    2 * len(LOCAL_METRICS) * len(BANDS) * len(POSE_CONTRASTS)
    + len(CONNECTIVITY_METRICS) * len(BANDS) * len(POSE_CONTRASTS)
    + len(PARAMETERS) * len(PARAMETER_REGIONS) * len(POSE_CONTRASTS)
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


def source_contrast(first_pose: str, second_pose: str) -> str:
    return f"{first_pose} - {second_pose}"


def display_comparison(first_pose: str, second_pose: str) -> str:
    return f"{first_pose} vs {second_pose}"


def load_pose_row(
    path: Path,
    *,
    first_pose: str,
    second_pose: str,
    region: str | None,
) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Tukey CSV: {path}")

    contrast = source_contrast(first_pose, second_pose)
    matches: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if clean_value(row.get("contrast")) != contrast:
                continue
            if region is not None and clean_value(row.get("Region")) != region:
                continue
            matches.append(row)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {contrast!r} row in {path} "
            f"for region={region!r}; got {len(matches)}."
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
    first_pose: str,
    second_pose: str,
    source_csv: Path,
    row: dict[str, str],
) -> dict[str, object]:
    source_estimate = parse_float(row.get("estimate"))
    flipped_estimate = -source_estimate if math.isfinite(source_estimate) else math.nan
    p_value = parse_float(row.get("p_tukey"))
    value = signed_logp(flipped_estimate, p_value)
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
        "source_pose_first": first_pose,
        "source_pose_second": second_pose,
        "plot_pose_first": first_pose,
        "plot_pose_second": second_pose,
        "source_contrast": clean_value(row.get("contrast")),
        "comparison": display_comparison(first_pose, second_pose),
        "estimate_label": f"{second_pose} - {first_pose}",
        "source_model": MODEL_DIR,
        "source_estimate_first_minus_second": source_estimate,
        "estimate_second_minus_first": flipped_estimate,
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
            for first_pose, second_pose in POSE_CONTRASTS:
                row = load_pose_row(
                    path,
                    first_pose=first_pose,
                    second_pose=second_pose,
                    region=group.region,
                )
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
                        first_pose=first_pose,
                        second_pose=second_pose,
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
            for first_pose, second_pose in POSE_CONTRASTS:
                row = load_pose_row(
                    path,
                    first_pose=first_pose,
                    second_pose=second_pose,
                    region=region,
                )
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
                        first_pose=first_pose,
                        second_pose=second_pose,
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
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} signed-logP rows, got {len(table)}.")

    if (table["source_model"] != MODEL_DIR).any():
        raise ValueError(f"All source rows must use {MODEL_DIR}.")
    if set(table["source_contrast"].unique()) != set(SOURCE_CONTRASTS):
        raise ValueError(f"Source contrasts must match {list(SOURCE_CONTRASTS)}.")
    if tuple(table["comparison"].drop_duplicates()) != DISPLAY_COMPARISONS:
        raise ValueError(f"Comparison order must be {list(DISPLAY_COMPARISONS)}.")

    source_estimate = pd.to_numeric(table["source_estimate_first_minus_second"], errors="coerce")
    signed_estimate = pd.to_numeric(table["estimate_second_minus_first"], errors="coerce")
    if not np.allclose(signed_estimate, -source_estimate, equal_nan=True):
        raise ValueError("estimate_second_minus_first must equal -source_estimate_first_minus_second.")

    display = pd.to_numeric(table["signed_logp_display"], errors="coerce")
    finite_display = display[np.isfinite(display)]
    if ((finite_display < -VALUE_LIMIT) | (finite_display > VALUE_LIMIT)).any():
        raise ValueError(f"signed_logp_display must be clipped to [-{VALUE_LIMIT}, {VALUE_LIMIT}].")

    bad_paths = [
        path
        for path in table["source_csv"].astype(str)
        if f"/{MODEL_DIR}/" not in path
    ]
    if bad_paths:
        raise ValueError(f"Found source paths outside {MODEL_DIR}: {bad_paths[:5]}")


def run_workflow(output_path: str | Path = SIGNED_LOGP_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_signed_logp_table()
    table.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_path = run_workflow()
    print(f"Saved {EXPECTED_ROW_COUNT} signed-logP rows: {saved_path}")
