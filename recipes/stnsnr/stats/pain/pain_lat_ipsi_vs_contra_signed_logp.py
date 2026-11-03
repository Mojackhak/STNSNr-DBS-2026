from __future__ import annotations

import os

import csv
from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
PAIN_DIR = PROJECT / "summary" / "stats" / "pain"
OUTPUT_DIR = PAIN_DIR / "corr" / "lat_ipsi_vs_contra"
SIGNED_LOGP_OUTPUT_PATH = OUTPUT_DIR / "pain_lat_ipsi_vs_contra_signed_logp_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_TUKEY_FILE = "Lat-Phase-Region_tukey.csv"
CONNECTIVITY_TUKEY_FILE = "Lat-Phase_tukey.csv"
SOURCE_CONTRAST = "Ipsi - Contra"
COMPARISON_TEXT = "Ipsi vs Contra"
POSITIVE_DIRECTION_LABEL = "Ipsi > Contra"
NEGATIVE_DIRECTION_LABEL = "Ipsi < Contra"
VALUE_LIMIT = 3.0

PHASES = ("ONS", "SUS", "OFS", "PST")

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
    MetricSpec("raw", "Total power", PAIN_DIR / "tfr" / "raw" / "raw-mean-scalar"),
    MetricSpec("period", "Periodic power", PAIN_DIR / "tfr" / "period" / "period-mean-scalar"),
    MetricSpec(
        "aperiod",
        "Aperiodic power",
        PAIN_DIR / "tfr" / "aperiod" / "aperiod-mean-scalar",
    ),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        PAIN_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-rate",
        r"$\operatorname{arcsinh}$ burst rate",
        PAIN_DIR / "burst" / "burst" / "burst-rate-scalar",
    ),
    MetricSpec(
        "burst-duration",
        r"$\operatorname{ln}$ burst duration",
        PAIN_DIR / "burst" / "burst" / "burst-duration-scalar",
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        PAIN_DIR / "burst" / "burst" / "burst-mean-scalar",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        PAIN_DIR / "conn" / "coh" / "coh-mean-scalar",
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        PAIN_DIR / "conn" / "wpli" / "wpli-mean-scalar",
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        PAIN_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        PAIN_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "PSI", PAIN_DIR / "psi" / "psi" / "psi-mean-scalar"),
)

GROUPS = (
    GroupSpec(
        key="snr_local",
        label_top="SNr",
        metrics=LOCAL_METRICS,
        tukey_file=LOCAL_TUKEY_FILE,
        region="SNr",
    ),
    GroupSpec(
        key="stn_local",
        label_top="STN",
        metrics=LOCAL_METRICS,
        tukey_file=LOCAL_TUKEY_FILE,
        region="STN",
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
PARAMETER_ROOT = PAIN_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL_TOP = "Params"

EXPECTED_ROW_COUNT = (
    2 * len(LOCAL_METRICS) * len(BANDS) * len(PHASES)
    + len(CONNECTIVITY_METRICS) * len(BANDS) * len(PHASES)
    + len(PARAMETERS) * len(PARAMETER_REGIONS) * len(PHASES)
)


def clean_value(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.upper() in {"", "NA", "NAN", "NONE"} else text


def pain_change_label(label: object) -> str:
    text = clean_value(label)
    return text if text.startswith("Δ") else f"Δ{text}"


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


def load_lat_phase_row(
    path: Path,
    *,
    phase: str,
    region: str | None,
) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Tukey CSV: {path}")

    matches: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if clean_value(row.get("contrast")) != SOURCE_CONTRAST:
                continue
            if clean_value(row.get("Phase")) != phase:
                continue
            if region is not None and clean_value(row.get("Region")) != region:
                continue
            matches.append(row)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {SOURCE_CONTRAST!r} row in {path} "
            f"for phase={phase!r}, region={region!r}; got {len(matches)}."
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
    phase: str,
    region: str,
    source_csv: Path,
    row: dict[str, str],
) -> dict[str, object]:
    source_estimate = parse_float(row.get("estimate"))
    signed_estimate = source_estimate
    p_value = parse_float(row.get("p_tukey"))
    value = signed_logp(signed_estimate, p_value)
    value_display = float(np.clip(value, -VALUE_LIMIT, VALUE_LIMIT)) if math.isfinite(value) else math.nan

    return {
        "source_csv": str(source_csv),
        "group": group_key,
        "label_top": label_top,
        "phase": phase,
        "region": region,
        "metric": metric_key,
        "metric_label": metric_label,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "source_model": MODEL_DIR,
        "source_contrast": SOURCE_CONTRAST,
        "comparison": COMPARISON_TEXT,
        "positive_direction": POSITIVE_DIRECTION_LABEL,
        "negative_direction": NEGATIVE_DIRECTION_LABEL,
        "source_estimate_ipsi_minus_contra": source_estimate,
        "signed_estimate": signed_estimate,
        "estimate_label": "Ipsi - Contra",
        "p_tukey": p_value,
        "signed_logp": value,
        "signed_logp_display": value_display,
        "star": p_to_star(p_value),
    }


def build_group_records(group: GroupSpec) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for metric in group.metrics:
        for band in BANDS:
            path = source_path(metric, band, group.tukey_file)
            for phase in PHASES:
                row = load_lat_phase_row(path, phase=phase, region=group.region)
                records.append(
                    make_signed_logp_record(
                        group_key=group.key,
                        label_top=group.label_top,
                        metric_key=metric.key,
                        metric_label=pain_change_label(metric.label),
                        band=band,
                        band_label=BAND_LABELS[band],
                        parameter="",
                        parameter_label="",
                        phase=phase,
                        region=group.region or "",
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
            for phase in PHASES:
                row = load_lat_phase_row(path, phase=phase, region=region)
                records.append(
                    make_signed_logp_record(
                        group_key=PARAMETER_GROUP_KEY,
                        label_top=PARAMETER_LABEL_TOP,
                        metric_key="params",
                        metric_label=region,
                        band="",
                        band_label="",
                        parameter=parameter.key,
                        parameter_label=pain_change_label(parameter.label),
                        phase=phase,
                        region=region,
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
    if (table["source_contrast"] != SOURCE_CONTRAST).any():
        raise ValueError(f"Source contrast must be {SOURCE_CONTRAST!r}.")
    if tuple(table["phase"].drop_duplicates()) != PHASES:
        raise ValueError(f"Phase order must be {list(PHASES)}.")
    if (table["comparison"] != COMPARISON_TEXT).any():
        raise ValueError(f"Comparison must be {COMPARISON_TEXT!r}.")
    if (table["positive_direction"] != POSITIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Positive direction must be {POSITIVE_DIRECTION_LABEL!r}.")
    if (table["negative_direction"] != NEGATIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Negative direction must be {NEGATIVE_DIRECTION_LABEL!r}.")

    banded_metric_labels = table.loc[
        table["group"] != PARAMETER_GROUP_KEY,
        "metric_label",
    ].astype(str)
    if not banded_metric_labels.str.startswith("Δ").all():
        raise ValueError("BL-normalized banded metric labels must start with 'Δ'.")

    parameter_labels = table.loc[
        table["group"] == PARAMETER_GROUP_KEY,
        "parameter_label",
    ].astype(str)
    if not parameter_labels.str.startswith("Δ").all():
        raise ValueError("BL-normalized parameter labels must start with 'Δ'.")

    source_estimate = pd.to_numeric(table["source_estimate_ipsi_minus_contra"], errors="coerce")
    signed_estimate = pd.to_numeric(table["signed_estimate"], errors="coerce")
    if not np.allclose(source_estimate, signed_estimate, equal_nan=True):
        raise ValueError("signed_estimate must equal source_estimate_ipsi_minus_contra.")

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
