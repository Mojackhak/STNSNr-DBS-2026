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
PAIN_EVOKED_DIR = PROJECT / "summary" / "stats" / "pain_evoked"
OUTPUT_DIR = PAIN_EVOKED_DIR / "corr" / "phase_ons_vs_ofs"
OUTPUT_PATH = OUTPUT_DIR / "pain_evoked_phase_ons_vs_ofs_signed_logp_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_TUKEY_FILE = "Phase-Region_tukey.csv"
CONNECTIVITY_TUKEY_FILE = "Phase_tukey.csv"
SOURCE_CONTRAST = "ONS - OFS"
COMPARISON_TEXT = "ONS vs OFS"
POSITIVE_DIRECTION_LABEL = "ONS > OFS"
NEGATIVE_DIRECTION_LABEL = "ONS < OFS"
VALUE_LIMIT = 3.0

REGIONS = ("SNr", "STN")
CONNECTIVITY_REGION = "SNr-/→STN"

EVOKED_METRICS = (
    ("duration", "Duration"),
    ("onset_time", "Onset time"),
    ("offset_time", "Offset time"),
    ("peak_time", "Peak time"),
    ("peak_value", "Peak value"),
    ("auc_rel_left", "AUC"),
    ("mean_slope_rise", "Rise slope"),
    ("mean_slope_fall", "Fall slope"),
)

BAND_LABELS = {
    "Delta": "δ",
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Gamma": "γ",
}


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    feature_group: str
    path_parts: tuple[str, ...]
    folder_prefix: str
    metric_label: str
    band: str
    region: str | None
    tukey_file: str


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


LOCAL_FEATURES_BY_REGION = {
    "SNr": (
        FeatureSpec(
            "raw",
            "frequency_metric",
            ("tfr", "raw"),
            "raw-mean",
            "Total power",
            "Delta",
            "SNr",
            LOCAL_TUKEY_FILE,
        ),
        FeatureSpec(
            "raw",
            "frequency_metric",
            ("tfr", "raw"),
            "raw-mean",
            "Total power",
            "Theta",
            "SNr",
            LOCAL_TUKEY_FILE,
        ),
    ),
    "STN": (
        FeatureSpec(
            "raw",
            "frequency_metric",
            ("tfr", "raw"),
            "raw-mean",
            "Total power",
            "Delta",
            "STN",
            LOCAL_TUKEY_FILE,
        ),
        FeatureSpec(
            "period",
            "frequency_metric",
            ("tfr", "period"),
            "period-mean",
            "Periodic power",
            "Delta",
            "STN",
            LOCAL_TUKEY_FILE,
        ),
        FeatureSpec(
            "raw",
            "frequency_metric",
            ("tfr", "raw"),
            "raw-mean",
            "Total power",
            "Theta",
            "STN",
            LOCAL_TUKEY_FILE,
        ),
        FeatureSpec(
            "period",
            "frequency_metric",
            ("tfr", "period"),
            "period-mean",
            "Periodic power",
            "Alpha",
            "STN",
            LOCAL_TUKEY_FILE,
        ),
        FeatureSpec(
            "period",
            "frequency_metric",
            ("tfr", "period"),
            "period-mean",
            "Periodic power",
            "Beta_low",
            "STN",
            LOCAL_TUKEY_FILE,
        ),
        FeatureSpec(
            "period",
            "frequency_metric",
            ("tfr", "period"),
            "period-mean",
            "Periodic power",
            "Gamma",
            "STN",
            LOCAL_TUKEY_FILE,
        ),
    ),
}

CONNECTIVITY_FEATURES = (
    FeatureSpec(
        "coh",
        "connectivity_metric",
        ("conn", "coh"),
        "coh-mean",
        "Coherence",
        "Delta",
        None,
        CONNECTIVITY_TUKEY_FILE,
    ),
    FeatureSpec(
        "wpli",
        "connectivity_metric",
        ("conn", "wpli"),
        "wpli-mean",
        "wPLI",
        "Delta",
        None,
        CONNECTIVITY_TUKEY_FILE,
    ),
)

PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)

EXPECTED_ROW_COUNT = (
    sum(len(features) for features in LOCAL_FEATURES_BY_REGION.values()) * len(EVOKED_METRICS)
    + len(CONNECTIVITY_FEATURES) * len(EVOKED_METRICS)
    + len(PARAMETERS) * len(REGIONS) * len(EVOKED_METRICS)
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


def feature_source_path(spec: FeatureSpec, evoked_metric: str) -> Path:
    return (
        PAIN_EVOKED_DIR.joinpath(*spec.path_parts)
        / f"{spec.folder_prefix}-{evoked_metric}"
        / spec.band
        / MODEL_DIR
        / spec.tukey_file
    )


def parameter_source_path(parameter: ParameterSpec, evoked_metric: str) -> Path:
    return (
        PAIN_EVOKED_DIR
        / "params"
        / "params"
        / f"params-mean-{evoked_metric}"
        / parameter.key
        / MODEL_DIR
        / LOCAL_TUKEY_FILE
    )


def load_phase_row(path: Path, *, region: str | None) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Tukey CSV: {path}")

    matches: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if clean_value(row.get("contrast")) != SOURCE_CONTRAST:
                continue
            if region is not None and clean_value(row.get("Region")) != region:
                continue
            matches.append(row)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {SOURCE_CONTRAST!r} row in {path} "
            f"for region={region!r}; got {len(matches)}."
        )
    return matches[0]


def make_signed_logp_record(
    *,
    feature_group: str,
    metric: str,
    metric_label: str,
    band: str,
    band_label: str,
    parameter: str,
    parameter_label: str,
    evoked_metric: str,
    evoked_label: str,
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
        "feature_group": feature_group,
        "region": region,
        "metric": metric,
        "metric_label": metric_label,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "evoked_metric": evoked_metric,
        "evoked_label": evoked_label,
        "source_model": MODEL_DIR,
        "source_contrast": SOURCE_CONTRAST,
        "source_phase_first": "ONS",
        "source_phase_second": "OFS",
        "comparison": COMPARISON_TEXT,
        "positive_direction": POSITIVE_DIRECTION_LABEL,
        "negative_direction": NEGATIVE_DIRECTION_LABEL,
        "source_estimate_ons_minus_ofs": source_estimate,
        "signed_estimate": signed_estimate,
        "estimate_label": SOURCE_CONTRAST,
        "p_tukey": p_value,
        "signed_logp": value,
        "signed_logp_display": value_display,
        "star": p_to_star(p_value),
    }


def build_feature_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for evoked_metric, evoked_label in EVOKED_METRICS:
        for region, specs in LOCAL_FEATURES_BY_REGION.items():
            for spec in specs:
                path = feature_source_path(spec, evoked_metric)
                row = load_phase_row(path, region=region)
                records.append(
                    make_signed_logp_record(
                        feature_group=spec.feature_group,
                        metric=spec.key,
                        metric_label=spec.metric_label,
                        band=spec.band,
                        band_label=BAND_LABELS[spec.band],
                        parameter="",
                        parameter_label="",
                        evoked_metric=evoked_metric,
                        evoked_label=evoked_label,
                        region=region,
                        source_csv=path,
                        row=row,
                    )
                )

        for spec in CONNECTIVITY_FEATURES:
            path = feature_source_path(spec, evoked_metric)
            row = load_phase_row(path, region=None)
            records.append(
                make_signed_logp_record(
                    feature_group=spec.feature_group,
                    metric=spec.key,
                    metric_label=spec.metric_label,
                    band=spec.band,
                    band_label=BAND_LABELS[spec.band],
                    parameter="",
                    parameter_label="",
                    evoked_metric=evoked_metric,
                    evoked_label=evoked_label,
                    region=CONNECTIVITY_REGION,
                    source_csv=path,
                    row=row,
                )
            )
    return records


def build_parameter_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for evoked_metric, evoked_label in EVOKED_METRICS:
        for parameter in PARAMETERS:
            path = parameter_source_path(parameter, evoked_metric)
            for region in REGIONS:
                row = load_phase_row(path, region=region)
                records.append(
                    make_signed_logp_record(
                        feature_group="parameter",
                        metric="params",
                        metric_label="Aperiodic parameters",
                        band=parameter.key,
                        band_label=parameter.label,
                        parameter=parameter.key,
                        parameter_label=parameter.label,
                        evoked_metric=evoked_metric,
                        evoked_label=evoked_label,
                        region=region,
                        source_csv=path,
                        row=row,
                    )
                )
    return records


def build_signed_logp_table() -> pd.DataFrame:
    table = pd.DataFrame.from_records([*build_feature_records(), *build_parameter_records()])
    validate_signed_logp_table(table)
    return table


def validate_signed_logp_table(table: pd.DataFrame) -> None:
    if len(table) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} signed-logP rows, got {len(table)}.")

    if (table["source_model"] != MODEL_DIR).any():
        raise ValueError(f"All source rows must use {MODEL_DIR}.")
    if (table["source_contrast"] != SOURCE_CONTRAST).any():
        raise ValueError(f"Source contrast must be {SOURCE_CONTRAST!r}.")
    if (table["comparison"] != COMPARISON_TEXT).any():
        raise ValueError(f"Comparison must be {COMPARISON_TEXT!r}.")
    if (table["positive_direction"] != POSITIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Positive direction must be {POSITIVE_DIRECTION_LABEL!r}.")
    if (table["negative_direction"] != NEGATIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Negative direction must be {NEGATIVE_DIRECTION_LABEL!r}.")

    source_estimate = pd.to_numeric(table["source_estimate_ons_minus_ofs"], errors="coerce")
    signed_estimate = pd.to_numeric(table["signed_estimate"], errors="coerce")
    if not np.allclose(source_estimate, signed_estimate, equal_nan=True):
        raise ValueError("signed_estimate must equal source_estimate_ons_minus_ofs.")

    for column in ["p_tukey", "signed_logp", "signed_logp_display"]:
        invalid_count = int(pd.to_numeric(table[column], errors="coerce").isna().sum())
        if invalid_count:
            raise ValueError(f"Column {column!r} contains {invalid_count} invalid values.")

    display = pd.to_numeric(table["signed_logp_display"], errors="coerce")
    if ((display < -VALUE_LIMIT) | (display > VALUE_LIMIT)).any():
        raise ValueError(f"signed_logp_display must be clipped to [-{VALUE_LIMIT}, {VALUE_LIMIT}].")

    bad_paths = [
        path
        for path in table["source_csv"].astype(str)
        if f"/{MODEL_DIR}/" not in path
    ]
    if bad_paths:
        raise ValueError(f"Found source paths outside {MODEL_DIR}: {bad_paths[:5]}")


def run_workflow(output_path: str | Path = OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_signed_logp_table()
    table.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_path = run_workflow()
    print(f"Saved {EXPECTED_ROW_COUNT} signed-logP rows: {saved_path}")
