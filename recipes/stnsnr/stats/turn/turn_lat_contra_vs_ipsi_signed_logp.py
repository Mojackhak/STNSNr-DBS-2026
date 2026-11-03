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
TURN_DIR = PROJECT / "summary" / "stats" / "turn"
OUTPUT_DIR = TURN_DIR / "corr" / "lat_contra_vs_ipsi"
SIGNED_LOGP_OUTPUT_PATH = OUTPUT_DIR / "turn_lat_contra_vs_ipsi_signed_logp_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_TUKEY_FILE = "Lat-Region_tukey.csv"
CONNECTIVITY_TUKEY_FILE = "Lat_tukey.csv"
SOURCE_CONTRAST = "Ipsi - Contra"
COMPARISON_TEXT = "Contra vs Ipsi"
POSITIVE_DIRECTION_LABEL = "Contra < Ipsi"
NEGATIVE_DIRECTION_LABEL = "Contra > Ipsi"
VALUE_LIMIT = 3.0

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
    MetricSpec("raw", "Total power", TURN_DIR / "tfr" / "raw" / "raw-mean-scalar"),
    MetricSpec("period", "Periodic power", TURN_DIR / "tfr" / "period" / "period-mean-scalar"),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        TURN_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-rate",
        r"$\operatorname{arcsinh}$ burst rate",
        TURN_DIR / "burst" / "burst" / "burst-rate-scalar",
    ),
    MetricSpec(
        "burst-duration",
        r"$\operatorname{ln}$ burst duration",
        TURN_DIR / "burst" / "burst" / "burst-duration-scalar",
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        TURN_DIR / "burst" / "burst" / "burst-mean-scalar",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        TURN_DIR / "conn" / "coh" / "coh-mean-scalar",
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        TURN_DIR / "conn" / "wpli" / "wpli-mean-scalar",
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        TURN_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        TURN_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "PSI", TURN_DIR / "psi" / "psi" / "psi-mean-scalar"),
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
PARAMETER_ROOT = TURN_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL_TOP = "Params"

EXPECTED_ROW_COUNT = (
    sum(len(group.metrics) * len(BANDS) for group in GROUPS)
    + len(PARAMETERS) * len(PARAMETER_REGIONS)
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


def load_lat_row(path: Path, *, region: str | None) -> dict[str, str]:
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
    group_key: str,
    label_top: str,
    metric_key: str,
    metric_label: str,
    band: str,
    band_label: str,
    parameter: str,
    parameter_label: str,
    region: str,
    source_csv: Path,
    row: dict[str, str],
) -> dict[str, object]:
    source_estimate = parse_float(row.get("estimate"))
    signed_estimate = source_estimate
    p_value = parse_float(row.get("p_tukey"))
    value = signed_logp(signed_estimate, p_value)
    display_value = float(np.clip(value, -VALUE_LIMIT, VALUE_LIMIT)) if math.isfinite(value) else math.nan
    star = p_to_star(p_value)

    return {
        "source_csv": str(source_csv),
        "group": group_key,
        "label_top": label_top,
        "metric": metric_key,
        "metric_label": metric_label,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "region": region,
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
        "signed_logp_display": display_value,
        "star": star,
    }


def iter_group_records() -> Iterable[dict[str, object]]:
    for group in GROUPS:
        for metric in group.metrics:
            for band in BANDS:
                path = source_path(metric, band, group.tukey_file)
                row = load_lat_row(path, region=group.region)
                yield make_signed_logp_record(
                    group_key=group.key,
                    label_top=group.label_top,
                    metric_key=metric.key,
                    metric_label=metric.label,
                    band=band,
                    band_label=BAND_LABELS[band],
                    parameter="",
                    parameter_label="",
                    region=group.region or "",
                    source_csv=path,
                    row=row,
                )


def iter_parameter_records() -> Iterable[dict[str, object]]:
    for parameter in PARAMETERS:
        path = parameter_source_path(parameter)
        for region in PARAMETER_REGIONS:
            row = load_lat_row(path, region=region)
            yield make_signed_logp_record(
                group_key=PARAMETER_GROUP_KEY,
                label_top=PARAMETER_LABEL_TOP,
                metric_key="params",
                metric_label=region,
                band=parameter.key,
                band_label=parameter.label,
                parameter=parameter.key,
                parameter_label=parameter.label,
                region=region,
                source_csv=path,
                row=row,
            )


def build_signed_logp_table() -> pd.DataFrame:
    rows = [*iter_group_records(), *iter_parameter_records()]
    table = pd.DataFrame(rows)
    validate_signed_logp_table(table)
    return table


def validate_signed_logp_table(table: pd.DataFrame) -> None:
    if len(table) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} signed-logP rows, got {len(table)}.")

    if (table["source_model"] != MODEL_DIR).any():
        raise ValueError(f"All source rows must use {MODEL_DIR}.")
    if (table["source_contrast"] != SOURCE_CONTRAST).any():
        raise ValueError(f"All source contrasts must be {SOURCE_CONTRAST!r}.")
    if (table["positive_direction"] != POSITIVE_DIRECTION_LABEL).any():
        raise ValueError(f"Positive direction must be {POSITIVE_DIRECTION_LABEL!r}.")

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
