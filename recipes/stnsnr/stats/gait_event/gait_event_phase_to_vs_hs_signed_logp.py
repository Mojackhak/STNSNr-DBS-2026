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
GAIT_EVENT_DIR = PROJECT / "summary" / "stats" / "gait_event"
OUTPUT_DIR = GAIT_EVENT_DIR / "corr" / "phase_to_vs_hs"
SIGNED_LOGP_OUTPUT_PATH = OUTPUT_DIR / "gait_event_phase_to_vs_hs_signed_logp_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_TUKEY_FILE = "Phase-Region-Lat_tukey.csv"
CONNECTIVITY_TUKEY_FILE = "Phase-Lat_tukey.csv"
SOURCE_CONTRAST = "HS - TO"
COMPARISON_TEXT = "TO vs HS"
POSITIVE_DIRECTION_LABEL = "TO > HS"
NEGATIVE_DIRECTION_LABEL = "TO < HS"
SOURCE_PHASE_FIRST = "HS"
SOURCE_PHASE_SECOND = "TO"
PLOT_PHASE_FIRST = "TO"
PLOT_PHASE_SECOND = "HS"
SOURCE_ESTIMATE_COLUMN = "source_estimate_hs_minus_to"
SIGNED_ESTIMATE_COLUMN = "signed_estimate_to_minus_hs"
SIGNED_ESTIMATE_LABEL = "TO - HS"
VALUE_LIMIT = 3.0

BANDS = ("Theta", "Alpha", "Beta_low", "Beta_high", "Gamma")
BAND_LABELS = {
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}
LAT_ORDER = ("Ipsi", "Contra")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    base_dir: Path


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label: str
    metrics: tuple[MetricSpec, ...]
    tukey_file: str
    region: str | None


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


LOCAL_METRICS = (
    MetricSpec("raw", "Total power", GAIT_EVENT_DIR / "tfr" / "raw" / "raw-mean-scalar"),
    MetricSpec("period", "Periodic power", GAIT_EVENT_DIR / "tfr" / "period" / "period-mean-scalar"),
    MetricSpec(
        "aperiod",
        "Aperiodic power",
        GAIT_EVENT_DIR / "tfr" / "aperiod" / "aperiod-mean-scalar",
    ),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        GAIT_EVENT_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        GAIT_EVENT_DIR / "burst" / "burst" / "burst-mean-scalar",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        GAIT_EVENT_DIR / "conn" / "coh" / "coh-mean-scalar",
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        GAIT_EVENT_DIR / "conn" / "wpli" / "wpli-mean-scalar",
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        GAIT_EVENT_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        GAIT_EVENT_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "PSI", GAIT_EVENT_DIR / "psi" / "psi" / "psi-mean-scalar"),
)

GROUPS = (
    GroupSpec(
        key="stn_local",
        label="STN",
        metrics=LOCAL_METRICS,
        tukey_file=LOCAL_TUKEY_FILE,
        region="STN",
    ),
    GroupSpec(
        key="snr_local",
        label="SNr",
        metrics=LOCAL_METRICS,
        tukey_file=LOCAL_TUKEY_FILE,
        region="SNr",
    ),
    GroupSpec(
        key="connectivity",
        label="SNr-/→STN",
        metrics=CONNECTIVITY_METRICS,
        tukey_file=CONNECTIVITY_TUKEY_FILE,
        region=None,
    ),
)

SUMMARY_GROUP_ORDER = ("snr_local", "stn_local", "connectivity")

PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)
PARAMETER_ROOT = GAIT_EVENT_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL = "Params"

EXPECTED_ROW_COUNT = (
    sum(len(group.metrics) * len(BANDS) * len(LAT_ORDER) for group in GROUPS)
    + len(PARAMETERS) * len(PARAMETER_REGIONS) * len(LAT_ORDER)
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


def load_phase_row(path: Path, *, lat: str, region: str | None) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Tukey CSV: {path}")

    matches: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if clean_value(row.get("contrast")) != SOURCE_CONTRAST:
                continue
            if clean_value(row.get("Lat")) != lat:
                continue
            if region is not None and clean_value(row.get("Region")) != region:
                continue
            matches.append(row)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {SOURCE_CONTRAST!r} row in {path} "
            f"for lat={lat!r}, region={region!r}; got {len(matches)}."
        )
    return matches[0]


def make_signed_logp_record(
    *,
    group_key: str,
    group_label: str,
    metric_key: str,
    metric_label: str,
    band: str,
    band_label: str,
    parameter: str,
    parameter_label: str,
    region: str,
    lat: str,
    source_csv: Path,
    row: dict[str, str],
) -> dict[str, object]:
    source_estimate = parse_float(row.get("estimate"))
    signed_estimate = -source_estimate if math.isfinite(source_estimate) else math.nan
    p_value = parse_float(row.get("p_tukey"))
    value = signed_logp(signed_estimate, p_value)
    display_value = float(np.clip(value, -VALUE_LIMIT, VALUE_LIMIT)) if math.isfinite(value) else math.nan
    star = p_to_star(p_value)

    return {
        "source_csv": str(source_csv),
        "group": group_key,
        "group_label": group_label,
        "metric": metric_key,
        "metric_label": metric_label,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "region": region,
        "lat": lat,
        "source_model": MODEL_DIR,
        "source_contrast": SOURCE_CONTRAST,
        "comparison": COMPARISON_TEXT,
        "positive_direction": POSITIVE_DIRECTION_LABEL,
        "negative_direction": NEGATIVE_DIRECTION_LABEL,
        "source_phase_first": SOURCE_PHASE_FIRST,
        "source_phase_second": SOURCE_PHASE_SECOND,
        "plot_phase_first": PLOT_PHASE_FIRST,
        "plot_phase_second": PLOT_PHASE_SECOND,
        SOURCE_ESTIMATE_COLUMN: source_estimate,
        SIGNED_ESTIMATE_COLUMN: signed_estimate,
        "estimate_label": SIGNED_ESTIMATE_LABEL,
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
                for lat in LAT_ORDER:
                    row = load_phase_row(path, lat=lat, region=group.region)
                    yield make_signed_logp_record(
                        group_key=group.key,
                        group_label=group.label,
                        metric_key=metric.key,
                        metric_label=metric.label,
                        band=band,
                        band_label=BAND_LABELS[band],
                        parameter="",
                        parameter_label="",
                        region=group.region or "",
                        lat=lat,
                        source_csv=path,
                        row=row,
                    )


def iter_parameter_records() -> Iterable[dict[str, object]]:
    for parameter in PARAMETERS:
        path = parameter_source_path(parameter)
        for region in PARAMETER_REGIONS:
            for lat in LAT_ORDER:
                row = load_phase_row(path, lat=lat, region=region)
                yield make_signed_logp_record(
                    group_key=PARAMETER_GROUP_KEY,
                    group_label=PARAMETER_LABEL,
                    metric_key="params",
                    metric_label=region,
                    band="",
                    band_label="",
                    parameter=parameter.key,
                    parameter_label=parameter.label,
                    region=region,
                    lat=lat,
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
    if "Delta" in set(table["band"].astype(str)):
        raise ValueError("Delta band must be excluded from this workflow.")
    if set(table["lat"].unique()) != set(LAT_ORDER):
        raise ValueError(f"Lat values must match {list(LAT_ORDER)}.")

    source_estimate = pd.to_numeric(table[SOURCE_ESTIMATE_COLUMN], errors="coerce")
    signed_estimate = pd.to_numeric(table[SIGNED_ESTIMATE_COLUMN], errors="coerce")
    if not np.allclose(signed_estimate, -source_estimate, equal_nan=True):
        raise ValueError(f"{SIGNED_ESTIMATE_COLUMN} must equal -{SOURCE_ESTIMATE_COLUMN}.")

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
