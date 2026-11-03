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
MED_DIR = PROJECT / "summary" / "stats" / "med"
OUTPUT_DIR = MED_DIR / "corr"
SIGNED_LOGP_OUTPUT_PATH = OUTPUT_DIR / "med_medon_vs_medoff_signed_logp_values.csv"

MODEL_DIR = "rlmer_model"
SOURCE_CONTRAST = "Off - On"
COMPARISON_TEXT = "MedON vs MedOFF"
POSITIVE_DIRECTION_LABEL = "ON > OFF"
NEGATIVE_DIRECTION_LABEL = "ON < OFF"
VALUE_LIMIT = 3.0

BANDS = ["Delta", "Theta", "Alpha", "Beta_low", "Beta_high", "Gamma"]
BAND_LABELS = {
    "Delta": "δ",
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}
ASCII_BAND_LABELS = {
    "Delta": "delta",
    "Theta": "theta",
    "Alpha": "alpha",
    "Beta_low": "Low-beta",
    "Beta_high": "High-beta",
    "Gamma": "gamma",
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
    MetricSpec("raw", "Total power", MED_DIR / "tfr" / "raw" / "raw-mean-scalar"),
    MetricSpec("period", "Periodic power", MED_DIR / "tfr" / "period" / "period-mean-scalar"),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        MED_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-rate",
        r"$\operatorname{arcsinh}$ burst rate",
        MED_DIR / "burst" / "burst" / "burst-rate-scalar",
    ),
    MetricSpec(
        "burst-duration",
        r"$\operatorname{ln}$ burst duration",
        MED_DIR / "burst" / "burst" / "burst-duration-scalar",
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        MED_DIR / "burst" / "burst" / "burst-mean-scalar",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        MED_DIR / "conn" / "coh" / "coh-mean-scalar",
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        MED_DIR / "conn" / "wpli" / "wpli-mean-scalar",
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        MED_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        MED_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "PSI", MED_DIR / "psi" / "psi" / "psi-mean-scalar"),
)

GROUPS = (
    GroupSpec(
        key="stn_local",
        label_top="STN",
        metrics=LOCAL_METRICS,
        tukey_file="Med-Region_tukey.csv",
        region="STN",
    ),
    GroupSpec(
        key="snr_local",
        label_top="SNr",
        metrics=LOCAL_METRICS,
        tukey_file="Med-Region_tukey.csv",
        region="SNr",
    ),
    GroupSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        metrics=CONNECTIVITY_METRICS,
        tukey_file="Med_tukey.csv",
        region=None,
    ),
)

PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)
PARAMETER_ROOT = MED_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL_TOP = "Params"
PARAMETER_TUKEY_FILE = "Med-Region_tukey.csv"

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
    return PARAMETER_ROOT / parameter.key / MODEL_DIR / PARAMETER_TUKEY_FILE


def load_off_vs_on_row(path: Path, *, region: str | None) -> dict[str, str]:
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
    estimate_on_minus_off = -source_estimate if math.isfinite(source_estimate) else math.nan
    p_value = parse_float(row.get("p_tukey"))
    value = signed_logp(estimate_on_minus_off, p_value)
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
        "comparison": COMPARISON_TEXT,
        "positive_direction": POSITIVE_DIRECTION_LABEL,
        "negative_direction": NEGATIVE_DIRECTION_LABEL,
        "source_model": MODEL_DIR,
        "source_contrast": clean_value(row.get("contrast")),
        "source_estimate_off_minus_on": source_estimate,
        "estimate_on_minus_off": estimate_on_minus_off,
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
            row = load_off_vs_on_row(path, region=group.region)
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
            row = load_off_vs_on_row(path, region=region)
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
        "source_model",
        "source_contrast",
        "source_estimate_off_minus_on",
        "estimate_on_minus_off",
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

    bad_contrasts = table.loc[table["source_contrast"] != SOURCE_CONTRAST, "source_csv"].tolist()
    if bad_contrasts:
        raise ValueError(f"Found non-{SOURCE_CONTRAST!r} source rows: {bad_contrasts[:5]}")

    bad_paths = [
        source_csv
        for source_csv in table["source_csv"].astype(str)
        if f"/{MODEL_DIR}/" not in source_csv
    ]
    if bad_paths:
        raise ValueError(f"Found source paths outside {MODEL_DIR}: {bad_paths[:5]}")

    numeric_columns = [
        "source_estimate_off_minus_on",
        "estimate_on_minus_off",
        "p_tukey",
        "signed_logp",
        "signed_logp_display",
    ]
    for column in numeric_columns:
        invalid_count = int(pd.to_numeric(table[column], errors="coerce").isna().sum())
        if invalid_count:
            raise ValueError(f"Column {column!r} contains {invalid_count} missing values.")

    duplicate_columns = ["group", "region", "metric", "band", "parameter"]
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
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} output rows; got {len(table)}.")
    validate_signed_logp_table(table)


def iter_expected_source_paths() -> Iterable[Path]:
    for group in GROUPS:
        for metric in group.metrics:
            for band in BANDS:
                yield source_path(metric, band, group.tukey_file)
    for parameter in PARAMETERS:
        yield parameter_source_path(parameter)


if __name__ == "__main__":
    workflow_table = run_workflow()
    validate_output_file(SIGNED_LOGP_OUTPUT_PATH)
    print(f"Saved signed-logP table: {SIGNED_LOGP_OUTPUT_PATH}")
    print(f"Rows: {len(workflow_table)}")
