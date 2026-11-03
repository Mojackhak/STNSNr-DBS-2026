from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
import math
import re
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plugin.default.viz_defaults as cfg


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
PAIN_DIR = PROJECT / "summary" / "stats" / "pain"
OUTPUT_DIR = PAIN_DIR / "corr" / "logforce_standardized_slope"
OUTPUT_PATH = OUTPUT_DIR / "pain_logforce_standardized_slope_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_SLOPE_FILE = "LogForce-Phase-Region_slope.csv"
CONNECTIVITY_SLOPE_FILE = "LogForce-Phase_slope.csv"
VALUE_LIMIT = 0.6
SD_SOURCE = "region_normalized"
LOGFORCE_PREDICTOR = "LogForce_c"
UNIT_SUFFIX_RE = re.compile(r"\s+\((?:dB|Hz|s|V|a\.u\.)\)\s*$")

BANDS = ("Delta", "Theta", "Alpha", "Beta_low", "Beta_high", "Gamma")
BAND_LABELS = {
    "Delta": "δ",
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}

PHASES = ("ONS", "SUS", "OFS", "PST")
REGIONS = ("SNr", "STN")
CONNECTIVITY_REGION = "SNr-/→STN"
PARAM_ROOT = PAIN_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL_TOP = "Params"
PARAM_METRIC_KEY = "params"
PARAM_METRIC_LABEL = "Aperiodic parameters"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    param_type: str
    base_dir: Path
    slope_file: str
    group: str
    label_top: str
    region: str


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


LOCAL_METRICS = (
    MetricSpec(
        "raw",
        "raw-mean",
        PAIN_DIR / "tfr" / "raw" / "raw-mean-scalar",
        LOCAL_SLOPE_FILE,
        "local",
        "",
        "",
    ),
    MetricSpec(
        "period",
        "period-mean",
        PAIN_DIR / "tfr" / "period" / "period-mean-scalar",
        LOCAL_SLOPE_FILE,
        "local",
        "",
        "",
    ),
    MetricSpec(
        "burst-occupation",
        "burst-occupation",
        PAIN_DIR / "burst" / "burst" / "burst-occupation-scalar",
        LOCAL_SLOPE_FILE,
        "local",
        "",
        "",
    ),
    MetricSpec(
        "burst-rate",
        "burst-rate",
        PAIN_DIR / "burst" / "burst" / "burst-rate-scalar",
        LOCAL_SLOPE_FILE,
        "local",
        "",
        "",
    ),
    MetricSpec(
        "burst-duration",
        "burst-duration",
        PAIN_DIR / "burst" / "burst" / "burst-duration-scalar",
        LOCAL_SLOPE_FILE,
        "local",
        "",
        "",
    ),
    MetricSpec(
        "burst-mean",
        "burst-mean",
        PAIN_DIR / "burst" / "burst" / "burst-mean-scalar",
        LOCAL_SLOPE_FILE,
        "local",
        "",
        "",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        "coh-mean",
        PAIN_DIR / "conn" / "coh" / "coh-mean-scalar",
        CONNECTIVITY_SLOPE_FILE,
        "connectivity",
        CONNECTIVITY_REGION,
        CONNECTIVITY_REGION,
    ),
    MetricSpec(
        "wpli",
        "wpli-mean",
        PAIN_DIR / "conn" / "wpli" / "wpli-mean-scalar",
        CONNECTIVITY_SLOPE_FILE,
        "connectivity",
        CONNECTIVITY_REGION,
        CONNECTIVITY_REGION,
    ),
    MetricSpec(
        "ciplv",
        "ciplv-mean",
        PAIN_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
        CONNECTIVITY_SLOPE_FILE,
        "connectivity",
        CONNECTIVITY_REGION,
        CONNECTIVITY_REGION,
    ),
    MetricSpec(
        "delta_net_gc",
        "delta_net_gc-mean",
        PAIN_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
        CONNECTIVITY_SLOPE_FILE,
        "connectivity",
        CONNECTIVITY_REGION,
        CONNECTIVITY_REGION,
    ),
    MetricSpec(
        "psi",
        "psi-mean",
        PAIN_DIR / "psi" / "psi" / "psi-mean-scalar",
        CONNECTIVITY_SLOPE_FILE,
        "connectivity",
        CONNECTIVITY_REGION,
        CONNECTIVITY_REGION,
    ),
)

METRICS = LOCAL_METRICS + CONNECTIVITY_METRICS

PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
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


def metric_label(param_type: str) -> str:
    label = cfg.format_label(param_type, prefix="", key="")
    label = UNIT_SUFFIX_RE.sub("", label)
    label = " ".join(label.split())
    if label.startswith("burst "):
        label = label[0].upper() + label[1:]
    return label.strip()


def sd_source_path(base_dir: Path) -> Path:
    return base_dir / f"{base_dir.name}_{SD_SOURCE}.csv"


def load_local_value_sd(metric: MetricSpec) -> pd.DataFrame:
    path = sd_source_path(metric.base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Region", "Phase", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[df["Band"].isin(BANDS) & df["Region"].isin(REGIONS) & df["Phase"].isin(PHASES)]
        .groupby(["Band", "Region", "Phase"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = SD_SOURCE
    return grouped


def load_connectivity_value_sd(metric: MetricSpec) -> pd.DataFrame:
    path = sd_source_path(metric.base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Phase", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[df["Band"].isin(BANDS) & df["Phase"].isin(PHASES)]
        .groupby(["Band", "Phase"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = SD_SOURCE
    return grouped


def load_parameter_value_sd() -> pd.DataFrame:
    path = sd_source_path(PARAM_ROOT)
    if not path.exists():
        raise FileNotFoundError(f"Missing parameter SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Region", "Phase", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    parameter_keys = [parameter.key for parameter in PARAMETERS]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[df["Band"].isin(parameter_keys) & df["Region"].isin(REGIONS) & df["Phase"].isin(PHASES)]
        .groupby(["Band", "Region", "Phase"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = SD_SOURCE
    return grouped


def load_local_slope_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing slope table: {path}")

    df = pd.read_csv(path)
    required = {"Phase", "Region", "slope", "lower", "upper", "p", "q"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["Phase"].isin(PHASES) & df["Region"].isin(REGIONS)].copy()
    for column in ["slope", "lower", "upper", "p", "q"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_connectivity_slope_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing slope table: {path}")

    df = pd.read_csv(path)
    required = {"Phase", "slope", "lower", "upper", "p", "q"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["Phase"].isin(PHASES)].copy()
    for column in ["slope", "lower", "upper", "p", "q"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_standardized_slope_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for metric in LOCAL_METRICS:
        sd_table = load_local_value_sd(metric)
        metric_label_text = metric_label(metric.param_type)

        for region in REGIONS:
            group_key = f"{region.lower()}_local"
            sd_for_region = sd_table[sd_table["Region"].eq(region)]

            for band in BANDS:
                sd_for_band = sd_for_region[sd_for_region["Band"].eq(band)].set_index("Phase")
                slope_path = metric.base_dir / band / MODEL_DIR / metric.slope_file
                slope_rows = load_local_slope_rows(slope_path)
                slope_rows = slope_rows[slope_rows["Region"].eq(region)]

                for _, slope_row in slope_rows.iterrows():
                    phase = str(slope_row["Phase"])
                    if phase not in sd_for_band.index:
                        raise ValueError(
                            f"Missing SD row for metric={metric.key}, band={band}, "
                            f"region={region}, phase={phase}"
                        )

                    sd_row = sd_for_band.loc[phase]
                    rows.append(
                        make_standardized_slope_record(
                            feature_group="frequency_metric",
                            group_key=group_key,
                            label_top=region,
                            metric_key=metric.key,
                            metric_label_text=metric_label_text,
                            band=band,
                            band_label=BAND_LABELS[band],
                            parameter="",
                            parameter_label="",
                            region=region,
                            phase=phase,
                            slope_row=slope_row,
                            sd_row=sd_row,
                            slope_path=slope_path,
                        )
                    )

    for metric in CONNECTIVITY_METRICS:
        sd_table = load_connectivity_value_sd(metric)
        metric_label_text = metric_label(metric.param_type)

        for band in BANDS:
            sd_for_band = sd_table[sd_table["Band"].eq(band)].set_index("Phase")
            slope_path = metric.base_dir / band / MODEL_DIR / metric.slope_file
            slope_rows = load_connectivity_slope_rows(slope_path)

            for _, slope_row in slope_rows.iterrows():
                phase = str(slope_row["Phase"])
                if phase not in sd_for_band.index:
                    raise ValueError(
                        f"Missing SD row for metric={metric.key}, band={band}, phase={phase}"
                    )

                sd_row = sd_for_band.loc[phase]
                rows.append(
                    make_standardized_slope_record(
                        feature_group="frequency_metric",
                        group_key="connectivity",
                        label_top=CONNECTIVITY_REGION,
                        metric_key=metric.key,
                        metric_label_text=metric_label_text,
                        band=band,
                        band_label=BAND_LABELS[band],
                        parameter="",
                        parameter_label="",
                        region=CONNECTIVITY_REGION,
                        phase=phase,
                        slope_row=slope_row,
                        sd_row=sd_row,
                        slope_path=slope_path,
                    )
                )

    parameter_sd_table = load_parameter_value_sd()
    for parameter in PARAMETERS:
        sd_for_parameter = parameter_sd_table[parameter_sd_table["Band"].eq(parameter.key)].set_index(
            ["Region", "Phase"]
        )
        slope_path = PARAM_ROOT / parameter.key / MODEL_DIR / LOCAL_SLOPE_FILE
        slope_rows = load_local_slope_rows(slope_path)

        for _, slope_row in slope_rows.iterrows():
            region = str(slope_row["Region"])
            phase = str(slope_row["Phase"])
            sd_key = (region, phase)
            if sd_key not in sd_for_parameter.index:
                raise ValueError(f"Missing SD row for parameter={parameter.key}, region={region}, phase={phase}")

            sd_row = sd_for_parameter.loc[sd_key]
            rows.append(
                make_standardized_slope_record(
                    feature_group="parameter",
                    group_key=PARAMETER_GROUP_KEY,
                    label_top=PARAMETER_LABEL_TOP,
                    metric_key=PARAM_METRIC_KEY,
                    metric_label_text=PARAM_METRIC_LABEL,
                    band=parameter.key,
                    band_label=parameter.label,
                    parameter=parameter.key,
                    parameter_label=parameter.label,
                    region=region,
                    phase=phase,
                    slope_row=slope_row,
                    sd_row=sd_row,
                    slope_path=slope_path,
                )
            )

    return pd.DataFrame(rows)


def make_standardized_slope_record(
    *,
    feature_group: str,
    group_key: str,
    label_top: str,
    metric_key: str,
    metric_label_text: str,
    band: str,
    band_label: str,
    parameter: str,
    parameter_label: str,
    region: str,
    phase: str,
    slope_row: pd.Series,
    sd_row: pd.Series,
    slope_path: Path,
) -> dict[str, object]:
    slope = parse_float(slope_row["slope"])
    lower = parse_float(slope_row["lower"])
    upper = parse_float(slope_row["upper"])
    p_value = parse_float(slope_row["p"])
    q_value = parse_float(slope_row["q"])
    eval_at = parse_float(slope_row.get("eval_at", math.nan))
    value_sd = parse_float(sd_row["value_sd"])
    value_n = int(sd_row["value_n"])

    if not math.isfinite(value_sd) or value_sd <= 0:
        standardized_slope = math.nan
        standardized_lower = math.nan
        standardized_upper = math.nan
        standardized_display = math.nan
    else:
        standardized_slope = slope / value_sd if math.isfinite(slope) else math.nan
        standardized_lower = lower / value_sd if math.isfinite(lower) else math.nan
        standardized_upper = upper / value_sd if math.isfinite(upper) else math.nan
        standardized_display = (
            float(np.clip(standardized_slope, -VALUE_LIMIT, VALUE_LIMIT))
            if math.isfinite(standardized_slope)
            else math.nan
        )

    return {
        "feature_group": feature_group,
        "group": group_key,
        "label_top": label_top,
        "metric": metric_key,
        "metric_label": metric_label_text,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "region": region,
        "phase": phase,
        "predictor": LOGFORCE_PREDICTOR,
        "eval_at": eval_at,
        "slope": slope,
        "lower": lower,
        "upper": upper,
        "p": p_value,
        "q": q_value,
        "value_sd": value_sd,
        "value_n": value_n,
        "standardized_slope": standardized_slope,
        "standardized_lower": standardized_lower,
        "standardized_upper": standardized_upper,
        "standardized_slope_display": standardized_display,
        "star": p_to_star(q_value),
        "source_slope_csv": str(slope_path),
        "source_sd_csv": str(sd_row["source_sd_csv"]),
        "sd_source": str(sd_row["sd_source"]),
    }


def validate_standardized_slope_table(df: pd.DataFrame) -> None:
    expected_local_rows = len(LOCAL_METRICS) * len(BANDS) * len(REGIONS) * len(PHASES)
    expected_connectivity_rows = len(CONNECTIVITY_METRICS) * len(BANDS) * len(PHASES)
    expected_parameter_rows = len(PARAMETERS) * len(REGIONS) * len(PHASES)
    expected_rows = expected_local_rows + expected_connectivity_rows + expected_parameter_rows
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")

    expected_groups = {"snr_local", "stn_local", "connectivity", PARAMETER_GROUP_KEY}
    if set(df["group"].dropna()) != expected_groups:
        raise ValueError(f"Unexpected group values: {sorted(df['group'].dropna().unique())}")

    if set(df["phase"].dropna()) != set(PHASES):
        raise ValueError(f"Unexpected phase values: {sorted(df['phase'].dropna().unique())}")

    if df["value_sd"].isna().any() or (df["value_sd"] <= 0).any():
        bad = df[df["value_sd"].isna() | (df["value_sd"] <= 0)]
        raise ValueError(f"Found missing or non-positive value_sd rows: {bad.head().to_dict('records')}")

    if df["standardized_slope"].isna().any():
        bad = df[df["standardized_slope"].isna()]
        raise ValueError(f"Found missing standardized_slope rows: {bad.head().to_dict('records')}")

    non_rlmer = df[~df["source_slope_csv"].str.contains(f"/{MODEL_DIR}/", regex=False)]
    if not non_rlmer.empty:
        raise ValueError(f"Found non-{MODEL_DIR} slope rows: {non_rlmer.head().to_dict('records')}")

    bad_sd_source = df[~df["sd_source"].eq(SD_SOURCE)]
    if not bad_sd_source.empty:
        raise ValueError(f"All rows must use {SD_SOURCE}: {bad_sd_source.head().to_dict('records')}")

    bad_sd_path = df[~df["source_sd_csv"].str.endswith(f"_{SD_SOURCE}.csv")]
    if not bad_sd_path.empty:
        raise ValueError(f"Found unexpected SD source paths: {bad_sd_path.head().to_dict('records')}")


def run_workflow() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_standardized_slope_table()
    validate_standardized_slope_table(table)
    table.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = run_workflow()
    print(f"standardized LogForce slope table: saved CSV: {output_path}")
