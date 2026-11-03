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
WALK_DIR = PROJECT / "summary" / "stats" / "walk"
OUTPUT_DIR = WALK_DIR / "corr" / "mni_standardized_slope"
OUTPUT_PATH = OUTPUT_DIR / "walk_mni_standardized_slope_values.csv"

MODEL_DIR = "rlmer_model"
VALUE_LIMIT = 1.0
UNIT_SUFFIX_RE = re.compile(r"\s+\((?:dB|Hz|s|V|a\.u\.)\)\s*$")

BANDS = ["Delta", "Theta", "Alpha", "Beta_low", "Beta_high", "Gamma"]
BAND_LABELS = {
    "Delta": "δ",
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}

POSES = ("Stand", "Walk", "Turn")
REGIONS = ("STN", "SNr")
LOCAL_SD_SOURCE = "contact_normalized"
CONNECTIVITY_SD_SOURCE = "contact_normalized_split"
PARAM_SD_SOURCE = "contact_normalized"
PARAM_ROOT = WALK_DIR / "params" / "params" / "params-mean-scalar"
PARAM_METRIC_KEY = "params"
PARAM_METRIC_LABEL = "Aperiodic parameters"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    param_type: str
    base_dir: Path
    sd_source: str


@dataclass(frozen=True)
class CoordinateSpec:
    key: str
    predictor: str
    label: str
    slope_file: str


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


LOCAL_METRICS = (
    MetricSpec("raw", "raw-mean", WALK_DIR / "tfr" / "raw" / "raw-mean-scalar", LOCAL_SD_SOURCE),
    MetricSpec("period", "period-mean", WALK_DIR / "tfr" / "period" / "period-mean-scalar", LOCAL_SD_SOURCE),
    MetricSpec(
        "burst-occupation",
        "burst-occupation",
        WALK_DIR / "burst" / "burst" / "burst-occupation-scalar",
        LOCAL_SD_SOURCE,
    ),
    MetricSpec(
        "burst-rate",
        "burst-rate",
        WALK_DIR / "burst" / "burst" / "burst-rate-scalar",
        LOCAL_SD_SOURCE,
    ),
    MetricSpec(
        "burst-duration",
        "burst-duration",
        WALK_DIR / "burst" / "burst" / "burst-duration-scalar",
        LOCAL_SD_SOURCE,
    ),
    MetricSpec(
        "burst-mean",
        "burst-mean",
        WALK_DIR / "burst" / "burst" / "burst-mean-scalar",
        LOCAL_SD_SOURCE,
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec("coh", "coh-mean", WALK_DIR / "conn" / "coh" / "coh-mean-scalar", CONNECTIVITY_SD_SOURCE),
    MetricSpec("wpli", "wpli-mean", WALK_DIR / "conn" / "wpli" / "wpli-mean-scalar", CONNECTIVITY_SD_SOURCE),
    MetricSpec("ciplv", "ciplv-mean", WALK_DIR / "conn" / "ciplv" / "ciplv-mean-scalar", CONNECTIVITY_SD_SOURCE),
    MetricSpec(
        "delta_net_gc",
        "delta_net_gc-mean",
        WALK_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
        CONNECTIVITY_SD_SOURCE,
    ),
    MetricSpec("psi", "psi-mean", WALK_DIR / "psi" / "psi" / "psi-mean-scalar", CONNECTIVITY_SD_SOURCE),
)

METRICS = LOCAL_METRICS + CONNECTIVITY_METRICS

COORDINATES = (
    CoordinateSpec("x", "zMX", "MNI-x (scored)", "zMX-Pose-Region_slope.csv"),
    CoordinateSpec("y", "zMY", "MNI-y (scored)", "zMY-Pose-Region_slope.csv"),
    CoordinateSpec("z", "zMZ", "MNI-z (scored)", "zMZ-Pose-Region_slope.csv"),
)

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


def sd_source_path(metric: MetricSpec) -> Path:
    return metric.base_dir / f"{metric.base_dir.name}_{metric.sd_source}.csv"


def filter_single_task(df: pd.DataFrame) -> pd.DataFrame:
    if "Task" not in df.columns:
        return df
    return df[df["Task"].astype(str).eq("Single")].copy()


def load_value_sd(metric: MetricSpec) -> pd.DataFrame:
    path = sd_source_path(metric)
    if not path.exists():
        raise FileNotFoundError(f"Missing SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Region", "Pose", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = filter_single_task(df)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[df["Band"].isin(BANDS) & df["Region"].isin(REGIONS) & df["Pose"].isin(POSES)]
        .groupby(["Band", "Region", "Pose"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = metric.sd_source
    return grouped


def load_parameter_value_sd() -> pd.DataFrame:
    path = PARAM_ROOT / f"{PARAM_ROOT.name}_{PARAM_SD_SOURCE}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing parameter SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Region", "Pose", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    parameter_keys = [parameter.key for parameter in PARAMETERS]
    df = filter_single_task(df)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[df["Band"].isin(parameter_keys) & df["Region"].isin(REGIONS) & df["Pose"].isin(POSES)]
        .groupby(["Band", "Region", "Pose"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = PARAM_SD_SOURCE
    return grouped


def load_slope_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing slope table: {path}")

    df = pd.read_csv(path)
    required = {"Pose", "Region", "slope", "lower", "upper", "p", "q"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["Pose"].isin(POSES) & df["Region"].isin(REGIONS)].copy()
    for column in ["slope", "lower", "upper", "p", "q"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_standardized_slope_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for metric in METRICS:
        sd_table = load_value_sd(metric)
        metric_label_text = metric_label(metric.param_type)

        for band in BANDS:
            band_label = BAND_LABELS[band]
            sd_for_band = sd_table[sd_table["Band"].eq(band)].set_index(["Region", "Pose"])

            for coordinate in COORDINATES:
                slope_path = metric.base_dir / band / MODEL_DIR / coordinate.slope_file
                slope_rows = load_slope_rows(slope_path)

                for _, slope_row in slope_rows.iterrows():
                    region = str(slope_row["Region"])
                    pose = str(slope_row["Pose"])
                    sd_key = (region, pose)
                    if sd_key not in sd_for_band.index:
                        raise ValueError(
                            f"Missing SD row for metric={metric.key}, band={band}, "
                            f"region={region}, pose={pose}"
                        )

                    sd_row = sd_for_band.loc[sd_key]
                    rows.append(
                        make_standardized_slope_record(
                            feature_group="frequency_metric",
                            metric_key=metric.key,
                            metric_label_text=metric_label_text,
                            band=band,
                            band_label=band_label,
                            parameter="",
                            parameter_label="",
                            region=region,
                            pose=pose,
                            coordinate=coordinate,
                            slope_row=slope_row,
                            sd_row=sd_row,
                            slope_path=slope_path,
                        )
                    )

    parameter_sd_table = load_parameter_value_sd()
    for parameter in PARAMETERS:
        sd_for_parameter = parameter_sd_table[parameter_sd_table["Band"].eq(parameter.key)].set_index(
            ["Region", "Pose"]
        )

        for coordinate in COORDINATES:
            slope_path = PARAM_ROOT / parameter.key / MODEL_DIR / coordinate.slope_file
            slope_rows = load_slope_rows(slope_path)

            for _, slope_row in slope_rows.iterrows():
                region = str(slope_row["Region"])
                pose = str(slope_row["Pose"])
                sd_key = (region, pose)
                if sd_key not in sd_for_parameter.index:
                    raise ValueError(
                        f"Missing SD row for parameter={parameter.key}, region={region}, pose={pose}"
                    )

                sd_row = sd_for_parameter.loc[sd_key]
                rows.append(
                    make_standardized_slope_record(
                        feature_group="parameter",
                        metric_key=PARAM_METRIC_KEY,
                        metric_label_text=PARAM_METRIC_LABEL,
                        band=parameter.key,
                        band_label=parameter.label,
                        parameter=parameter.key,
                        parameter_label=parameter.label,
                        region=region,
                        pose=pose,
                        coordinate=coordinate,
                        slope_row=slope_row,
                        sd_row=sd_row,
                        slope_path=slope_path,
                    )
                )

    return pd.DataFrame(rows)


def make_standardized_slope_record(
    *,
    feature_group: str,
    metric_key: str,
    metric_label_text: str,
    band: str,
    band_label: str,
    parameter: str,
    parameter_label: str,
    region: str,
    pose: str,
    coordinate: CoordinateSpec,
    slope_row: pd.Series,
    sd_row: pd.Series,
    slope_path: Path,
) -> dict[str, object]:
    slope = parse_float(slope_row["slope"])
    lower = parse_float(slope_row["lower"])
    upper = parse_float(slope_row["upper"])
    p_value = parse_float(slope_row["p"])
    q_value = parse_float(slope_row["q"])
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
        "metric": metric_key,
        "metric_label": metric_label_text,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "region": region,
        "pose": pose,
        "coordinate": coordinate.key,
        "coordinate_label": coordinate.label,
        "predictor": coordinate.predictor,
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
    expected_frequency_rows = len(METRICS) * len(BANDS) * len(COORDINATES) * len(REGIONS) * len(POSES)
    expected_parameter_rows = len(PARAMETERS) * len(COORDINATES) * len(REGIONS) * len(POSES)
    expected_rows = expected_frequency_rows + expected_parameter_rows
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")

    if set(df["feature_group"].dropna()) != {"frequency_metric", "parameter"}:
        raise ValueError(f"Unexpected feature_group values: {sorted(df['feature_group'].dropna().unique())}")

    if set(df["pose"].dropna()) != set(POSES):
        raise ValueError(f"Unexpected pose values: {sorted(df['pose'].dropna().unique())}")

    if df["value_sd"].isna().any() or (df["value_sd"] <= 0).any():
        bad = df[df["value_sd"].isna() | (df["value_sd"] <= 0)]
        raise ValueError(f"Found missing or non-positive value_sd rows: {bad.head().to_dict('records')}")

    if df["standardized_slope"].isna().any():
        bad = df[df["standardized_slope"].isna()]
        raise ValueError(f"Found missing standardized_slope rows: {bad.head().to_dict('records')}")

    non_rlmer = df[~df["source_slope_csv"].str.contains(f"/{MODEL_DIR}/", regex=False)]
    if not non_rlmer.empty:
        raise ValueError(f"Found non-{MODEL_DIR} slope rows: {non_rlmer.head().to_dict('records')}")

    local_bad = df[df["metric"].isin([metric.key for metric in LOCAL_METRICS]) & ~df["sd_source"].eq(LOCAL_SD_SOURCE)]
    if not local_bad.empty:
        raise ValueError(
            f"Local metrics must use {LOCAL_SD_SOURCE} SD sources: {local_bad.head().to_dict('records')}"
        )

    conn_bad = df[
        df["metric"].isin([metric.key for metric in CONNECTIVITY_METRICS])
        & ~df["sd_source"].eq(CONNECTIVITY_SD_SOURCE)
    ]
    if not conn_bad.empty:
        raise ValueError(
            f"Connectivity/gc/psi metrics must use {CONNECTIVITY_SD_SOURCE} SD sources: "
            f"{conn_bad.head().to_dict('records')}"
        )

    parameter_bad = df[df["feature_group"].eq("parameter") & ~df["sd_source"].eq(PARAM_SD_SOURCE)]
    if not parameter_bad.empty:
        raise ValueError(
            f"Aperiodic parameters must use {PARAM_SD_SOURCE} SD sources: "
            f"{parameter_bad.head().to_dict('records')}"
        )


def run_workflow() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_standardized_slope_table()
    validate_standardized_slope_table(table)
    table.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = run_workflow()
    print(f"standardized slope table: saved CSV: {output_path}")
