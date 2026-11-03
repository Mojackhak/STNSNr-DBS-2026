from __future__ import annotations

import os

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
OUTPUT_DIR = PAIN_EVOKED_DIR / "corr" / "logforce_slope"
OUTPUT_PATH = OUTPUT_DIR / "pain_evoked_logforce_slope_values.csv"

MODEL_DIR = "rlmer_model"
LOCAL_SLOPE_FILE = "LogForce-Phase-Region_slope.csv"
CONNECTIVITY_SLOPE_FILE = "LogForce-Phase_slope.csv"
LOGFORCE_PREDICTOR = "LogForce_c"
VALUE_LIMIT = 1.3

PHASES = ("ONS", "OFS")
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
    bands: tuple[str, ...]
    slope_file: str
    region_specific: bool


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


FEATURES = (
    FeatureSpec(
        key="period",
        feature_group="frequency_metric",
        path_parts=("tfr", "period"),
        folder_prefix="period-mean",
        metric_label="Periodic power",
        bands=("Delta", "Alpha", "Beta_low", "Gamma"),
        slope_file=LOCAL_SLOPE_FILE,
        region_specific=True,
    ),
    FeatureSpec(
        key="raw",
        feature_group="frequency_metric",
        path_parts=("tfr", "raw"),
        folder_prefix="raw-mean",
        metric_label="Total power",
        bands=("Delta", "Theta"),
        slope_file=LOCAL_SLOPE_FILE,
        region_specific=True,
    ),
    FeatureSpec(
        key="coh",
        feature_group="connectivity_metric",
        path_parts=("conn", "coh"),
        folder_prefix="coh-mean",
        metric_label="Coherence",
        bands=("Delta",),
        slope_file=CONNECTIVITY_SLOPE_FILE,
        region_specific=False,
    ),
    FeatureSpec(
        key="wpli",
        feature_group="connectivity_metric",
        path_parts=("conn", "wpli"),
        folder_prefix="wpli-mean",
        metric_label="wPLI",
        bands=("Delta",),
        slope_file=CONNECTIVITY_SLOPE_FILE,
        region_specific=False,
    ),
)

PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)


def parse_float(value: object) -> float:
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


def evoked_folder(spec: FeatureSpec, evoked_key: str) -> Path:
    return PAIN_EVOKED_DIR.joinpath(*spec.path_parts) / f"{spec.folder_prefix}-{evoked_key}"


def load_slope_rows(path: Path, *, region_specific: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing slope table: {path}")

    df = pd.read_csv(path)
    required = {"Phase", "slope", "lower", "upper", "p", "q"}
    if region_specific:
        required.add("Region")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["Phase"].isin(PHASES)].copy()
    if region_specific:
        df = df[df["Region"].isin(REGIONS)].copy()
    for column in ["slope", "lower", "upper", "p", "q"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def make_slope_record(
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
    phase: str,
    slope_row: pd.Series,
    slope_path: Path,
) -> dict[str, object]:
    slope = parse_float(slope_row["slope"])
    lower = parse_float(slope_row["lower"])
    upper = parse_float(slope_row["upper"])
    p_value = parse_float(slope_row["p"])
    q_value = parse_float(slope_row["q"])

    return {
        "feature_group": feature_group,
        "metric": metric,
        "metric_label": metric_label,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "evoked_metric": evoked_metric,
        "evoked_label": evoked_label,
        "region": region,
        "phase": phase,
        "predictor": LOGFORCE_PREDICTOR,
        "slope": slope,
        "lower": lower,
        "upper": upper,
        "p": p_value,
        "q": q_value,
        "slope_display": float(np.clip(slope, -VALUE_LIMIT, VALUE_LIMIT)) if math.isfinite(slope) else math.nan,
        "star": p_to_star(q_value),
        "source_slope_csv": str(slope_path),
    }


def build_logforce_slope_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for evoked_metric, evoked_label in EVOKED_METRICS:
        for spec in FEATURES:
            base_dir = evoked_folder(spec, evoked_metric)
            for band in spec.bands:
                slope_path = base_dir / band / MODEL_DIR / spec.slope_file
                slope_rows = load_slope_rows(slope_path, region_specific=spec.region_specific)
                for _, slope_row in slope_rows.iterrows():
                    region = str(slope_row["Region"]) if spec.region_specific else CONNECTIVITY_REGION
                    rows.append(
                        make_slope_record(
                            feature_group=spec.feature_group,
                            metric=spec.key,
                            metric_label=spec.metric_label,
                            band=band,
                            band_label=BAND_LABELS[band],
                            parameter="",
                            parameter_label="",
                            evoked_metric=evoked_metric,
                            evoked_label=evoked_label,
                            region=region,
                            phase=str(slope_row["Phase"]),
                            slope_row=slope_row,
                            slope_path=slope_path,
                        )
                    )

        param_base = PAIN_EVOKED_DIR / "params" / "params" / f"params-mean-{evoked_metric}"
        for parameter in PARAMETERS:
            slope_path = param_base / parameter.key / MODEL_DIR / LOCAL_SLOPE_FILE
            slope_rows = load_slope_rows(slope_path, region_specific=True)
            for _, slope_row in slope_rows.iterrows():
                rows.append(
                    make_slope_record(
                        feature_group="parameter",
                        metric="params",
                        metric_label="Aperiodic parameters",
                        band=parameter.key,
                        band_label=parameter.label,
                        parameter=parameter.key,
                        parameter_label=parameter.label,
                        evoked_metric=evoked_metric,
                        evoked_label=evoked_label,
                        region=str(slope_row["Region"]),
                        phase=str(slope_row["Phase"]),
                        slope_row=slope_row,
                        slope_path=slope_path,
                    )
                )

    return pd.DataFrame(rows)


def validate_logforce_slope_table(df: pd.DataFrame) -> None:
    expected_local_rows = 8 * (4 + 2) * len(REGIONS) * len(PHASES)
    expected_connectivity_rows = 8 * 2 * len(PHASES)
    expected_parameter_rows = 8 * len(PARAMETERS) * len(REGIONS) * len(PHASES)
    expected_rows = expected_local_rows + expected_connectivity_rows + expected_parameter_rows
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}.")

    if set(df["phase"].dropna()) != set(PHASES):
        raise ValueError(f"Unexpected phase values: {sorted(df['phase'].dropna().unique())}")

    for column in ["slope", "lower", "upper", "p", "q", "slope_display"]:
        invalid_count = int(pd.to_numeric(df[column], errors="coerce").isna().sum())
        if invalid_count:
            raise ValueError(f"Column {column!r} contains {invalid_count} invalid values.")

    non_rlmer = df[~df["source_slope_csv"].astype(str).str.contains(f"/{MODEL_DIR}/", regex=False)]
    if not non_rlmer.empty:
        raise ValueError(f"Found non-{MODEL_DIR} source rows: {non_rlmer.head().to_dict('records')}")


def run_workflow() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_logforce_slope_table()
    validate_logforce_slope_table(table)
    table.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = run_workflow()
    print(f"pain-evoked LogForce slope table: saved CSV: {output_path}")
