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

from recipes.stnsnr.stats.clinic.subject_effect_tables import (
    PHASE_3M,
    PROTOCOL_STN,
    SCALE_LABELS,
    delta_effect_column,
    total_effect_column,
)
from recipes.stnsnr.stats.mer.spike_scale_correlation import Y_COLUMNS as MER_SPIKE_SCALE_Y_COLUMNS


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
SCALE_DIR = PROJECT / "summary" / "stats" / "clinic" / "coords" / "scale"
OUTPUT_PATH = SCALE_DIR / "scale_3m_efficacy_standardized_slope_values.csv"

MODEL_DIR = "lm_model"
VALUE_LIMIT = 1.0

PAIR_COL = "ID"
REGION_COL = "Region"
SCALE_COL = "Scale"
VALUE_COL = "Value"

MER_TO_EFFECT_SCALE = {
    "MDS-UPDRS I": "MDS-UPDRS I",
    "MDS-UPDRS II": "MDS-UDPRS II",
    "MDS-UPDRS III (Med-OFF)": "MDS-UPDRS III score",
    "MDS-UPDRS III axial (Med-OFF)": "MDS-UPDRS III axial score",
    "MDS-UPDRS III rigidity (Med-OFF)": "Rigidity subscore (Med OFF)",
    "MDS-UPDRS III bradykinesia (Med-OFF)": "Bradykinesia subscore (Med OFF)",
    "MDS-UPDRS III rest tremor (Med-OFF)": "Tremor subscore (Med OFF)",
    "MDS-UPDRS III action tremor (Med-OFF)": "Tremor subscore (Med OFF)",
    "MDS-UPDRS IV": "MDS-UPDRS IV",
    "FOG-Q": "FOGQ score",
    "BDI": "BDI score",
    "MADRS": "MADRS score",
    "HAMD": "HAMD",
    "HAMA": "HAMA",
    "MAES": "MAES",
    "ESS": "ESS",
    "RBDSQ": "RBDSQ",
    "RBDSQ-HK": "RBDSQ-HK",
    "SCOPA-AUT": "SCOPA-AUT",
    "SDQ": "SDQ",
    "EAT-10": "EAT-10",
    "DSFS": "DSFS",
    "CCCS": "CCCS",
    "FSS": "FSS",
    "KPPS": "KPPS score",
    "ODQ": "ODQ",
    "ADL": "ADL",
    "SE-ADL": "SE-ADL score (%)",
    "PDQ-39": "PDQ39 score",
}


def mer_ordered_effect_scales() -> tuple[str, ...]:
    scales: list[str] = []
    seen: set[str] = set()
    for mer_label in MER_SPIKE_SCALE_Y_COLUMNS:
        scale = MER_TO_EFFECT_SCALE.get(mer_label)
        if scale is None or scale in seen:
            continue
        scales.append(scale)
        seen.add(scale)

    missing_scales = [scale for scale in SCALE_LABELS if scale not in seen]
    if missing_scales:
        raise ValueError(f"MER-referenced scale order is missing effect scales: {missing_scales}")
    if len(scales) != len(SCALE_LABELS):
        raise ValueError(f"Expected {len(SCALE_LABELS)} MER-ordered scales, got {len(scales)}.")
    return tuple(scales)


MER_ORDERED_EFFECT_SCALES = mer_ordered_effect_scales()
STN_3M_SCALE_COLUMNS = tuple(
    total_effect_column(scale, PROTOCOL_STN, PHASE_3M)
    for scale in MER_ORDERED_EFFECT_SCALES
)
DELTA_3M_SCALE_COLUMNS = tuple(
    delta_effect_column(scale, PHASE_3M)
    for scale in MER_ORDERED_EFFECT_SCALES
)


@dataclass(frozen=True)
class CoordinateSpec:
    key: str
    predictor: str
    label: str
    slope_file: str


@dataclass(frozen=True)
class ScaleGroupSpec:
    key: str
    label: str
    region: str
    scales: tuple[str, ...]


COORDINATES = (
    CoordinateSpec("x", "zMX", "MNI-x (z-scored)", "zMX_slope.csv"),
    CoordinateSpec("y", "zMY", "MNI-y (z-scored)", "zMY_slope.csv"),
    CoordinateSpec("z", "zMZ", "MNI-z (z-scored)", "zMZ_slope.csv"),
)

SCALE_GROUPS = (
    ScaleGroupSpec(
        key="stn_3m_efficacy",
        label="STN 3m improvement",
        region="STN",
        scales=STN_3M_SCALE_COLUMNS,
    ),
    ScaleGroupSpec(
        key="stn_snr_3m_delta_efficacy",
        label="STN+SNr 3m additional improvement",
        region="SNr",
        scales=DELTA_3M_SCALE_COLUMNS,
    ),
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


def scale_subject_path(scale: str) -> Path:
    return SCALE_DIR / scale / f"scale_subject_{scale}.xlsx"


def load_scale_value_summary(scale: str, expected_region: str) -> dict[str, object]:
    input_path = scale_subject_path(scale)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing scale subject workbook: {input_path}")

    table = pd.read_excel(input_path)
    required_columns = {PAIR_COL, REGION_COL, SCALE_COL, VALUE_COL}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing_columns)}")

    table = table.loc[:, [PAIR_COL, REGION_COL, SCALE_COL, VALUE_COL]].copy()
    observed_scales = sorted(set(table[SCALE_COL].dropna().astype(str)))
    if observed_scales != [scale]:
        raise ValueError(
            f"{input_path} has unexpected Scale values: {observed_scales}. "
            f"Expected only {scale!r}."
        )

    observed_regions = sorted(set(table[REGION_COL].dropna().astype(str)))
    if observed_regions != [expected_region]:
        raise ValueError(
            f"{input_path} has unexpected Region values: {observed_regions}. "
            f"Expected only {expected_region!r}."
        )

    values = pd.to_numeric(table[VALUE_COL], errors="coerce").dropna()
    value_n = int(values.shape[0])
    if value_n < 2:
        raise ValueError(f"{input_path} has fewer than two non-missing Value entries.")

    value_sd = parse_float(values.std(ddof=1))
    if not math.isfinite(value_sd) or value_sd <= 0:
        raise ValueError(f"{input_path} has missing or non-positive Value SD: {value_sd}")

    return {
        "value_sd": value_sd,
        "value_n": value_n,
        "source_value_xlsx": str(input_path),
    }


def load_slope_row(scale: str, coordinate: CoordinateSpec) -> dict[str, object]:
    input_path = SCALE_DIR / scale / MODEL_DIR / coordinate.slope_file
    if not input_path.exists():
        return {
            "slope": math.nan,
            "lower": math.nan,
            "upper": math.nan,
            "p": math.nan,
            "q": math.nan,
            "source_slope_csv": str(input_path),
            "model_available": False,
        }

    table = pd.read_csv(input_path)
    required_columns = {"slope", "lower", "upper", "p", "q"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing_columns)}")
    if table.shape[0] != 1:
        raise ValueError(f"{input_path} must contain exactly one slope row; got {table.shape[0]}.")

    row = table.iloc[0]
    return {
        "slope": parse_float(row["slope"]),
        "lower": parse_float(row["lower"]),
        "upper": parse_float(row["upper"]),
        "p": parse_float(row["p"]),
        "q": parse_float(row["q"]),
        "source_slope_csv": str(input_path),
        "model_available": True,
    }


def build_standardized_slope_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for group in SCALE_GROUPS:
        for scale_order, scale in enumerate(group.scales):
            _validate_scale_name(scale, group)
            value_summary = load_scale_value_summary(scale, expected_region=group.region)
            value_sd = parse_float(value_summary["value_sd"])

            for coordinate in COORDINATES:
                slope_row = load_slope_row(scale, coordinate)
                slope = parse_float(slope_row["slope"])
                lower = parse_float(slope_row["lower"])
                upper = parse_float(slope_row["upper"])
                p_value = parse_float(slope_row["p"])
                q_value = parse_float(slope_row["q"])

                standardized_slope = slope / value_sd if math.isfinite(slope) else math.nan
                standardized_lower = lower / value_sd if math.isfinite(lower) else math.nan
                standardized_upper = upper / value_sd if math.isfinite(upper) else math.nan
                standardized_display = (
                    float(np.clip(standardized_slope, -VALUE_LIMIT, VALUE_LIMIT))
                    if math.isfinite(standardized_slope)
                    else math.nan
                )

                rows.append(
                    {
                        "group": group.key,
                        "group_label": group.label,
                        "region": group.region,
                        "scale": scale,
                        "scale_order": scale_order,
                        "coordinate": coordinate.key,
                        "coordinate_label": coordinate.label,
                        "predictor": coordinate.predictor,
                        "slope": slope,
                        "lower": lower,
                        "upper": upper,
                        "p": p_value,
                        "q": q_value,
                        "value_sd": value_sd,
                        "value_n": value_summary["value_n"],
                        "standardized_slope": standardized_slope,
                        "standardized_lower": standardized_lower,
                        "standardized_upper": standardized_upper,
                        "standardized_slope_display": standardized_display,
                        "star": p_to_star(q_value),
                        "source_slope_csv": slope_row["source_slope_csv"],
                        "source_value_xlsx": value_summary["source_value_xlsx"],
                        "model_available": slope_row["model_available"],
                    }
                )

    return pd.DataFrame(rows)


def validate_standardized_slope_table(table: pd.DataFrame) -> None:
    expected_rows = sum(len(group.scales) for group in SCALE_GROUPS) * len(COORDINATES)
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(table)}.")

    expected_columns = {
        "group",
        "region",
        "scale",
        "coordinate",
        "q",
        "value_sd",
        "standardized_slope",
        "standardized_slope_display",
    }
    missing_columns = expected_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(f"Standardized slope table is missing columns: {sorted(missing_columns)}")

    if table["value_sd"].isna().any() or (table["value_sd"] <= 0).any():
        bad = table[table["value_sd"].isna() | (table["value_sd"] <= 0)]
        raise ValueError(f"Found missing or non-positive value_sd rows: {bad.head().to_dict('records')}")

    if "model_available" not in table.columns:
        raise ValueError("Standardized slope table is missing columns: ['model_available']")

    available_model = table["model_available"].astype(bool)
    if table.loc[available_model, "standardized_slope"].isna().any():
        bad = table[available_model & table["standardized_slope"].isna()]
        raise ValueError(f"Found missing standardized_slope rows: {bad.head().to_dict('records')}")

    missing_model_with_slope = table.loc[
        ~available_model & table["standardized_slope"].notna()
    ]
    if not missing_model_with_slope.empty:
        raise ValueError(
            "Rows without slope models must keep standardized_slope blank: "
            f"{missing_model_with_slope.head().to_dict('records')}"
        )

    for group in SCALE_GROUPS:
        group_table = table[table["group"].eq(group.key)]
        expected_group_rows = len(group.scales) * len(COORDINATES)
        if len(group_table) != expected_group_rows:
            raise ValueError(
                f"{group.key}: expected {expected_group_rows} rows, got {len(group_table)}."
            )
        if set(group_table["region"]) != {group.region}:
            raise ValueError(f"{group.key}: unexpected regions: {sorted(set(group_table['region']))}")
        if group_table["scale"].tolist()[:: len(COORDINATES)] != list(group.scales):
            raise ValueError(f"{group.key}: scale order does not match the configured 3m order.")

    non_lm = table[~table["source_slope_csv"].str.contains(f"/{MODEL_DIR}/", regex=False)]
    if not non_lm.empty:
        raise ValueError(f"Found non-{MODEL_DIR} slope rows: {non_lm.head().to_dict('records')}")


def run_workflow() -> Path:
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    table = build_standardized_slope_table()
    validate_standardized_slope_table(table)
    table.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


def _validate_scale_name(scale: str, group: ScaleGroupSpec) -> None:
    if "Δ%" not in scale:
        raise ValueError(f"Scale name does not contain Δ%: {scale!r}")
    if "3m" not in scale:
        raise ValueError(f"Scale name is not a 3-month variable: {scale!r}")
    if group.key == "stn_3m_efficacy" and scale.startswith("Δ"):
        raise ValueError(f"STN-only scale must not start with Δ: {scale!r}")
    if group.key == "stn_snr_3m_delta_efficacy" and not scale.startswith("Δ"):
        raise ValueError(f"STN+SNr delta scale must start with Δ: {scale!r}")


if __name__ == "__main__":
    output_path = run_workflow()
    print(f"standardized slope table: saved CSV: {output_path}")
