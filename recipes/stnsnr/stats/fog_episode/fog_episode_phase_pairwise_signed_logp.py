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
FOG_EPISODE_DIR = PROJECT / "summary" / "stats" / "fog_episode"
OUTPUT_DIR = FOG_EPISODE_DIR / "corr" / "phase_pairwise"
SIGNED_LOGP_OUTPUT_PATH = OUTPUT_DIR / "fog_episode_phase_pairwise_signed_logp_values.csv"
SIGNED_LOGP_WITH_DELTA_OUTPUT_PATH = (
    OUTPUT_DIR / "fog_episode_phase_pairwise_signed_logp_values_with_delta.csv"
)

MODEL_DIR = "rlmer_model"
LOCAL_TUKEY_FILE = "Phase-Region_tukey.csv"
CONNECTIVITY_TUKEY_FILE = "Phase_tukey.csv"
VALUE_LIMIT = 3.0

PHASE_CONTRASTS = (
    ("BL", "Pre"),
    ("BL", "ONS"),
    ("Pre", "ONS"),
)
SOURCE_CONTRASTS = tuple(f"{first} - {second}" for first, second in PHASE_CONTRASTS)
DISPLAY_COMPARISONS = tuple(f"{first} vs {second}" for first, second in PHASE_CONTRASTS)

BANDS = ("Theta", "Alpha", "Beta_low", "Beta_high", "Gamma")
BANDS_WITH_DELTA = ("Delta", "Theta", "Alpha", "Beta_low", "Beta_high", "Gamma")
BAND_LABELS = {
    "Delta": "δ",
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}
PARTIAL_BURST_BANDS = ("Alpha", "Beta_low", "Gamma")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    base_dir: Path
    bands: tuple[str, ...] = BANDS


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
    MetricSpec(
        "raw",
        "Total power",
        FOG_EPISODE_DIR / "tfr" / "raw" / "raw-mean-scalar",
    ),
    MetricSpec(
        "period",
        "Periodic power",
        FOG_EPISODE_DIR / "tfr" / "period" / "period-mean-scalar",
    ),
    MetricSpec(
        "aperiod",
        "Aperiodic power",
        FOG_EPISODE_DIR / "tfr" / "aperiod" / "aperiod-mean-scalar",
    ),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        FOG_EPISODE_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-rate",
        r"$\operatorname{arcsinh}$ burst rate",
        FOG_EPISODE_DIR / "burst" / "burst" / "burst-rate-scalar",
    ),
    MetricSpec(
        "burst-duration",
        r"$\operatorname{ln}$ burst duration",
        FOG_EPISODE_DIR / "burst" / "burst" / "burst-duration-scalar",
        PARTIAL_BURST_BANDS,
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        FOG_EPISODE_DIR / "burst" / "burst" / "burst-mean-scalar",
        PARTIAL_BURST_BANDS,
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        FOG_EPISODE_DIR / "conn" / "coh" / "coh-mean-scalar",
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        FOG_EPISODE_DIR / "conn" / "wpli" / "wpli-mean-scalar",
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        FOG_EPISODE_DIR / "conn" / "ciplv" / "ciplv-mean-scalar",
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        FOG_EPISODE_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "PSI", FOG_EPISODE_DIR / "psi" / "psi" / "psi-mean-scalar"),
)


def metrics_with_full_bands(
    metrics: tuple[MetricSpec, ...],
    full_bands: tuple[str, ...],
) -> tuple[MetricSpec, ...]:
    return tuple(
        MetricSpec(
            key=metric.key,
            label=metric.label,
            base_dir=metric.base_dir,
            bands=full_bands if metric.bands == BANDS else metric.bands,
        )
        for metric in metrics
    )


LOCAL_METRICS_WITH_DELTA = metrics_with_full_bands(LOCAL_METRICS, BANDS_WITH_DELTA)
CONNECTIVITY_METRICS_WITH_DELTA = metrics_with_full_bands(
    CONNECTIVITY_METRICS,
    BANDS_WITH_DELTA,
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

GROUPS_WITH_DELTA = (
    GroupSpec(
        key="stn_local",
        label_top="STN",
        metrics=LOCAL_METRICS_WITH_DELTA,
        tukey_file=LOCAL_TUKEY_FILE,
        region="STN",
    ),
    GroupSpec(
        key="snr_local",
        label_top="SNr",
        metrics=LOCAL_METRICS_WITH_DELTA,
        tukey_file=LOCAL_TUKEY_FILE,
        region="SNr",
    ),
    GroupSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        metrics=CONNECTIVITY_METRICS_WITH_DELTA,
        tukey_file=CONNECTIVITY_TUKEY_FILE,
        region=None,
    ),
)


PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)
PARAMETER_ROOT = FOG_EPISODE_DIR / "params" / "params" / "params-mean-scalar"
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_GROUP_KEY = "parameters"
PARAMETER_LABEL_TOP = "Params"

EXPECTED_ROW_COUNT = (
    2 * sum(len(metric.bands) for metric in LOCAL_METRICS) * len(PHASE_CONTRASTS)
    + len(CONNECTIVITY_METRICS) * len(BANDS) * len(PHASE_CONTRASTS)
    + len(PARAMETERS) * len(PARAMETER_REGIONS) * len(PHASE_CONTRASTS)
)
EXPECTED_ROW_COUNT_WITH_DELTA = (
    2 * sum(len(metric.bands) for metric in LOCAL_METRICS_WITH_DELTA) * len(PHASE_CONTRASTS)
    + len(CONNECTIVITY_METRICS_WITH_DELTA) * len(BANDS_WITH_DELTA) * len(PHASE_CONTRASTS)
    + len(PARAMETERS) * len(PARAMETER_REGIONS) * len(PHASE_CONTRASTS)
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


def source_contrast(first_phase: str, second_phase: str) -> str:
    return f"{first_phase} - {second_phase}"


def display_comparison(first_phase: str, second_phase: str) -> str:
    return f"{first_phase} vs {second_phase}"


def load_phase_row(
    path: Path,
    *,
    first_phase: str,
    second_phase: str,
    region: str | None,
) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Tukey CSV: {path}")

    contrast = source_contrast(first_phase, second_phase)
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
    first_phase: str,
    second_phase: str,
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
        "source_phase_first": first_phase,
        "source_phase_second": second_phase,
        "plot_phase_first": first_phase,
        "plot_phase_second": second_phase,
        "source_contrast": clean_value(row.get("contrast")),
        "comparison": display_comparison(first_phase, second_phase),
        "estimate_label": f"{second_phase} - {first_phase}",
        "source_model": MODEL_DIR,
        "source_estimate_first_minus_second": source_estimate,
        "estimate_second_minus_first": flipped_estimate,
        "p_tukey": p_value,
        "signed_logp": value,
        "signed_logp_display": value_display,
        "star": p_to_star(p_value),
        "source_csv": str(source_csv),
    }


def build_group_records(
    group: GroupSpec,
    *,
    allowed_bands: tuple[str, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for metric in group.metrics:
        for band in metric.bands:
            if band not in allowed_bands:
                raise ValueError(
                    f"Unexpected band {band!r}; allowed bands are {list(allowed_bands)}."
                )
            path = source_path(metric, band, group.tukey_file)
            for first_phase, second_phase in PHASE_CONTRASTS:
                row = load_phase_row(
                    path,
                    first_phase=first_phase,
                    second_phase=second_phase,
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
                        first_phase=first_phase,
                        second_phase=second_phase,
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
            for first_phase, second_phase in PHASE_CONTRASTS:
                row = load_phase_row(
                    path,
                    first_phase=first_phase,
                    second_phase=second_phase,
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
                        first_phase=first_phase,
                        second_phase=second_phase,
                        source_csv=path,
                        row=row,
                    )
                )
    return records


def build_signed_logp_table(
    *,
    groups: tuple[GroupSpec, ...] = GROUPS,
    allowed_bands: tuple[str, ...] = BANDS,
    expected_row_count: int = EXPECTED_ROW_COUNT,
    allow_delta: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group in groups:
        records.extend(build_group_records(group, allowed_bands=allowed_bands))
    records.extend(build_parameter_records())

    table = pd.DataFrame.from_records(records)
    validate_signed_logp_table(
        table,
        expected_row_count=expected_row_count,
        allow_delta=allow_delta,
    )
    return table


def validate_signed_logp_table(
    table: pd.DataFrame,
    *,
    expected_row_count: int = EXPECTED_ROW_COUNT,
    allow_delta: bool = False,
) -> None:
    if len(table) != expected_row_count:
        raise ValueError(f"Expected {expected_row_count} signed-logP rows, got {len(table)}.")

    if (table["source_model"] != MODEL_DIR).any():
        raise ValueError(f"All source rows must use {MODEL_DIR}.")
    if set(table["source_contrast"].unique()) != set(SOURCE_CONTRASTS):
        raise ValueError(f"Source contrasts must match {list(SOURCE_CONTRASTS)}.")
    if tuple(table["comparison"].drop_duplicates()) != DISPLAY_COMPARISONS:
        raise ValueError(f"Comparison order must be {list(DISPLAY_COMPARISONS)}.")
    has_delta = (table["band"] == "Delta").any()
    if not allow_delta and has_delta:
        raise ValueError("Delta rows are excluded from this workflow.")
    if allow_delta and not has_delta:
        raise ValueError("The with-Delta workflow must include Delta rows.")

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


def run_workflow(
    output_path: str | Path = SIGNED_LOGP_OUTPUT_PATH,
    *,
    groups: tuple[GroupSpec, ...] = GROUPS,
    allowed_bands: tuple[str, ...] = BANDS,
    expected_row_count: int = EXPECTED_ROW_COUNT,
    allow_delta: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_signed_logp_table(
        groups=groups,
        allowed_bands=allowed_bands,
        expected_row_count=expected_row_count,
        allow_delta=allow_delta,
    )
    table.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_path = run_workflow()
    print(f"Saved {EXPECTED_ROW_COUNT} signed-logP rows: {saved_path}")
    saved_with_delta_path = run_workflow(
        SIGNED_LOGP_WITH_DELTA_OUTPUT_PATH,
        groups=GROUPS_WITH_DELTA,
        allowed_bands=BANDS_WITH_DELTA,
        expected_row_count=EXPECTED_ROW_COUNT_WITH_DELTA,
        allow_delta=True,
    )
    print(
        f"Saved {EXPECTED_ROW_COUNT_WITH_DELTA} with-Delta signed-logP rows: "
        f"{saved_with_delta_path}"
    )
