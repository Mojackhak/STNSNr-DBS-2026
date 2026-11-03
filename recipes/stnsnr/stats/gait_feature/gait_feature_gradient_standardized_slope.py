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
STATS_DIR = PROJECT / "summary" / "stats"
OUTPUT_SUBDIR = "corr/gait_standardized_slope"

MODEL_DIR = "rlmer_model"
VALUE_LIMIT = 0.6
LOCAL_SD_SOURCE = "contact_normalized"
CONNECTIVITY_SD_SOURCE = "contact_normalized_split"
PARAM_SD_SOURCE = "contact_normalized"

BANDS = ("Theta", "Alpha", "Beta_low", "Beta_high", "Gamma")
BAND_LABELS = {
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}
REGIONS = ("STN", "SNr")
PARAMETER_REGIONS = ("SNr", "STN")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    relative_dir: tuple[str, ...]
    sd_source: str


@dataclass(frozen=True)
class GradientSpec:
    key: str
    label: str
    predictor: str


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


@dataclass(frozen=True)
class GaitFeatureSpec:
    key: str
    label: str
    slug: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    root: Path
    output_dir: Path
    output_path: Path
    local_metrics: tuple[MetricSpec, ...]


BASE_LOCAL_METRICS = (
    MetricSpec("raw", "Total power", ("tfr", "raw", "raw-mean-scalar"), LOCAL_SD_SOURCE),
    MetricSpec("period", "Periodic power", ("tfr", "period", "period-mean-scalar"), LOCAL_SD_SOURCE),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        ("burst", "burst", "burst-occupation-scalar"),
        LOCAL_SD_SOURCE,
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        ("burst", "burst", "burst-mean-scalar"),
        LOCAL_SD_SOURCE,
    ),
)

FULL_LOCAL_METRICS = (
    MetricSpec("raw", "Total power", ("tfr", "raw", "raw-mean-scalar"), LOCAL_SD_SOURCE),
    MetricSpec("period", "Periodic power", ("tfr", "period", "period-mean-scalar"), LOCAL_SD_SOURCE),
    MetricSpec(
        "burst-occupation",
        "Burst occupation",
        ("burst", "burst", "burst-occupation-scalar"),
        LOCAL_SD_SOURCE,
    ),
    MetricSpec(
        "burst-rate",
        r"$\operatorname{asinh}$ burst rate",
        ("burst", "burst", "burst-rate-scalar"),
        LOCAL_SD_SOURCE,
    ),
    MetricSpec(
        "burst-duration",
        r"$\operatorname{ln}$ burst duration",
        ("burst", "burst", "burst-duration-scalar"),
        LOCAL_SD_SOURCE,
    ),
    MetricSpec(
        "burst-mean",
        r"$\operatorname{ln}$ burst amplitude",
        ("burst", "burst", "burst-mean-scalar"),
        LOCAL_SD_SOURCE,
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec(
        "coh",
        r"Fisher z of $\sqrt{\,\mathrm{coherence}}$",
        ("conn", "coh", "coh-mean-scalar"),
        CONNECTIVITY_SD_SOURCE,
    ),
    MetricSpec(
        "wpli",
        r"$\operatorname{logit}(\,\mathrm{wPLI})$",
        ("conn", "wpli", "wpli-mean-scalar"),
        CONNECTIVITY_SD_SOURCE,
    ),
    MetricSpec(
        "ciplv",
        r"$\operatorname{logit}(\,\mathrm{ciPLV})$",
        ("conn", "ciplv", "ciplv-mean-scalar"),
        CONNECTIVITY_SD_SOURCE,
    ),
    MetricSpec(
        "delta_net_gc",
        "TRGC",
        ("gc", "delta_net_gc", "delta_net_gc-mean-scalar"),
        CONNECTIVITY_SD_SOURCE,
    ),
    MetricSpec("psi", "PSI", ("psi", "psi", "psi-mean-scalar"), CONNECTIVITY_SD_SOURCE),
)

PARAM_RELATIVE_DIR = ("params", "params", "params-mean-scalar")
PARAM_METRIC_KEY = "params"
PARAM_METRIC_LABEL = "Aperiodic parameters"
PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)

GAIT_FEATURES = (
    GaitFeatureSpec("Walk_speed", "walk speed", "walk_speed"),
    GaitFeatureSpec("Stride_length", "stride length", "stride_length"),
    GaitFeatureSpec("Cycle_duration", "cycle duration", "cycle_duration"),
)

GRADIENTS = (
    GradientSpec("between", "Between", "z_Feature_between"),
    GradientSpec("within", "Within", "z_Feature_within"),
)


def dataset_output_path(key: str) -> Path:
    output_dir = STATS_DIR / key / OUTPUT_SUBDIR
    return output_dir / f"{key}_gradient_standardized_slope_values.csv"


DATASETS = (
    DatasetSpec(
        key="gait_feature",
        root=STATS_DIR / "gait_feature",
        output_dir=STATS_DIR / "gait_feature" / OUTPUT_SUBDIR,
        output_path=dataset_output_path("gait_feature"),
        local_metrics=FULL_LOCAL_METRICS,
    ),
    DatasetSpec(
        key="gait_feature_linear",
        root=STATS_DIR / "gait_feature_linear",
        output_dir=STATS_DIR / "gait_feature_linear" / OUTPUT_SUBDIR,
        output_path=dataset_output_path("gait_feature_linear"),
        local_metrics=BASE_LOCAL_METRICS,
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


def metric_dir(dataset: DatasetSpec, metric: MetricSpec) -> Path:
    return dataset.root.joinpath(*metric.relative_dir)


def parameter_root(dataset: DatasetSpec) -> Path:
    return dataset.root.joinpath(*PARAM_RELATIVE_DIR)


def value_table_path(dataset: DatasetSpec, metric: MetricSpec) -> Path:
    base_dir = metric_dir(dataset, metric)
    return base_dir / f"{base_dir.name}_{metric.sd_source}.csv"


def load_value_sd_by_region(dataset: DatasetSpec, metric: MetricSpec, feature: GaitFeatureSpec) -> pd.DataFrame:
    path = value_table_path(dataset, metric)
    if not path.exists():
        raise FileNotFoundError(f"Missing SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Region", "Feature", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[
            df["Feature"].astype(str).eq(feature.key)
            & df["Band"].isin(BANDS)
            & df["Region"].isin(REGIONS)
        ]
        .groupby(["Band", "Region"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = metric.sd_source
    grouped["sd_group"] = "Feature x Band x Region"
    return grouped


def load_value_sd_by_band(dataset: DatasetSpec, metric: MetricSpec, feature: GaitFeatureSpec) -> pd.DataFrame:
    path = value_table_path(dataset, metric)
    if not path.exists():
        raise FileNotFoundError(f"Missing SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Feature", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[df["Feature"].astype(str).eq(feature.key) & df["Band"].isin(BANDS)]
        .groupby(["Band"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = metric.sd_source
    grouped["sd_group"] = "Feature x Band"
    return grouped


def load_parameter_value_sd(dataset: DatasetSpec, feature: GaitFeatureSpec) -> pd.DataFrame:
    root = parameter_root(dataset)
    path = root / f"{root.name}_{PARAM_SD_SOURCE}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing parameter SD source table: {path}")

    df = pd.read_csv(path)
    required = {"Band", "Region", "Feature", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    parameter_keys = [parameter.key for parameter in PARAMETERS]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    grouped = (
        df[
            df["Feature"].astype(str).eq(feature.key)
            & df["Band"].isin(parameter_keys)
            & df["Region"].isin(REGIONS)
        ]
        .groupby(["Band", "Region"], observed=True)["Value"]
        .agg(value_sd=lambda values: values.std(ddof=1), value_n="count")
        .reset_index()
    )
    grouped["source_sd_csv"] = str(path)
    grouped["sd_source"] = PARAM_SD_SOURCE
    grouped["sd_group"] = "Feature x Band x Region"
    return grouped


def local_slope_file(gradient: GradientSpec, feature: GaitFeatureSpec) -> str:
    return f"{gradient.predictor}-Region_{feature.key}_slope.csv"


def connectivity_slope_file(gradient: GradientSpec, feature: GaitFeatureSpec) -> str:
    return f"{gradient.predictor}_{feature.key}_slope.csv"


def load_slope_rows(path: Path, *, by_region: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing slope table: {path}")

    df = pd.read_csv(path)
    required = {"slope", "lower", "upper", "p", "q"}
    if by_region:
        required.add("Region")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if by_region:
        df = df[df["Region"].isin(REGIONS)].copy()
    else:
        df = df.copy()
        df["Region"] = ""
    for column in ["slope", "lower", "upper", "p", "q"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_standardized_slope_table(dataset: DatasetSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for feature in GAIT_FEATURES:
        rows.extend(build_local_records(dataset, feature))
        rows.extend(build_connectivity_records(dataset, feature))
        rows.extend(build_parameter_records(dataset, feature))

    return pd.DataFrame(rows)


def build_local_records(dataset: DatasetSpec, feature: GaitFeatureSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in dataset.local_metrics:
        sd_table = load_value_sd_by_region(dataset, metric, feature)
        base_dir = metric_dir(dataset, metric)
        for band in BANDS:
            sd_for_band = sd_table[sd_table["Band"].eq(band)].set_index("Region")
            for gradient in GRADIENTS:
                slope_path = base_dir / band / MODEL_DIR / local_slope_file(gradient, feature)
                slope_rows = load_slope_rows(slope_path, by_region=True)
                for _, slope_row in slope_rows.iterrows():
                    region = str(slope_row["Region"])
                    if region not in sd_for_band.index:
                        raise ValueError(
                            f"Missing SD row for dataset={dataset.key}, feature={feature.key}, "
                            f"metric={metric.key}, band={band}, region={region}"
                        )
                    rows.append(
                        make_standardized_slope_record(
                            dataset=dataset,
                            feature=feature,
                            feature_group="frequency_metric",
                            heatmap_group=f"{region.lower()}_local",
                            metric_key=metric.key,
                            metric_label_text=metric.label,
                            band=band,
                            band_label=BAND_LABELS[band],
                            parameter="",
                            parameter_label="",
                            region=region,
                            gradient=gradient,
                            slope_row=slope_row,
                            sd_row=sd_for_band.loc[region],
                            slope_path=slope_path,
                        )
                    )
    return rows


def build_connectivity_records(dataset: DatasetSpec, feature: GaitFeatureSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in CONNECTIVITY_METRICS:
        sd_table = load_value_sd_by_band(dataset, metric, feature).set_index("Band")
        base_dir = metric_dir(dataset, metric)
        for band in BANDS:
            if band not in sd_table.index:
                raise ValueError(
                    f"Missing SD row for dataset={dataset.key}, feature={feature.key}, metric={metric.key}, band={band}"
                )
            for gradient in GRADIENTS:
                slope_path = base_dir / band / MODEL_DIR / connectivity_slope_file(gradient, feature)
                slope_rows = load_slope_rows(slope_path, by_region=False)
                if len(slope_rows) != 1:
                    raise ValueError(f"Expected one slope row in {slope_path}, got {len(slope_rows)}")
                rows.append(
                    make_standardized_slope_record(
                        dataset=dataset,
                        feature=feature,
                        feature_group="frequency_metric",
                        heatmap_group="connectivity",
                        metric_key=metric.key,
                        metric_label_text=metric.label,
                        band=band,
                        band_label=BAND_LABELS[band],
                        parameter="",
                        parameter_label="",
                        region="",
                        gradient=gradient,
                        slope_row=slope_rows.iloc[0],
                        sd_row=sd_table.loc[band],
                        slope_path=slope_path,
                    )
                )
    return rows


def build_parameter_records(dataset: DatasetSpec, feature: GaitFeatureSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = parameter_root(dataset)
    parameter_sd_table = load_parameter_value_sd(dataset, feature)
    for parameter in PARAMETERS:
        sd_for_parameter = parameter_sd_table[parameter_sd_table["Band"].eq(parameter.key)].set_index("Region")
        for gradient in GRADIENTS:
            slope_path = root / parameter.key / MODEL_DIR / local_slope_file(gradient, feature)
            slope_rows = load_slope_rows(slope_path, by_region=True)
            for _, slope_row in slope_rows.iterrows():
                region = str(slope_row["Region"])
                if region not in sd_for_parameter.index:
                    raise ValueError(
                        f"Missing SD row for dataset={dataset.key}, feature={feature.key}, "
                        f"parameter={parameter.key}, region={region}"
                    )
                rows.append(
                    make_standardized_slope_record(
                        dataset=dataset,
                        feature=feature,
                        feature_group="parameter",
                        heatmap_group="params",
                        metric_key=PARAM_METRIC_KEY,
                        metric_label_text=PARAM_METRIC_LABEL,
                        band=parameter.key,
                        band_label=parameter.label,
                        parameter=parameter.key,
                        parameter_label=parameter.label,
                        region=region,
                        gradient=gradient,
                        slope_row=slope_row,
                        sd_row=sd_for_parameter.loc[region],
                        slope_path=slope_path,
                    )
                )
    return rows


def make_standardized_slope_record(
    *,
    dataset: DatasetSpec,
    feature: GaitFeatureSpec,
    feature_group: str,
    heatmap_group: str,
    metric_key: str,
    metric_label_text: str,
    band: str,
    band_label: str,
    parameter: str,
    parameter_label: str,
    region: str,
    gradient: GradientSpec,
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
        "dataset": dataset.key,
        "feature_group": feature_group,
        "heatmap_group": heatmap_group,
        "metric": metric_key,
        "metric_label": metric_label_text,
        "band": band,
        "band_label": band_label,
        "parameter": parameter,
        "parameter_label": parameter_label,
        "region": region,
        "feature": feature.key,
        "feature_label": feature.label,
        "gradient": gradient.key,
        "gradient_label": gradient.label,
        "predictor": gradient.predictor,
        "source_model": MODEL_DIR,
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
        "sd_group": str(sd_row["sd_group"]),
    }


def expected_row_count(dataset: DatasetSpec) -> int:
    per_feature = (
        len(dataset.local_metrics) * len(BANDS) * len(REGIONS) * len(GRADIENTS)
        + len(CONNECTIVITY_METRICS) * len(BANDS) * len(GRADIENTS)
        + len(PARAMETERS) * len(PARAMETER_REGIONS) * len(GRADIENTS)
    )
    return len(GAIT_FEATURES) * per_feature


def validate_standardized_slope_table(df: pd.DataFrame, dataset: DatasetSpec) -> None:
    expected_rows = expected_row_count(dataset)
    if len(df) != expected_rows:
        raise ValueError(f"{dataset.key}: expected {expected_rows} rows, got {len(df)}")

    if set(df["dataset"].dropna()) != {dataset.key}:
        raise ValueError(f"{dataset.key}: unexpected dataset values {sorted(df['dataset'].dropna().unique())}")
    if set(df["feature"].dropna()) != {feature.key for feature in GAIT_FEATURES}:
        raise ValueError(f"{dataset.key}: unexpected feature values {sorted(df['feature'].dropna().unique())}")
    if "Delta" in set(df["band"].dropna()):
        raise ValueError(f"{dataset.key}: Delta band should be excluded")
    if set(df["source_model"].dropna()) != {MODEL_DIR}:
        raise ValueError(f"{dataset.key}: unexpected source_model values {sorted(df['source_model'].dropna().unique())}")

    bad_paths = [path for path in df["source_slope_csv"].astype(str) if f"/{MODEL_DIR}/" not in path]
    if bad_paths:
        raise ValueError(f"{dataset.key}: found source paths outside {MODEL_DIR}: {bad_paths[:5]}")

    if df["value_sd"].isna().any() or (df["value_sd"] <= 0).any():
        bad = df[df["value_sd"].isna() | (df["value_sd"] <= 0)]
        raise ValueError(f"{dataset.key}: found missing or non-positive value_sd rows: {bad.head().to_dict('records')}")

    if df["standardized_slope"].isna().any():
        bad = df[df["standardized_slope"].isna()]
        raise ValueError(f"{dataset.key}: found missing standardized_slope rows: {bad.head().to_dict('records')}")

    connectivity_bad = df[df["heatmap_group"].eq("connectivity") & ~df["sd_source"].eq(CONNECTIVITY_SD_SOURCE)]
    if not connectivity_bad.empty:
        raise ValueError(
            f"{dataset.key}: connectivity rows must use contact_normalized_split SD sources: "
            f"{connectivity_bad.head().to_dict('records')}"
        )

    local_bad = df[
        df["heatmap_group"].isin(["stn_local", "snr_local", "params"])
        & ~df["sd_source"].eq(LOCAL_SD_SOURCE)
    ]
    if not local_bad.empty:
        raise ValueError(
            f"{dataset.key}: local and parameter rows must use contact_normalized SD sources: "
            f"{local_bad.head().to_dict('records')}"
        )


def write_standardized_slope_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_workflow() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for dataset in DATASETS:
        standardized_table = build_standardized_slope_table(dataset)
        validate_standardized_slope_table(standardized_table, dataset)
        write_standardized_slope_table(standardized_table, dataset.output_path)
        outputs[dataset.key] = dataset.output_path
    return outputs


if __name__ == "__main__":
    for dataset_key, output_path in run_workflow().items():
        print(f"{dataset_key}: saved gait-feature gradient standardized slopes: {output_path}")
