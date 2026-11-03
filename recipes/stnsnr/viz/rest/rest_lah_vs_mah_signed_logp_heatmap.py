from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import math
import re
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plugin.default.viz_defaults as cfg
import src.viz.visualdf as vdf
from recipes.stnsnr.stats.rest.rest_scale_correlation import OUTPUT_DIR, REST_DIR


BANDS = ["Delta", "Theta", "Alpha", "Beta_low", "Beta_high", "Gamma"]
BAND_LABELS = {
    "Delta": "δ",
    "Theta": "θ",
    "Alpha": "α",
    "Beta_low": "Low-β",
    "Beta_high": "High-β",
    "Gamma": "γ",
}
VALUE_LIMIT = 4.0
COMPARISON_TEXT = "LAH vs MAH"
CONTRAST_TEXT = "LAH - MAH"
MODEL_DIR = "rlmer_model"
UNIT_SUFFIX_RE = re.compile(r"\s+\((?:dB|Hz|s|V|a\.u\.)\)\s*$")
COLORBAR_LABEL = r"Signed $-\log_{10}(p_{\mathrm{Tukey}})$"
COLORBAR_TOP_LABEL = "LAH > MAH"
COLORBAR_BOTTOM_LABEL = "LAH < MAH"
COLORBAR_DIRECTION_X_OFFSET_MM = 1.0
COLORBAR_DIRECTION_Y_OFFSET_MM = 2.5
COLORBAR_DIRECTION_FONTSIZE = 6


@dataclass(frozen=True)
class MetricSpec:
    key: str
    param_type: str
    base_dir: Path


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label_top: str
    metrics: tuple[MetricSpec, ...]
    tukey_file: str
    region: str | None
    pdf_path: Path
    png_path: Path


LOCAL_METRICS = (
    MetricSpec("raw", "raw-mean", REST_DIR / "tfr" / "raw" / "raw-mean-scalar"),
    MetricSpec("period", "period-mean", REST_DIR / "tfr" / "period" / "period-mean-scalar"),
    MetricSpec(
        "burst-occupation",
        "burst-occupation",
        REST_DIR / "burst" / "burst" / "burst-occupation-scalar",
    ),
    MetricSpec(
        "burst-rate",
        "burst-rate",
        REST_DIR / "burst" / "burst" / "burst-rate-scalar",
    ),
    MetricSpec(
        "burst-duration",
        "burst-duration",
        REST_DIR / "burst" / "burst" / "burst-duration-scalar",
    ),
    MetricSpec(
        "burst-mean",
        "burst-mean",
        REST_DIR / "burst" / "burst" / "burst-mean-scalar",
    ),
)

CONNECTIVITY_METRICS = (
    MetricSpec("coh", "coh-mean", REST_DIR / "conn" / "coh" / "coh-mean-scalar"),
    MetricSpec("wpli", "wpli-mean", REST_DIR / "conn" / "wpli" / "wpli-mean-scalar"),
    MetricSpec("ciplv", "ciplv-mean", REST_DIR / "conn" / "ciplv" / "ciplv-mean-scalar"),
    MetricSpec(
        "delta_net_gc",
        "delta_net_gc-mean",
        REST_DIR / "gc" / "delta_net_gc" / "delta_net_gc-mean-scalar",
    ),
    MetricSpec("psi", "psi-mean", REST_DIR / "psi" / "psi" / "psi-mean-scalar"),
)

GROUPS = (
    GroupSpec(
        key="stn_local",
        label_top="STN",
        metrics=LOCAL_METRICS,
        tukey_file="Hemi-Region_tukey.csv",
        region="STN",
        pdf_path=OUTPUT_DIR / "rest_stn_local_lah_vs_mah_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "rest_stn_local_lah_vs_mah_signed_logp_heatmap.png",
    ),
    GroupSpec(
        key="snr_local",
        label_top="SNr",
        metrics=LOCAL_METRICS,
        tukey_file="Hemi-Region_tukey.csv",
        region="SNr",
        pdf_path=OUTPUT_DIR / "rest_snr_local_lah_vs_mah_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "rest_snr_local_lah_vs_mah_signed_logp_heatmap.png",
    ),
    GroupSpec(
        key="connectivity",
        label_top="SNr-/→STN",
        metrics=CONNECTIVITY_METRICS,
        tukey_file="Hemi_tukey.csv",
        region=None,
        pdf_path=OUTPUT_DIR / "rest_connectivity_lah_vs_mah_signed_logp_heatmap.pdf",
        png_path=OUTPUT_DIR / "rest_connectivity_lah_vs_mah_signed_logp_heatmap.png",
    ),
)

PARAM_ROOT = REST_DIR / "params" / "params" / "params-mean-scalar"
PARAMETERS = (
    ParameterSpec("Exponent", "Exponent"),
    ParameterSpec("Offset", "Offset"),
)
PARAMETER_REGIONS = ("SNr", "STN")
PARAMETER_TUKEY_FILE = "Hemi-Region_tukey.csv"
PARAMETER_LABEL_TOP = "Params"
PARAMETER_PDF_PATH = OUTPUT_DIR / "rest_params_lah_vs_mah_signed_logp_heatmap.pdf"
PARAMETER_PNG_PATH = OUTPUT_DIR / "rest_params_lah_vs_mah_signed_logp_heatmap.png"


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


def comparison_label(row: dict[str, str]) -> str:
    left = clean_value(row.get("group1")) or clean_value(row.get("lhs"))
    right = clean_value(row.get("group2")) or clean_value(row.get("rhs"))
    if left and right:
        return f"{left} vs {right}"
    contrast = clean_value(row.get("contrast"))
    if contrast == CONTRAST_TEXT:
        return COMPARISON_TEXT
    return contrast


def is_lah_vs_mah(row: dict[str, str]) -> bool:
    return comparison_label(row) == COMPARISON_TEXT or clean_value(row.get("contrast")) == CONTRAST_TEXT


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


def metric_label(param_type: str) -> str:
    label = cfg.format_label(param_type, prefix="", key="")
    label = UNIT_SUFFIX_RE.sub("", label)
    label = " ".join(label.split())
    if label.startswith("burst "):
        label = label[0].upper() + label[1:]
    return label.strip()


def load_lah_vs_mah_row(path: Path, *, region: str | None) -> dict[str, str] | None:
    if path.name.lower().startswith("region"):
        raise ValueError(f"Region-prefixed files must not be read: {path}")
    if not path.exists():
        return None

    matches: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not is_lah_vs_mah(row):
                continue
            if region is not None and clean_value(row.get("Region")) != region:
                continue
            matches.append(row)

    if len(matches) > 1:
        raise ValueError(f"Expected at most one {COMPARISON_TEXT} row in {path}, got {len(matches)}.")
    return matches[0] if matches else None


def source_path(metric: MetricSpec, band: str, tukey_file: str) -> Path:
    return metric.base_dir / band / MODEL_DIR / tukey_file


def parameter_source_path(parameter: ParameterSpec) -> Path:
    return PARAM_ROOT / parameter.key / MODEL_DIR / PARAMETER_TUKEY_FILE


def build_group_matrices(
    group: GroupSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    row_labels = [metric_label(metric.param_type) for metric in group.metrics]
    column_labels = [BAND_LABELS[band] for band in BANDS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    metadata: list[dict[str, object]] = []

    for metric, row_label in zip(group.metrics, row_labels, strict=True):
        for band, band_label in zip(BANDS, column_labels, strict=True):
            path = source_path(metric, band, group.tukey_file)
            row = load_lah_vs_mah_row(path, region=group.region)

            estimate = math.nan
            p_value = math.nan
            value = math.nan
            value_display = math.nan
            star = ""
            source_csv = str(path)

            if row is not None:
                estimate = parse_float(row.get("estimate"))
                p_value = parse_float(row.get("p_tukey"))
                value = signed_logp(estimate, p_value)
                value_display = float(np.clip(value, -VALUE_LIMIT, VALUE_LIMIT)) if math.isfinite(value) else math.nan
                star = p_to_star(p_value)
                value_matrix.loc[row_label, band_label] = value_display
                p_matrix.loc[row_label, band_label] = p_value

            metadata.append(
                {
                    "group": group.key,
                    "label_top": group.label_top,
                    "metric": metric.key,
                    "metric_label": row_label,
                    "band": band,
                    "band_label": band_label,
                    "region": group.region or "",
                    "comparison": COMPARISON_TEXT,
                    "estimate": estimate,
                    "p_tukey": p_value,
                    "signed_logp": value,
                    "signed_logp_display": value_display,
                    "star": star,
                    "source_csv": source_csv,
                }
            )

    return value_matrix, p_matrix, metadata


def build_parameter_matrices() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    row_labels = list(PARAMETER_REGIONS)
    column_labels = [parameter.label for parameter in PARAMETERS]
    value_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    p_matrix = pd.DataFrame(np.nan, index=row_labels, columns=column_labels)
    metadata: list[dict[str, object]] = []

    for parameter in PARAMETERS:
        path = parameter_source_path(parameter)
        for region in PARAMETER_REGIONS:
            row = load_lah_vs_mah_row(path, region=region)

            estimate = math.nan
            p_value = math.nan
            value = math.nan
            value_display = math.nan
            star = ""
            source_csv = str(path)

            if row is not None:
                estimate = parse_float(row.get("estimate"))
                p_value = parse_float(row.get("p_tukey"))
                value = signed_logp(estimate, p_value)
                value_display = float(np.clip(value, -VALUE_LIMIT, VALUE_LIMIT)) if math.isfinite(value) else math.nan
                star = p_to_star(p_value)
                value_matrix.loc[region, parameter.label] = value_display
                p_matrix.loc[region, parameter.label] = p_value

            metadata.append(
                {
                    "group": "parameters",
                    "label_top": PARAMETER_LABEL_TOP,
                    "metric": "params",
                    "metric_label": region,
                    "band": parameter.key,
                    "band_label": parameter.label,
                    "parameter": parameter.key,
                    "parameter_label": parameter.label,
                    "region": region,
                    "comparison": COMPARISON_TEXT,
                    "estimate": estimate,
                    "p_tukey": p_value,
                    "signed_logp": value,
                    "signed_logp_display": value_display,
                    "star": star,
                    "source_csv": source_csv,
                }
            )

    return value_matrix, p_matrix, metadata


def plot_heatmap(
    value_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    *,
    label_top: str,
    cellsize: tuple[float, float] = (4.0, 4.0),
    direction_y_offset_mm: float = COLORBAR_DIRECTION_Y_OFFSET_MM,
    direction_fontsize: float = COLORBAR_DIRECTION_FONTSIZE,
) -> plt.Figure:
    fig = vdf.plot_well_heatmap_df(
        df_value=value_matrix,
        df_p=p_matrix,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=label_top,
        cmap=cfg.cm.vik,
        vmin=-VALUE_LIMIT,
        vmax=VALUE_LIMIT,
        vmode="sym",
        cellsize=cellsize,
        label_top_height_mm=cfg.strip_top_height_mm,
        label_top_pad_mm=cfg.strip_pad_mm,
        label_top_bg_color=cfg.label_top_bg_color,
        label_top_text_color=cfg.label_text_color,
        label_top_fontsize=7,
        label_top_fontweight="bold",
        x_label_offset_mm=0.0,
        y_label_offset_mm=18.0,
        cbar_label_offset_mm=5.0,
        grid_lines=None,
        gline_width=0.6,
        gline_alpha=0.9,
        axis_label_fontsize=7,
        tick_label_fontsize=6,
        colorbar_label=COLORBAR_LABEL,
        colorbar_width_mm=3.0,
        colorbar_pad_mm=2.0,
        colorbar_length_mm=None,
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=7,
        transparent=True,
    )
    set_colorbar_ticks(fig)
    add_colorbar_end_labels(
        fig,
        y_offset_mm=direction_y_offset_mm,
        fontsize=direction_fontsize,
    )
    return fig


def set_colorbar_ticks(fig: plt.Figure) -> None:
    colorbar_axis = find_colorbar_axis(fig)
    if colorbar_axis is None:
        return
    colorbar_axis.set_yticks([-VALUE_LIMIT, 0.0, VALUE_LIMIT])
    colorbar_axis.set_yticklabels([f"{-VALUE_LIMIT:g}", "0", f"{VALUE_LIMIT:g}"])


def add_colorbar_end_labels(
    fig: plt.Figure,
    *,
    x_offset_mm: float = COLORBAR_DIRECTION_X_OFFSET_MM,
    y_offset_mm: float = COLORBAR_DIRECTION_Y_OFFSET_MM,
    fontsize: float = COLORBAR_DIRECTION_FONTSIZE,
) -> None:
    colorbar_axis = find_colorbar_axis(fig)
    if colorbar_axis is None:
        return
    heatmap_axis = find_heatmap_axis(fig, colorbar_axis)
    if heatmap_axis is None:
        return

    pos = colorbar_axis.get_position()
    heatmap_pos = heatmap_axis.get_position()
    fig_width_in = fig.get_size_inches()[0]
    fig_height_in = fig.get_size_inches()[1]
    x_offset = (x_offset_mm / 25.4) / fig_width_in
    y_offset = (y_offset_mm / 25.4) / fig_height_in
    x_label = heatmap_pos.x1 + x_offset
    fig.text(
        x_label,
        pos.y1 + y_offset,
        COLORBAR_TOP_LABEL,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )
    fig.text(
        x_label,
        pos.y0 - y_offset,
        COLORBAR_BOTTOM_LABEL,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="black",
    )


def find_colorbar_axis(fig: plt.Figure) -> plt.Axes | None:
    candidates = []
    for ax in fig.axes:
        pos = ax.get_position()
        if pos.width < 0.18 and pos.height > 0.25 and len(ax.get_xticks()) == 0 and len(ax.get_yticks()) > 0:
            candidates.append(ax)
    if not candidates:
        return None
    return max(candidates, key=lambda ax: ax.get_position().x0)


def find_heatmap_axis(fig: plt.Figure, colorbar_axis: plt.Axes) -> plt.Axes | None:
    candidates = []
    for ax in fig.axes:
        if ax is colorbar_axis:
            continue
        pos = ax.get_position()
        if pos.width > 0.2 and pos.height > 0.3:
            candidates.append(ax)
    if not candidates:
        return None
    return max(candidates, key=lambda ax: ax.get_position().width * ax.get_position().height)


def validate_metadata(metadata: Iterable[dict[str, object]]) -> None:
    bad_model_paths = [
        str(row["source_csv"])
        for row in metadata
        if row["source_csv"] and f"/{MODEL_DIR}/" not in str(row["source_csv"])
    ]
    if bad_model_paths:
        raise ValueError(f"Found non-{MODEL_DIR} source paths: {bad_model_paths[:5]}")

    bad_region_files = [
        str(row["source_csv"])
        for row in metadata
        if Path(str(row["source_csv"])).name.lower().startswith("region")
    ]
    if bad_region_files:
        raise ValueError(f"Found Region-prefixed source files: {bad_region_files[:5]}")


def run_workflow() -> dict[str, tuple[Path, Path]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[Path, Path]] = {}
    all_metadata: list[dict[str, object]] = []
    for group in GROUPS:
        value_matrix, p_matrix, metadata = build_group_matrices(group)
        validate_metadata(metadata)
        all_metadata.extend(metadata)

        fig = plot_heatmap(value_matrix, p_matrix, label_top=group.label_top)
        vdf._ensure_strip_background_opaque(fig)
        fig.savefig(group.pdf_path, bbox_inches="tight", dpi=cfg.dpi)
        fig.savefig(group.png_path, bbox_inches="tight", dpi=cfg.dpi)
        plt.close(fig)
        outputs[group.key] = (group.pdf_path, group.png_path)

    value_matrix, p_matrix, metadata = build_parameter_matrices()
    validate_metadata(metadata)
    all_metadata.extend(metadata)

    fig = plot_heatmap(
        value_matrix,
        p_matrix,
        label_top=PARAMETER_LABEL_TOP,
        cellsize=(4.0, 4.0),
        direction_y_offset_mm=7.0,
        direction_fontsize=5.0,
    )
    vdf._ensure_strip_background_opaque(fig)
    fig.savefig(PARAMETER_PDF_PATH, bbox_inches="tight", dpi=cfg.dpi)
    fig.savefig(PARAMETER_PNG_PATH, bbox_inches="tight", dpi=cfg.dpi)
    plt.close(fig)
    outputs["parameters"] = (PARAMETER_PDF_PATH, PARAMETER_PNG_PATH)

    metadata_path = OUTPUT_DIR / "rest_lah_vs_mah_signed_logp_heatmap_values.csv"
    pd.DataFrame(all_metadata).to_csv(metadata_path, index=False)
    outputs["metadata"] = (metadata_path, metadata_path)
    return outputs


if __name__ == "__main__":
    heatmap_outputs = run_workflow()
    for group_name, paths in heatmap_outputs.items():
        if group_name == "metadata":
            print(f"{group_name}: saved CSV: {paths[0]}")
        else:
            print(f"{group_name}: saved heatmap PDF: {paths[0]}")
            print(f"{group_name}: saved heatmap PNG: {paths[1]}")
