"""Preprocessing helpers for gait-cycle pointwise statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

import plugin.default.derive_default as derive_cfg
import plugin.default.stats_gait_cycle_default as gait_cfg
from plugin.default.tensors_default import bands as DEFAULT_BANDS
from src.io.pkl_io import load_pkl, save_pkl
from src.stats.preproc.aggregate import summarize_df
from src.stats.preproc.transform import transform_df
from src.tabular.reshape import explode_value_column
from src.utils.config import get_param
from src.utils.transforms import apply_transform_array


@dataclass(frozen=True)
class GaitCycleExportReport:
    """Summary of gait-cycle preprocessing outputs."""

    output_root: Path
    trace_paths: tuple[Path, ...]
    input_paths: tuple[Path, ...]
    n_trace_rows: int
    n_long_rows: int


def discover_raw_gait_paths(
    input_root: str | Path = gait_cfg.RAW_GAIT_ROOT,
) -> tuple[Path, ...]:
    """Return raw gait pickle paths in the expected parameter hierarchy."""
    root = Path(input_root).expanduser().resolve()
    return tuple(sorted(path for path in root.glob("*/*/*-na-raw.pkl") if path.is_file()))


def export_gait_cycle_region_tables(
    *,
    input_root: str | Path = gait_cfg.RAW_GAIT_ROOT,
    output_root: str | Path = gait_cfg.GAIT_CYCLE_STATS_ROOT,
    task: str = gait_cfg.DEFAULT_TASK,
    overwrite: bool = True,
) -> GaitCycleExportReport:
    """Build Region-level gait-cycle trace and band-level long input tables."""
    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    raw_paths = discover_raw_gait_paths(input_root)
    if not raw_paths:
        raise FileNotFoundError(f"No raw gait pickle files found under {input_root}")

    trace_paths: list[Path] = []
    input_paths: list[Path] = []
    n_trace_rows = 0
    n_long_rows = 0

    for raw_path in raw_paths:
        param, subparam = source_param_subparam(raw_path, input_root=input_root)
        trace_frame = build_region_trace_frame_for_path(raw_path, input_root=input_root, task=task)
        trace_dir = output_root / param / subparam / f"{subparam}-{gait_cfg.MEAN_TRACE_SUFFIX}"
        trace_path = trace_dir / gait_cfg.REGION_TRACE_FILE_TEMPLATE.format(subparam=subparam)
        if overwrite or not trace_path.exists():
            save_pkl(trace_frame, trace_path)
        trace_paths.append(trace_path)
        n_trace_rows += int(len(trace_frame))

        for band, band_frame in trace_frame.groupby("Band", observed=True, sort=False):
            band_dir = trace_dir / str(band)
            band_dir.mkdir(parents=True, exist_ok=True)
            long_frame = trace_frame_to_long_frame(band_frame)
            long_csv = band_dir / gait_cfg.TIMEPOINT_INPUT_FILE
            long_pkl = band_dir / gait_cfg.TIMEPOINT_INPUT_PKL
            if overwrite or not long_csv.exists():
                long_frame.to_csv(long_csv, index=False)
            if overwrite or not long_pkl.exists():
                save_pkl(long_frame, long_pkl)
            input_paths.append(long_csv)
            n_long_rows += int(len(long_frame))

    return GaitCycleExportReport(
        output_root=output_root,
        trace_paths=tuple(trace_paths),
        input_paths=tuple(input_paths),
        n_trace_rows=n_trace_rows,
        n_long_rows=n_long_rows,
    )


def build_region_trace_frame_for_path(
    raw_path: str | Path,
    *,
    input_root: str | Path = gait_cfg.RAW_GAIT_ROOT,
    task: str = gait_cfg.DEFAULT_TASK,
) -> pd.DataFrame:
    """Return Region-level 100-bin traces for one raw gait pickle."""
    raw_path = Path(raw_path).expanduser().resolve()
    input_root = Path(input_root).expanduser().resolve()
    param, subparam = source_param_subparam(raw_path, input_root=input_root)

    frame = load_pkl(raw_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {raw_path}, got {type(frame).__name__}")

    if "Task" in frame.columns:
        frame = frame[frame["Task"].astype(str) == task].reset_index(drop=True)
    if frame.empty:
        return empty_region_trace_frame()

    pre_mode = get_param(derive_cfg.pre_transform_mode_cfg, param, subparam)
    frame = transform_df(frame, mode=pre_mode, **derive_cfg.transform_func_cfg)

    contact_traces = build_contact_trace_frame(frame, param=param, subparam=subparam)
    if contact_traces.empty:
        return empty_region_trace_frame()

    group_cols = region_group_columns(contact_traces)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        region_frame = summarize_df(
            contact_traces,
            group_cols=group_cols,
            value_col="Value",
            mode="mean",
            drop_empty=True,
            align="force",
        )
    region_frame = stringify_metadata_columns(region_frame)
    region_frame = region_frame[
        [col for col in gait_cfg.REGION_TRACE_COLUMNS if col in region_frame.columns]
    ]
    return sort_region_trace_frame(region_frame)


def build_contact_trace_frame(frame: pd.DataFrame, *, param: str, subparam: str) -> pd.DataFrame:
    """Collapse each raw nested value into band-specific 100-bin traces."""
    records: list[dict[str, Any]] = []
    metric_class = metric_class_from_param(param)
    post_mode = get_param(derive_cfg.post_transform_mode_cfg, subparam, "mean")
    eps = float(derive_cfg.transform_func_cfg.get("eps", 1e-120))

    meta_cols = [col for col in frame.columns if col != "Value"]
    for row in frame.to_dict(orient="records"):
        value = row.get("Value")
        for band, trace in iter_band_traces(value, param=param):
            values = pd.to_numeric(trace, errors="coerce").to_numpy(dtype=float)
            values = apply_transform_array(values, mode=post_mode, eps=eps)
            centered = center_trace(values)
            resampled = adjacent_mean_resample_101_to_100(centered)
            if np.isnan(resampled).all():
                continue
            record = {col: row.get(col, pd.NA) for col in meta_cols}
            record.update(
                {
                    "Param": param,
                    "Subparam": subparam,
                    "Metric": subparam,
                    "MetricClass": metric_class,
                    "Band": format_band_label(band),
                    "Value": pd.Series(
                        resampled,
                        index=gait_cfg.CYCLE_BIN_CENTERS,
                        dtype=float,
                    ),
                }
            )
            records.append(record)

    if not records:
        return empty_region_trace_frame()
    return pd.DataFrame.from_records(records)


def iter_band_traces(value: Any, *, param: str) -> Iterable[tuple[str, pd.Series]]:
    """Yield band labels and 101-point traces from one nested value cell."""
    if not isinstance(value, pd.DataFrame):
        return

    matrix = coerce_numeric_matrix(value)
    if matrix.empty:
        return

    if param == "params":
        label_index = matrix.index.map(lambda item: str(item).lower())
        for band in gait_cfg.PARAM_BANDS:
            matches = np.flatnonzero(label_index == band)
            if len(matches) == 0:
                continue
            yield band, matrix.iloc[matches[0]]
        return

    freq_index = pd.to_numeric(pd.Index(matrix.index), errors="coerce")
    if np.isfinite(freq_index.to_numpy(dtype=float)).all():
        for band, (low, high) in DEFAULT_BANDS.items():
            freq_values = freq_index.to_numpy(dtype=float)
            if band == "gamma":
                mask = (freq_values >= low) & (freq_values <= high)
            else:
                mask = (freq_values >= low) & (freq_values < high)
            if not np.any(mask):
                continue
            yield band, matrix.loc[mask].mean(axis=0, skipna=True)
        return

    label_index = matrix.index.map(lambda item: str(item).lower())
    for band in DEFAULT_BANDS:
        matches = np.flatnonzero(label_index == band)
        if len(matches) == 0:
            continue
        yield band, matrix.iloc[matches[0]]


def coerce_numeric_matrix(value: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric copy of one nested trace matrix."""
    out = value.copy()
    out.columns = pd.to_numeric(pd.Index(out.columns), errors="coerce")
    valid_cols = pd.notna(out.columns)
    out = out.loc[:, valid_cols]
    out = out.reindex(sorted(out.columns), axis=1)
    out = out.apply(pd.to_numeric, errors="coerce")
    return out


def center_trace(values: np.ndarray) -> np.ndarray:
    """Center one cycle trace by its full-cycle mean."""
    values = np.asarray(values, dtype=float)
    if values.size != gait_cfg.RAW_CYCLE_POINTS:
        raise ValueError(f"Expected {gait_cfg.RAW_CYCLE_POINTS} cycle points, got {values.size}")
    if np.isnan(values).all():
        return values.copy()
    return values - float(np.nanmean(values))


def adjacent_mean_resample_101_to_100(values: np.ndarray) -> np.ndarray:
    """Resample a 0..100 trace to 0.5..99.5 bin centers by adjacent means."""
    values = np.asarray(values, dtype=float)
    if values.size != gait_cfg.RAW_CYCLE_POINTS:
        raise ValueError(f"Expected {gait_cfg.RAW_CYCLE_POINTS} cycle points, got {values.size}")
    pairs = np.vstack([values[:-1], values[1:]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(pairs, axis=0)
    return np.asarray(out, dtype=float)


def trace_frame_to_long_frame(trace_frame: pd.DataFrame) -> pd.DataFrame:
    """Explode nested Series traces into long-form timepoint rows."""
    long_frame = explode_value_column(
        trace_frame,
        value_col="Value",
        value_name="Value",
        row_coord_name="t_pct",
        drop_missing_cells=True,
        drop_empty=True,
    )
    long_frame["t_pct"] = pd.to_numeric(long_frame["t_pct"], errors="coerce")
    long_frame["Value"] = pd.to_numeric(long_frame["Value"], errors="coerce")
    long_frame = long_frame.dropna(subset=["t_pct", "Value"]).reset_index(drop=True)
    return sort_long_frame(long_frame)


def region_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return grouping columns for Region-level aggregation over Contact."""
    keep = [
        "ID",
        "Subject",
        "Region",
        *gait_cfg.SIDE_RELATED_COLUMNS,
        "Task",
        "Epoch",
        "Param",
        "Subparam",
        "Metric",
        "MetricClass",
        "Band",
    ]
    return [col for col in keep if col in frame.columns]


def sort_region_trace_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort Region-level nested trace rows stably."""
    sort_cols = [
        col
        for col in [
            "Param",
            "Subparam",
            "Metric",
            "Band",
            "Region",
            "ID",
            "Side",
            "Side_cycle",
            "Hemi",
            "Lat",
            "Epoch",
        ]
        if col in frame.columns
    ]
    return frame.sort_values(sort_cols, kind="stable", ignore_index=True)


def sort_long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort long-form timepoint rows stably."""
    sort_cols = [
        col
        for col in [
            "Param",
            "Subparam",
            "Metric",
            "Band",
            "Region",
            "ID",
            "Side",
            "Side_cycle",
            "Hemi",
            "Lat",
            "Epoch",
            "t_pct",
        ]
        if col in frame.columns
    ]
    return frame.sort_values(sort_cols, kind="stable", ignore_index=True)


def source_param_subparam(raw_path: Path, *, input_root: Path) -> tuple[str, str]:
    """Return parameter and subparameter names from one raw gait path."""
    rel = raw_path.relative_to(input_root)
    if len(rel.parts) < 3:
        raise ValueError(f"Expected <param>/<subparam>/<file> under {input_root}, got {raw_path}")
    return rel.parts[0], rel.parts[1]


def metric_class_from_param(param: str) -> str:
    """Return the model family for a raw parameter directory."""
    if param in gait_cfg.CONNECTION_PARAMS:
        return "connectivity"
    if param in gait_cfg.LOCAL_PARAMS:
        return "local"
    raise ValueError(f"Unknown gait parameter family: {param!r}")


def format_band_label(label: Any) -> str:
    """Format band labels to match existing stats directory names."""
    text = str(label)
    lower = text.lower()
    if lower == "beta_low":
        return "Beta_low"
    if lower == "beta_high":
        return "Beta_high"
    if lower == "error_mae":
        return "Error_mae"
    if lower == "gof_rsquared":
        return "Gof_rsquared"
    return lower.capitalize()


def stringify_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert tuple/list metadata values to stable hyphen-joined strings."""
    out = frame.copy()
    for col in out.columns:
        if col == "Value":
            continue
        out[col] = out[col].map(stringify_metadata_value)
    return out


def stringify_metadata_value(value: Any) -> Any:
    """Convert sequence metadata values to readable strings."""
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return value
    return value


def empty_region_trace_frame() -> pd.DataFrame:
    """Return an empty Region trace frame with stable columns."""
    return pd.DataFrame(columns=list(gait_cfg.REGION_TRACE_COLUMNS))
