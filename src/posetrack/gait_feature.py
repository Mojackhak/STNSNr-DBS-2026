from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


class GaitEpochLike(Protocol):
    """Protocol for gait epochs produced by src.lfp.warp.event_anchored_linear.gait_warper."""
    side: str
    start_t: float
    end_t: float
    events_t: Mapping[str, float]
    perc_actual: Mapping[str, float]


@dataclass(frozen=True)
class FeatureContext:
    epoch: GaitEpochLike
    epoch_index: int
    side_label: str
    fps: float


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    func: Callable[[pd.DataFrame, FeatureContext], float]
    required_cols: Callable[[FeatureContext], Sequence[str]] | None = None


def _warn(msg: str) -> None:
    print(f"[gait_feature] {msg}")


def _coerce_frame_times(
    df: pd.DataFrame,
    fps: float,
    *,
    frame_col: str | None = None,
    frame0: float = 0.0,
    t0_offset_sec: float = 0.0,
) -> np.ndarray:
    if frame_col is not None and frame_col in df.columns:
        frame_vals = np.asarray(df[frame_col], dtype=float)
    else:
        try:
            frame_vals = np.asarray(df.index, dtype=float)
        except Exception:
            frame_vals = np.arange(len(df), dtype=float)
    return (frame_vals - float(frame0)) / float(fps) + float(t0_offset_sec)


def _keypoint_cols(name: str) -> tuple[str, str, str]:
    return (f"{name}_x", f"{name}_y", f"{name}_z")


def _distance_first_last(df: pd.DataFrame, cols: Sequence[str]) -> float:
    if df.shape[0] == 0:
        return float("nan")
    p0 = df.iloc[0][list(cols)].to_numpy(dtype=float)
    p1 = df.iloc[-1][list(cols)].to_numpy(dtype=float)
    return float(np.linalg.norm(p1 - p0))


def _stride_length_feature(cycle_df: pd.DataFrame, ctx: FeatureContext) -> float:
    cols = _keypoint_cols(f"{ctx.side_label}_ankle")
    return _distance_first_last(cycle_df, cols)


def _cycle_duration_feature(_: pd.DataFrame, ctx: FeatureContext) -> float:
    return float(ctx.epoch.end_t) - float(ctx.epoch.start_t)


def _walk_speed_feature(cycle_df: pd.DataFrame, ctx: FeatureContext) -> float:
    left_cols = _keypoint_cols("left_hip")
    right_cols = _keypoint_cols("right_hip")
    if cycle_df.shape[0] == 0:
        return float("nan")
    left0 = cycle_df.iloc[0][list(left_cols)].to_numpy(dtype=float)
    right0 = cycle_df.iloc[0][list(right_cols)].to_numpy(dtype=float)
    left1 = cycle_df.iloc[-1][list(left_cols)].to_numpy(dtype=float)
    right1 = cycle_df.iloc[-1][list(right_cols)].to_numpy(dtype=float)
    mid0 = (left0 + right0) / 2.0
    mid1 = (left1 + right1) / 2.0
    dist = float(np.linalg.norm(mid1 - mid0))
    if not np.isfinite(dist) or dist == 0:
        return float("nan")
    duration = float(ctx.epoch.end_t) - float(ctx.epoch.start_t)
    if not np.isfinite(duration):
        return float("nan")
    return dist / duration


DEFAULT_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="stride_length",
        func=_stride_length_feature,
        required_cols=lambda ctx: _keypoint_cols(f"{ctx.side_label}_ankle"),
    ),
    FeatureSpec(
        name="cycle_duration",
        func=_cycle_duration_feature,
        required_cols=None,
    ),
    FeatureSpec(
        name="walk_speed",
        func=_walk_speed_feature,
        required_cols=lambda ctx: (
            *_keypoint_cols("left_hip"),
            *_keypoint_cols("right_hip"),
        ),
    ),
)


def extract_gait_cycle_features(
    pose_df: pd.DataFrame,
    gait_cycle_epochs: Mapping[str, Sequence[GaitEpochLike]] | Sequence[GaitEpochLike],
    fps: float,
    *,
    side_map: Mapping[str, str] | None = None,
    features: Iterable[FeatureSpec] | None = None,
    frame_col: str | None = None,
    frame0: float = 0.0,
    t0_offset_sec: float = 0.0,
) -> pd.DataFrame:
    """
    Extract per-cycle gait features from a PoseTrack dataframe.

    Parameters
    ----------
    pose_df
        Per-frame pose dataframe (world coordinates).
    gait_cycle_epochs
        Output from src.lfp.warp.event_anchored_linear.gait_warper. If a mapping,
        uses the "ALL" key by default.
    fps
        PoseTrack frame rate used to convert row index -> seconds.
    side_map
        Mapping from epoch side label to column prefix (default: {"L": "left", "R": "right"}).
    features
        Iterable of FeatureSpec definitions. Defaults to DEFAULT_FEATURES.
    frame_col
        Optional column containing frame numbers (used instead of index when provided).
    frame0
        Frame index treated as time 0 (default 0.0).
    t0_offset_sec
        Optional time offset (sec) added to frame time.
    """
    if not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f"fps must be finite and > 0, got: {fps}")

    if isinstance(gait_cycle_epochs, Mapping):
        epochs = list(gait_cycle_epochs.get("ALL", []))
    else:
        epochs = list(gait_cycle_epochs)

    side_map_use = {"L": "left", "R": "right"}
    if side_map is not None:
        side_map_use.update({str(k): str(v) for k, v in side_map.items()})

    features_use = tuple(features) if features is not None else DEFAULT_FEATURES

    t_sec = _coerce_frame_times(
        pose_df,
        float(fps),
        frame_col=frame_col,
        frame0=frame0,
        t0_offset_sec=t0_offset_sec,
    )

    rows: list[dict[str, object]] = []
    for ei, ep in enumerate(epochs):
        start_t = float(ep.start_t)
        end_t = float(ep.end_t)
        mask = (t_sec >= start_t) & (t_sec <= end_t)
        cycle_df = pose_df.loc[mask]

        side_key = str(ep.side)
        side_label = (
            side_map_use.get(side_key)
            or side_map_use.get(side_key.upper())
            or side_map_use.get(side_key.lower())
            or side_key
        )

        row: dict[str, object] = {
            "epoch": ei,
            "side": ep.side,
            "start_t": start_t,
            "end_t": end_t,
            "events_t": dict(ep.events_t) if ep.events_t is not None else {},
            "perc_actual": dict(ep.perc_actual) if ep.perc_actual is not None else {},
        }

        if cycle_df.shape[0] == 0:
            _warn(f"Epoch {ei} ({ep.side}) has no matched frames; features set to NaN.")
            for spec in features_use:
                row[spec.name] = float("nan")
            rows.append(row)
            continue

        ctx = FeatureContext(epoch=ep, epoch_index=ei, side_label=side_label, fps=float(fps))

        for spec in features_use:
            if spec.required_cols is not None:
                needed = list(spec.required_cols(ctx))
                missing = [c for c in needed if c not in pose_df.columns]
                if missing:
                    _warn(
                        f"Epoch {ei} ({ep.side}) missing columns for {spec.name}: {missing}. "
                        "Returning NaN."
                    )
                    row[spec.name] = float("nan")
                    continue
            row[spec.name] = spec.func(cycle_df, ctx)

        rows.append(row)

    return pd.DataFrame(rows)
