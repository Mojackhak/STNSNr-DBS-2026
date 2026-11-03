"""Render gait_cycle trace figures with model-specific significant intervals."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[4]))

import plugin.default.stats_gait_cycle_default as gait_cfg  # noqa: E402
import plugin.default.viz_defaults as viz_cfg  # noqa: E402
import src.viz.visualdf as vdf  # noqa: E402
from src.io.pkl_io import load_pkl  # noqa: E402

MODEL_DIR_BY_KEY = {
    "lmer": gait_cfg.LMER_MODEL_DIR,
    "rlmer": gait_cfg.RLMER_MODEL_DIR,
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot gait_cycle Region-level traces and significant intervals."
    )
    parser.add_argument(
        "--output-root",
        default=str(gait_cfg.GAIT_CYCLE_STATS_ROOT),
        help="gait_cycle stats output root.",
    )
    parser.add_argument(
        "--trace-pkl",
        action="append",
        default=None,
        help="Optional Region-level normalized trace pickle. May be provided multiple times.",
    )
    parser.add_argument(
        "--band",
        action="append",
        default=None,
        help="Optional band label to plot. May be provided multiple times.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lmer", "rlmer"],
        choices=sorted(MODEL_DIR_BY_KEY),
        help="Model interval folders to plot.",
    )
    return parser


def discover_trace_paths(output_root: str | Path) -> tuple[Path, ...]:
    """Return every Region-level normalized trace table."""
    root = Path(output_root).expanduser().resolve()
    pattern = f"*-{gait_cfg.MEAN_TRACE_SUFFIX}_region_normalized.pkl"
    return tuple(sorted(path for path in root.rglob(pattern) if path.is_file() and not path.name.startswith("._")))


def infer_param_type(trace_path: Path) -> str:
    """Return the parameter type label used by the default y-axis formatter."""
    return trace_path.parent.name.rsplit("-", 1)[0]


def present_levels(df: pd.DataFrame, column: str) -> list[Any]:
    """Return present levels in plotting order."""
    values = df[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [level for level in values.cat.categories.tolist() if (values == level).any()]
    return values.dropna().drop_duplicates().tolist()


def load_model_intervals(band_dir: Path, model_key: str) -> pd.DataFrame:
    """Load one model's interval table for one band directory."""
    interval_path = (
        band_dir
        / MODEL_DIR_BY_KEY[model_key]
        / gait_cfg.TIMEPOINT_INTERVALS_FILE
    )
    if not interval_path.exists():
        return pd.DataFrame()
    intervals = pd.read_csv(interval_path)
    for column in ("start_pct", "end_pct"):
        if column in intervals.columns:
            intervals[column] = pd.to_numeric(intervals[column], errors="coerce")
    if "span_pct" in intervals.columns:
        intervals["span_pct"] = pd.to_numeric(intervals["span_pct"], errors="coerce")
        intervals = intervals[intervals["span_pct"] >= gait_cfg.MIN_SIGNIFICANT_INTERVAL_SPAN]
    return intervals.dropna(subset=["start_pct", "end_pct"]).reset_index(drop=True)


def interval_segments(row: pd.Series) -> list[tuple[float, float]]:
    """Return one or two x-axis segments for an interval row."""
    start = float(row["start_pct"])
    end = float(row["end_pct"])
    wraps = bool(row.get("wraps_cycle", False)) or start > end
    if wraps:
        return [(start, 100.0), (0.0, end)]
    return [(start, end)]


def model_p_value(row: pd.Series, model_key: str) -> float:
    """Return the interval p-value used for star labels."""
    candidates = [f"min_p_{model_key}", f"p_{model_key}", "p_value", "p"]
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    return float("nan")


def line_color_map(levels: list[Any]) -> dict[str, str]:
    """Return the Region color map used by visualdf for the current line order."""
    palette = list(viz_cfg.palette)
    return {
        str(level): palette[index % len(palette)]
        for index, level in enumerate(levels)
    }


def interval_region(row: pd.Series, region_levels: list[Any]) -> str:
    """Return the plotted Region name for one interval row."""
    region = str(row.get("Region", region_levels[0]))
    region_names = {str(level) for level in region_levels}
    if region not in region_names:
        region = str(region_levels[0])
    return region


def find_data_axis(fig: plt.Figure) -> plt.Axes:
    """Return the primary data axis from a visualdf series figure."""
    for axis in fig.axes:
        if axis.get_lines():
            return axis
    if not fig.axes:
        raise ValueError("Figure has no axes.")
    return fig.axes[0]


def add_interval_shadows(
    fig: plt.Figure,
    intervals: pd.DataFrame,
    region_levels: list[Any],
) -> None:
    """Add Region-colored vertical shadows for significant intervals."""
    if intervals.empty or not region_levels:
        return

    ax = find_data_axis(fig)
    colors = line_color_map(region_levels)
    for _, row in intervals.iterrows():
        color = colors.get(interval_region(row, region_levels), "black")
        for x0, x1 in interval_segments(row):
            x0 = max(0.0, min(100.0, float(x0)))
            x1 = max(0.0, min(100.0, float(x1)))
            if x1 <= x0:
                continue
            ax.axvspan(
                x0,
                x1,
                facecolor=color,
                alpha=gait_cfg.SIGNIFICANCE_SHADOW_ALPHA,
                edgecolor="none",
                linewidth=0,
                zorder=0,
            )


def assign_segment_lanes(segments: list[dict[str, Any]]) -> None:
    """Assign non-overlapping lanes to interval label segments in place."""
    lane_ends_by_direction: dict[str, list[float]] = {"above_0": [], "below_0": []}
    for segment in sorted(segments, key=lambda item: (item["direction"], item["x0"], item["x1"])):
        direction = str(segment["direction"])
        lane_ends = lane_ends_by_direction.setdefault(direction, [])
        lane = None
        for index, lane_end in enumerate(lane_ends):
            if float(segment["x0"]) > lane_end:
                lane = index
                lane_ends[index] = float(segment["x1"])
                break
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(float(segment["x1"]))
        segment["lane"] = lane


def axis_lane_step(lanes: list[int], margin: float) -> float:
    """Return an axes-coordinate lane step that keeps labels inside the plot box."""
    if not lanes:
        return gait_cfg.SIGNIFICANCE_STAR_LANE_STEP
    max_lane = max(int(lane) for lane in lanes)
    if max_lane <= 0:
        return gait_cfg.SIGNIFICANCE_STAR_LANE_STEP
    available = max(0.0, 0.5 - float(margin))
    if available <= 0:
        return gait_cfg.SIGNIFICANCE_STAR_LANE_STEP
    return min(gait_cfg.SIGNIFICANCE_STAR_LANE_STEP, available / max_lane)


def add_interval_stars(
    fig: plt.Figure,
    intervals: pd.DataFrame,
    model_key: str,
    region_levels: list[Any],
) -> None:
    """Add model-specific significance stars near matching Region traces."""
    if intervals.empty or not region_levels:
        return

    ax = find_data_axis(fig)
    colors = line_color_map(region_levels)

    segments: list[dict[str, Any]] = []
    for _, row in intervals.iterrows():
        p_value = model_p_value(row, model_key)
        stars = vdf.p_to_stars(p_value)
        if stars == "n.s.":
            continue

        region = interval_region(row, region_levels)
        color = colors.get(region, "black")
        direction = str(row.get("direction", "above_0"))
        if direction not in {"above_0", "below_0"}:
            continue

        for x0, x1 in interval_segments(row):
            x0 = max(0.0, min(100.0, float(x0)))
            x1 = max(0.0, min(100.0, float(x1)))
            if x1 <= x0:
                continue
            x_mid = (x0 + x1) / 2.0
            segments.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "x_mid": x_mid,
                    "direction": direction,
                    "stars": stars,
                    "color": color,
                }
            )

    if not segments:
        return

    assign_segment_lanes(segments)
    top_lanes = [segment["lane"] for segment in segments if segment["direction"] == "above_0"]
    bottom_lanes = [segment["lane"] for segment in segments if segment["direction"] == "below_0"]
    top_step = axis_lane_step(top_lanes, gait_cfg.SIGNIFICANCE_STAR_TOP_MARGIN)
    bottom_step = axis_lane_step(bottom_lanes, gait_cfg.SIGNIFICANCE_STAR_BOTTOM_MARGIN)
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)

    for segment in segments:
        lane = int(segment["lane"])
        if segment["direction"] == "above_0":
            y = 1.0 - gait_cfg.SIGNIFICANCE_STAR_TOP_MARGIN - top_step * lane
            va = "top"
        else:
            y = gait_cfg.SIGNIFICANCE_STAR_BOTTOM_MARGIN + bottom_step * lane
            va = "bottom"

        ax.text(
            segment["x_mid"],
            max(0.02, min(0.98, y)),
            segment["stars"],
            color=segment["color"],
            fontsize=viz_cfg.font_size,
            fontweight="bold",
            ha="center",
            va=va,
            transform=text_transform,
            zorder=8,
            clip_on=True,
        )


def plot_band_model(
    trace_path: Path,
    df_band: pd.DataFrame,
    band: Any,
    model_key: str,
) -> Path:
    """Plot one band and one model interval set."""
    band_dir = trace_path.parent / str(band)
    save_dir = band_dir / MODEL_DIR_BY_KEY[model_key]
    save_dir.mkdir(parents=True, exist_ok=True)

    intervals = load_model_intervals(band_dir, model_key)
    region_levels = present_levels(df_band, "Region")
    param_type = infer_param_type(trace_path)

    fig = viz_cfg.plot_series_wrapper(
        df=df_band,
        df_type="norm",
        save_dir=save_dir,
        param_type=param_type,
        bd=str(band),
        x_var="Region",
        derive_type="trace",
        section="gait",
        output_filebase="Region",
        config_overrides={"vertical_shadows": None},
    )
    add_interval_shadows(fig, intervals, region_levels)
    add_interval_stars(fig, intervals, model_key, region_levels)
    output_path = save_dir / gait_cfg.FIGURE_FILE
    viz_cfg.save_fig(fig, output_path)
    plt.close(fig)
    return output_path


def plot_trace_table(
    trace_path: str | Path,
    *,
    bands: set[str] | None = None,
    model_keys: tuple[str, ...] = ("lmer", "rlmer"),
) -> tuple[Path, ...]:
    """Plot all requested bands and model interval sets for one trace table."""
    trace_path = Path(trace_path).expanduser().resolve()
    df = load_pkl(trace_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {trace_path}, got {type(df).__name__}")
    if df.empty:
        return tuple()
    if "Band" not in df.columns or "Region" not in df.columns:
        raise KeyError(f"{trace_path} must contain Band and Region columns.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df = viz_cfg.smooth_trace(df.copy())
    output_paths: list[Path] = []
    for band in present_levels(df, "Band"):
        if bands is not None and str(band) not in bands:
            continue
        df_band = df[df["Band"].astype(str) == str(band)].reset_index(drop=True)
        if df_band.empty:
            continue
        for model_key in model_keys:
            output_paths.append(plot_band_model(trace_path, df_band, band, model_key))
    return tuple(output_paths)


def main() -> None:
    """Run the plotting command."""
    args = build_arg_parser().parse_args()
    trace_paths = (
        tuple(Path(path).expanduser().resolve() for path in args.trace_pkl)
        if args.trace_pkl
        else discover_trace_paths(args.output_root)
    )
    if not trace_paths:
        raise FileNotFoundError(
            f"No Region-level normalized trace pickles found under {Path(args.output_root).resolve()}"
        )

    bands = set(args.band) if args.band else None
    model_keys = tuple(args.models)
    for trace_path in trace_paths:
        for output_path in plot_trace_table(trace_path, bands=bands, model_keys=model_keys):
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
