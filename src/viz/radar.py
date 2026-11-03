"""Radar chart utilities for category-wise visualization.

This module provides a reusable matplotlib radar chart function with
optional grouping and p-value asterisk annotations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import src.viz.visualdf as vdf


AggFunc = Literal["mean", "median", "max", "min"] | Callable[[pd.Series], float]


def _resolve_colors(color_palette: str | Sequence[str], n_colors: int) -> list[str]:
    """Resolve plotting colors from a colormap name or explicit color list."""
    if n_colors < 1:
        raise ValueError("n_colors must be >= 1.")

    if isinstance(color_palette, str):
        cmap = plt.get_cmap(color_palette)
        if n_colors == 1:
            return [cmap(0.6)]
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    if not isinstance(color_palette, Sequence) or len(color_palette) == 0:
        raise ValueError("color_palette must be a colormap name or a non-empty list of colors.")

    return [color_palette[i % len(color_palette)] for i in range(n_colors)]


def _p_to_asterisk(
    p_value: float,
    thresholds: tuple[float, float, float],
    symbols: tuple[str, str, str],
) -> str:
    """Convert p-value to an asterisk label using descending significance."""
    if not np.isfinite(p_value):
        return ""

    th_1, th_2, th_3 = thresholds
    sym_1, sym_2, sym_3 = symbols

    if p_value < th_3:
        return sym_3
    if p_value < th_2:
        return sym_2
    if p_value < th_1:
        return sym_1
    return ""


def _infer_order(series: pd.Series) -> list:
    """Infer plotting order from an input series.

    Ordered categoricals use their declared category sequence. Other dtypes keep
    first-seen order from non-null values.
    """
    series_non_null = series.dropna()
    if isinstance(series.dtype, pd.CategoricalDtype) and bool(series.dtype.ordered):
        observed = set(series_non_null)
        return [cat for cat in series.cat.categories if cat in observed]
    return list(pd.unique(series_non_null))


def plot_radar(
    df: pd.DataFrame,
    col_cats: str,
    col_value: str,
    col_group: str | None = None,
    col_p: str | None = None,
    *,
    category_order: list[str] | None = None,
    group_order: list[str] | None = None,
    value_agg: AggFunc = "mean",
    p_agg: AggFunc = "min",
    color_palette: str | Sequence[str] = "tab10",
    fill_alpha: float = 0.15,
    line_width: float = 2.0,
    marker_size: float = 4.0,
    p_thresholds: tuple[float, float, float] = (0.05, 0.01, 0.001),
    p_symbols: tuple[str, str, str] = ("*", "**", "***"),
    p_offset_ratio: float = 0.05,
    category_scale_to: tuple[float, float] | None = None,
    radial_lim: tuple[float, float] | None = None,
    radial_grid_count: int = 4,
    radial_tick_format: Literal["numeric", "percent"] = "percent",
    title: str | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    fontsize: int = 10,
    show_tick_labels: bool = True,
    show_radial_tick_labels: bool | None = None,
    tick_label_size: int = 9,
    asterisk_size: int = 11,
    figure_size: tuple[float, float] = (6.0, 6.0),
    dpi: int = 300,
    use_symbol_for_greek: bool = False,
) -> Figure:
    """Plot a radar chart from tidy tabular data.

    Args:
        df: Source dataframe in long/tidy format.
        col_cats: Category column name (axis labels around radar).
        col_value: Numeric value column name.
        col_group: Optional grouping column name; when None, a single polygon is drawn.
        col_p: Optional p-value column used for category-wise asterisk annotations.
        category_order: Optional explicit category order. If None and `col_cats`
            is an ordered categorical dtype, uses that category order.
        group_order: Optional explicit group order when `col_group` is provided.
            If None and `col_group` is an ordered categorical dtype, uses that
            category order.
        value_agg: Aggregation for repeated rows at the same category/group.
        p_agg: Aggregation for p-values at the same category/group.
        color_palette: Matplotlib colormap name or explicit list of colors.
        fill_alpha: Fill transparency for polygons.
        line_width: Line width of radar polygons.
        marker_size: Marker size for polygon vertices.
        p_thresholds: Thresholds mapping to significance symbols.
        p_symbols: Symbols for p < thresholds[0/1/2].
        p_offset_ratio: Vertical offset ratio for asterisk annotations.
        category_scale_to: Optional per-category min-max rescale target `(low, high)`.
            When set, each category in `col_cats` is scaled independently across groups
            before plotting. Example: `(0.2, 1.0)` maps each category's min/max to
            20%/100% of the radius. By default, the radial axis then spans `[0, high]`
            so `low` is positioned at the expected fractional radius.
        radial_lim: Optional radial limits (ymin, ymax).
        radial_grid_count: Number of radial grid intervals.
        radial_tick_format: Radial tick label format. `"percent"` displays y ticks as `%`.
        title: Optional figure title.
        show_legend: Whether to draw legend when grouped.
        legend_loc: Matplotlib legend location string.
        fontsize: Base font size for title and radial labels.
        show_tick_labels: Whether to display category (angular) tick labels.
        show_radial_tick_labels: Whether to display radial (r-axis) tick labels.
            If None, falls back to `show_tick_labels` for backward compatibility.
        tick_label_size: Font size for category labels.
        asterisk_size: Font size for p-value asterisk markers.
        figure_size: Figure size in inches.
        dpi: Figure resolution.
        use_symbol_for_greek: Draw Greek characters with Symbol while keeping
            all other text in Arial.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If required columns are missing or configuration is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_cols = [col_cats, col_value]
    if col_group is not None:
        required_cols.append(col_group)
    if col_p is not None:
        required_cols.append(col_p)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}.")

    if radial_grid_count < 1:
        raise ValueError("radial_grid_count must be >= 1.")
    if dpi < 1:
        raise ValueError("dpi must be >= 1.")
    if not (0.0 <= fill_alpha <= 1.0):
        raise ValueError("fill_alpha must be in [0, 1].")
    if len(p_thresholds) != 3 or len(p_symbols) != 3:
        raise ValueError("p_thresholds and p_symbols must each contain exactly 3 elements.")
    if radial_tick_format not in ("numeric", "percent"):
        raise ValueError("radial_tick_format must be one of {'numeric', 'percent'}.")
    if category_scale_to is not None:
        scale_low, scale_high = category_scale_to
        if not (np.isfinite(scale_low) and np.isfinite(scale_high)):
            raise ValueError("category_scale_to must contain finite numbers.")
        if scale_low >= scale_high:
            raise ValueError("category_scale_to must satisfy low < high.")
    if show_radial_tick_labels is None:
        show_radial_tick_labels = show_tick_labels
    vdf._configure_plot_fonts("Arial")

    # Use a compact working table and enforce numeric value parsing.
    df_plot = df[required_cols].copy()
    df_plot[col_value] = pd.to_numeric(df_plot[col_value], errors="coerce")
    df_plot = df_plot.dropna(subset=[col_cats, col_value])

    if col_p is not None:
        df_plot[col_p] = pd.to_numeric(df_plot[col_p], errors="coerce")

    if df_plot.empty:
        raise ValueError("No valid rows remain after filtering missing categories/values.")

    if category_order is None:
        categories = _infer_order(df_plot[col_cats])
    else:
        categories = list(category_order)
        unknown_categories = [cat for cat in categories if cat not in set(df_plot[col_cats])]
        if unknown_categories:
            raise ValueError(
                "category_order contains labels not present in data: "
                f"{unknown_categories}."
            )

    if len(categories) < 3:
        raise ValueError("Radar plot requires at least 3 categories.")

    if col_group is None:
        groups = ["All"]
        df_plot = df_plot.assign(__group__="All")
        group_col_name = "__group__"
    else:
        group_col_name = col_group
        if group_order is None:
            groups = _infer_order(df_plot[group_col_name])
        else:
            groups = list(group_order)
            unknown_groups = [g for g in groups if g not in set(df_plot[group_col_name])]
            if unknown_groups:
                raise ValueError(
                    "group_order contains labels not present in data: "
                    f"{unknown_groups}."
                )

    # Aggregate values to one number per (category, group).
    df_value = (
        df_plot.groupby([col_cats, group_col_name], observed=True)[col_value]
        .agg(value_agg)
        .reset_index()
    )
    df_matrix = (
        df_value.pivot(index=col_cats, columns=group_col_name, values=col_value)
        .reindex(index=categories, columns=groups)
    )

    valid_groups = [group for group in groups if df_matrix[group].notna().any()]
    if len(valid_groups) == 0:
        raise ValueError("No valid group contains numeric values after aggregation.")

    df_matrix_plot = df_matrix.copy()
    if category_scale_to is not None:
        scale_low, scale_high = category_scale_to
        scale_span = scale_high - scale_low

        def _rescale_row(row: pd.Series) -> pd.Series:
            out = row.astype(float).copy()
            finite = out[np.isfinite(out)]
            if finite.empty:
                return out

            row_min = float(finite.min())
            row_max = float(finite.max())
            if np.isclose(row_max, row_min):
                # Degenerate category: keep finite entries at the midpoint of target span.
                out.loc[finite.index] = scale_low + 0.5 * scale_span
                return out

            out.loc[finite.index] = scale_low + (finite - row_min) * scale_span / (row_max - row_min)
            return out

        df_matrix_plot.loc[:, valid_groups] = (
            df_matrix[valid_groups].apply(_rescale_row, axis=1)
        )

    colors = _resolve_colors(color_palette=color_palette, n_colors=len(valid_groups))

    # Prepare radar coordinates.
    n_categories = len(categories)
    angles = np.linspace(0.0, 2.0 * np.pi, n_categories, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    if show_tick_labels:
        ax.set_xticklabels(categories, fontsize=tick_label_size)
    else:
        ax.set_xticklabels([""] * len(categories))

    # Plot each group polygon.
    for i, group in enumerate(valid_groups):
        values = df_matrix_plot[group].to_numpy(dtype=float)
        values_closed = np.concatenate([values, [values[0]]])

        ax.plot(
            angles_closed,
            values_closed,
            color=colors[i],
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            label=str(group),
        )
        ax.fill(angles_closed, values_closed, color=colors[i], alpha=fill_alpha)

    finite_values = df_matrix_plot[valid_groups].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot infer radial limits because all values are non-finite.")

    if radial_lim is None:
        if category_scale_to is not None:
            _, scale_high = category_scale_to
            y_min, y_max = 0.0, float(scale_high)
        else:
            val_min = float(np.nanmin(finite_values))
            val_max = float(np.nanmax(finite_values))
            span = val_max - val_min
            pad = (0.1 * span) if span > 0 else max(abs(val_min) * 0.1, 1.0)
            y_min = val_min - pad
            y_max = val_max + pad
    else:
        y_min, y_max = radial_lim
        if y_min >= y_max:
            raise ValueError("radial_lim must satisfy radial_lim[0] < radial_lim[1].")

    # Optional p-value stars are placed per category using aggregated minimum p.
    if col_p is not None:
        df_p = (
            df_plot.groupby(col_cats, observed=True)[col_p]
            .agg(p_agg)
            .reindex(categories)
        )
        max_per_category = df_matrix_plot[valid_groups].max(axis=1).reindex(categories)
        span = y_max - y_min
        # Keep a constant radial gap from each category's max point.
        offset = p_offset_ratio * span

        star_records: list[tuple[float, float, str]] = []
        for idx, category in enumerate(categories):
            p_value = float(df_p.loc[category]) if pd.notna(df_p.loc[category]) else np.nan
            star = _p_to_asterisk(
                p_value=p_value,
                thresholds=p_thresholds,
                symbols=p_symbols,
            )
            if star:
                base = float(max_per_category.loc[category])
                if category_scale_to is not None and radial_lim is None:
                    base = min(base, y_max)
                star_radius = base + offset
                star_records.append((angles[idx], star_radius, star))

        if star_records:
            max_star_radius = max(radius for _, radius, _ in star_records)
            # Keep a fixed normalized radial scale when category_scale_to is used.
            if (category_scale_to is None) and (radial_lim is None) and (max_star_radius > y_max):
                y_max = max_star_radius + 0.5 * offset

            for angle, radius, star in star_records:
                ax.text(
                    angle,
                    radius,
                    star,
                    ha="center",
                    va="center",
                    fontsize=asterisk_size,
                    fontweight="bold",
                    clip_on=False,
                )

    ax.set_ylim(y_min, y_max)
    if category_scale_to is not None and radial_lim is None:
        scale_low, scale_high = category_scale_to
        radial_ticks = np.linspace(scale_low, scale_high, radial_grid_count + 1)
        tick_values = radial_ticks
    else:
        radial_ticks = np.linspace(y_min, y_max, radial_grid_count + 1)
        tick_values = radial_ticks[1:]
    ax.set_yticks(tick_values)
    if show_radial_tick_labels and radial_tick_format == "percent":
        ax.set_yticklabels([f"{tick * 100:.0f}%" for tick in tick_values])
    elif not show_radial_tick_labels:
        ax.set_yticklabels([""] * len(tick_values))
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(alpha=0.35)

    if title is not None:
        ax.set_title(title, fontsize=fontsize + 1, pad=15)

    if show_legend and col_group is not None:
        ax.legend(loc=legend_loc, bbox_to_anchor=(1.15, 1.05), frameon=False, fontsize=fontsize)

    fig.tight_layout()
    vdf.maybe_apply_mixed_font_text(fig, enabled=use_symbol_for_greek)
    return fig


__all__ = ["plot_radar"]
