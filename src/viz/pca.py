#%%

from collections.abc import Sequence
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from cmcrameri import cm

import src.viz.visualdf as vdf


DEFAULT_PCA_COLOR_PALETTE: tuple[str, ...] = (
    "#F2000E",
    "#0E6AAF",
    "#0CA228",
    "#E87000",
    "#884392",
)
DEFAULT_PCA_BOXSIZE_MM: tuple[float, float] = (30.0, 25.0)
DEFAULT_PCA_DPI: int = 600
DEFAULT_PCA_FONT_FAMILY: str = "Arial"
DEFAULT_PCA_AXIS_LABEL_FONTSIZE: int = 7
DEFAULT_PCA_TICK_LABEL_FONTSIZE: int = 6
DEFAULT_PCA_LEGEND_FONTSIZE: int = 7
DEFAULT_PCA_LINE_WIDTH: float = 1.0
DEFAULT_PCA_X_LABEL_OFFSET_MM: float = 5.0
DEFAULT_PCA_Y_LABEL_OFFSET_MM: float = 9.0


def create_ellipse(
    mean: np.ndarray,
    std: np.ndarray,
    color: str | tuple[float, float, float] | tuple[float, float, float, float],
    opacity: float = 0.5,
) -> Ellipse:
    """Create a 2D ellipse patch centered at `mean`.

    Note:
        The visual ellipse diameter is ``2 * std`` on each axis, so `std`
        acts as the half-axis length (radius-like scale).
    """
    mean_arr: np.ndarray = np.asarray(mean, dtype=float)
    std_arr: np.ndarray = np.nan_to_num(
        np.asarray(std, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    width: float = max(2.0 * float(std_arr[0]), 1e-9)
    height: float = max(2.0 * float(std_arr[1]), 1e-9)
    return Ellipse(
        xy=(float(mean_arr[0]), float(mean_arr[1])),
        width=width,
        height=height,
        facecolor=color,
        edgecolor="none",
        alpha=opacity,
        linewidth=1.5,
    )
    

def _format_group_label(group_key: object, group_cols: list[str]) -> str:
    """Format a compact label for one PCA group."""
    if len(group_cols) == 1:
        value = group_key[0] if isinstance(group_key, tuple) else group_key
        return str(value)

    values = group_key if isinstance(group_key, tuple) else (group_key,)
    return " | ".join(str(value) for value in values)


def visualize_PCA(
    df_result: pd.DataFrame,
    explained_variance: np.ndarray | list[float] | tuple[float, ...],
    var1: str,
    var2: str,
    size_sample: float = 50,
    size_center: float = 100,
    color_palette: str | Sequence[str] = DEFAULT_PCA_COLOR_PALETTE,
    opacity: float = 0.7,
    error_bar: Literal["sem", "std"] = "sem",
    boxsize: tuple[float, float] = DEFAULT_PCA_BOXSIZE_MM,
    x_label: str | None = None,
    y_label: str | None = None,
    x_label_offset_mm: float = DEFAULT_PCA_X_LABEL_OFFSET_MM,
    y_label_offset_mm: float = DEFAULT_PCA_Y_LABEL_OFFSET_MM,
    include_global_label_margins: bool = True,
    font_family: str = DEFAULT_PCA_FONT_FAMILY,
    axis_label_fontsize: int = DEFAULT_PCA_AXIS_LABEL_FONTSIZE,
    tick_label_fontsize: int = DEFAULT_PCA_TICK_LABEL_FONTSIZE,
    legend_fontsize: int = DEFAULT_PCA_LEGEND_FONTSIZE,
    legend_marker_size: float | None = None,
    title: str | None = None,
    title_fontsize: int = DEFAULT_PCA_AXIS_LABEL_FONTSIZE,
    line_width: float = DEFAULT_PCA_LINE_WIDTH,
    center_edge_width: float | None = None,
    grid: bool = False,
    grid_alpha: float = 0.3,
    transparent: bool = True,
    dpi: int = DEFAULT_PCA_DPI,
    tight_lim: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    scatter_x: Sequence[float] | float | None = None,
    scatter_y: Sequence[float] | float | None = None,
    scatter_color: str | tuple[float, float, float] | tuple[float, float, float, float] = "black",
    fontsize: int | None = None,
    tick_label_size: int | None = None,
    figure_size: tuple[float, float] | None = None,
    use_symbol_for_greek: bool = False,
    show: bool = True,
) -> Figure:
    """Plot PCA samples and group ellipses on the PC1-PC2 plane.

    Args:
        df_result: Input dataframe that must contain `PC1`, `PC2`, `var1`, and `var2`.
        explained_variance: Explained variance ratio vector from PCA.
        var1: First grouping variable for color/group identity.
        var2: Second grouping variable for color/group identity.
            If equal to `var1`, grouping falls back to a single variable.
        size_sample: Marker size for each sample.
        size_center: Marker size for the group centroid.
        color_palette: Palette name (seaborn/matplotlib) or explicit color list.
        opacity: Ellipse opacity in [0, 1].
        error_bar: Ellipse radius type, `"sem"` or `"std"`.
        boxsize: Inner plotting box size in millimeters: ``(width_mm, height_mm)``.
        x_label: Optional figure-level x label. Defaults to explained variance text.
        y_label: Optional figure-level y label. Defaults to explained variance text.
        x_label_offset_mm: X-label offset from the plotting box in millimeters.
        y_label_offset_mm: Y-label offset from the plotting box in millimeters.
        include_global_label_margins: Whether to reserve margins for figure-level labels.
        font_family: Font family used for labels, ticks, and legend.
        axis_label_fontsize: Font size for figure-level axis labels.
        tick_label_fontsize: Font size for x/y tick labels.
        legend_fontsize: Font size for legend text.
        legend_marker_size: Legend marker size in points. Defaults to the legacy
            `size_sample / 7.2` scaling.
        title: Optional figure title.
        title_fontsize: Font size for the optional title.
        line_width: Width for axes spines, ticks, and PC reference lines.
        center_edge_width: Outline width for group center and custom scatter points.
            Defaults to `line_width`.
        grid: Whether to show the axes grid.
        grid_alpha: Grid line alpha.
        transparent: Whether to make the figure and axes backgrounds transparent.
        dpi: Figure resolution in dots per inch.
        tight_lim: Whether to compute axis bounds from ellipse envelopes.
        xlim: Optional x-axis limits ``(xmin, xmax)``. If provided, overrides auto x-limits.
        ylim: Optional y-axis limits ``(ymin, ymax)``. If provided, overrides auto y-limits.
        scatter_x: Optional custom scatter x-coordinate(s) in PC space (scalar or sequence).
        scatter_y: Optional custom scatter y-coordinate(s) in PC space (scalar or sequence).
        scatter_color: Color for custom scatter point(s).
        fontsize: Legacy alias for axis-label and legend font sizes.
        tick_label_size: Legacy alias for tick-label font size.
        figure_size: Legacy size in inches. If provided, it is converted to `boxsize`.
        use_symbol_for_greek: Draw Greek characters with Symbol while keeping
            all other text in Arial.
        show: Whether to call ``plt.show()`` before returning.

    Returns:
        Matplotlib figure object for further customization or saving.
    """
    # Defensive copy to avoid accidental mutation of caller dataframe.
    explained_var_arr: np.ndarray = np.asarray(explained_variance, dtype=float)
    df_plot: pd.DataFrame = df_result.copy()

    # Validate required inputs early for clearer caller feedback.
    if var1 not in df_plot.columns or var2 not in df_plot.columns:
        raise ValueError(f"var1/var2 must exist in df_result columns: got {var1}, {var2}")
    if "PC1" not in df_plot.columns or "PC2" not in df_plot.columns:
        raise ValueError("df_result must contain PC1 and PC2 columns.")
    if error_bar not in ("sem", "std"):
        raise ValueError("error_bar must be one of {'sem', 'std'}.")
    if explained_var_arr.shape[0] < 2:
        raise ValueError("explained_variance must contain at least two components.")
    if dpi < 1:
        raise ValueError("dpi must be >= 1.")
    if fontsize is not None:
        axis_label_fontsize = fontsize
        legend_fontsize = fontsize
    if tick_label_size is not None:
        tick_label_fontsize = tick_label_size
    if figure_size is not None:
        figure_size_arr = np.asarray(figure_size, dtype=float)
        if figure_size_arr.shape != (2,) or not np.all(np.isfinite(figure_size_arr)):
            raise ValueError("figure_size must be a finite numeric pair: (width_in, height_in).")
        if np.any(figure_size_arr <= 0):
            raise ValueError("figure_size values must be positive.")
        boxsize = (
            float(figure_size_arr[0] * vdf.MM_PER_INCH),
            float(figure_size_arr[1] * vdf.MM_PER_INCH),
        )
    boxsize_mm = vdf._validate_boxsize(boxsize)
    if axis_label_fontsize <= 0 or tick_label_fontsize <= 0 or legend_fontsize <= 0:
        raise ValueError("Font sizes must be positive.")
    if legend_marker_size is None:
        legend_marker_size = size_sample / 7.2
    if legend_marker_size < 0:
        raise ValueError("legend_marker_size must be non-negative.")
    if line_width <= 0:
        raise ValueError("line_width must be positive.")
    if center_edge_width is None:
        center_edge_width = line_width
    if center_edge_width < 0:
        raise ValueError("center_edge_width must be non-negative.")

    default_x_label = f"PC1 ({explained_var_arr[0] * 100:.2f}%)"
    default_y_label = f"PC2 ({explained_var_arr[1] * 100:.2f}%)"
    x_label_text = default_x_label if x_label is None else x_label
    y_label_text = default_y_label if y_label is None else y_label

    fig, axes, layout = vdf._init_box_figure(
        nrows=1,
        ncols=1,
        boxsize_mm=boxsize_mm,
        single_x_label=True,
        single_y_label=True,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=x_label_text,
        y_label_text=y_label_text,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        transparent=transparent,
    )
    ax = axes[0, 0]
    xlim_arr: np.ndarray | None = None
    ylim_arr: np.ndarray | None = None
    if xlim is not None:
        xlim_arr = np.asarray(xlim, dtype=float)
        if xlim_arr.shape != (2,) or not np.all(np.isfinite(xlim_arr)):
            raise ValueError("xlim must be a finite numeric pair: (xmin, xmax).")
    if ylim is not None:
        ylim_arr = np.asarray(ylim, dtype=float)
        if ylim_arr.shape != (2,) or not np.all(np.isfinite(ylim_arr)):
            raise ValueError("ylim must be a finite numeric pair: (ymin, ymax).")

    custom_scatter_x: np.ndarray | None = None
    custom_scatter_y: np.ndarray | None = None
    if (scatter_x is None) != (scatter_y is None):
        raise ValueError("scatter_x and scatter_y must be both provided or both None.")
    if scatter_x is not None and scatter_y is not None:
        if np.isscalar(scatter_x):
            custom_scatter_x = np.asarray([scatter_x], dtype=float)
        else:
            custom_scatter_x = np.asarray(scatter_x, dtype=float)
        if np.isscalar(scatter_y):
            custom_scatter_y = np.asarray([scatter_y], dtype=float)
        else:
            custom_scatter_y = np.asarray(scatter_y, dtype=float)

        custom_scatter_x = np.ravel(custom_scatter_x)
        custom_scatter_y = np.ravel(custom_scatter_y)

        if custom_scatter_x.size == 0 or custom_scatter_y.size == 0:
            raise ValueError("scatter_x/scatter_y cannot be empty when provided.")
        if custom_scatter_x.size == 1 and custom_scatter_y.size > 1:
            custom_scatter_x = np.full(custom_scatter_y.shape, custom_scatter_x.item(), dtype=float)
        if custom_scatter_y.size == 1 and custom_scatter_x.size > 1:
            custom_scatter_y = np.full(custom_scatter_x.shape, custom_scatter_y.item(), dtype=float)
        if custom_scatter_x.size != custom_scatter_y.size:
            raise ValueError("scatter_x and scatter_y must have the same length, or one must be scalar.")
        if not np.all(np.isfinite(custom_scatter_x)) or not np.all(np.isfinite(custom_scatter_y)):
            raise ValueError("scatter_x/scatter_y must be finite numeric values.")

    group_cols: list[str] = [var1] if var1 == var2 else [var1, var2]
    grouped = list(df_plot.groupby(group_cols, observed=True))
    n_groups: int = max(len(grouped), 1)
    palette_colors = sns.color_palette(color_palette, n_colors=n_groups)

    # Track global ellipse envelopes and apply limits only once to avoid jitter.
    pc_min0: np.ndarray = np.array([np.inf, np.inf], dtype=float)
    pc_max0: np.ndarray = np.array([-np.inf, -np.inf], dtype=float)
    legend_handles: list[Line2D] = []

    for i, (group_key, group) in enumerate(grouped):
        color = palette_colors[i % len(palette_colors)]
        mean: np.ndarray = group[["PC1", "PC2"]].mean().to_numpy(dtype=float)
        std: np.ndarray = group[["PC1", "PC2"]].std(ddof=0).to_numpy(dtype=float)
        if error_bar == "sem" and len(group) > 0:
            std /= np.sqrt(len(group))

        pc_min: np.ndarray = mean - std
        pc_max: np.ndarray = mean + std
        pc_min0 = np.minimum(pc_min0, pc_min)
        pc_max0 = np.maximum(pc_max0, pc_max)

        ellipse_patch: Ellipse = create_ellipse(mean, std, color, opacity)
        ax.add_patch(ellipse_patch)

        # Draw individual points first, then overlay group center marker.
        ax.scatter(
            group["PC1"],
            group["PC2"],
            c=[color],
            s=size_sample,
            alpha=0.75,
            edgecolors="none",
            linewidths=0.6,
            zorder=2,
        )
        ax.scatter(
            mean[0],
            mean[1],
            c=[color],
            s=size_center,
            edgecolors="black",
            linewidths=center_edge_width,
            zorder=3,
        )

        label = _format_group_label(group_key, group_cols)

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=legend_marker_size,
                label=label,
                alpha=opacity,
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, title=None, 
                  fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    if custom_scatter_x is not None and custom_scatter_y is not None:
        ax.scatter(
            custom_scatter_x,
            custom_scatter_y,
            c=[scatter_color],
            s=size_center,
            edgecolors="black",
            linewidths=center_edge_width,
            zorder=4,
        )

    if tight_lim and np.all(np.isfinite(pc_min0)) and np.all(np.isfinite(pc_max0)):
        if custom_scatter_x is not None and custom_scatter_y is not None:
            custom_min = np.array([custom_scatter_x.min(), custom_scatter_y.min()], dtype=float)
            custom_max = np.array([custom_scatter_x.max(), custom_scatter_y.max()], dtype=float)
            pc_min0 = np.minimum(pc_min0, custom_min)
            pc_max0 = np.maximum(pc_max0, custom_max)

        x_range: float = pc_max0[0] - pc_min0[0]
        y_range: float = pc_max0[1] - pc_min0[1]
        x_margin: float = 0.1 * (x_range if x_range > 0 else 1.0)
        y_margin: float = 0.1 * (y_range if y_range > 0 else 1.0)
        ax.set_xlim(pc_min0[0] - x_margin, pc_max0[0] + x_margin)
        ax.set_ylim(pc_min0[1] - y_margin, pc_max0[1] + y_margin)
    else:
        pc1_min, pc1_max = df_plot["PC1"].min(), df_plot["PC1"].max()
        pc2_min, pc2_max = df_plot["PC2"].min(), df_plot["PC2"].max()
        if custom_scatter_x is not None and custom_scatter_y is not None:
            pc1_min = min(float(pc1_min), float(custom_scatter_x.min()))
            pc1_max = max(float(pc1_max), float(custom_scatter_x.max()))
            pc2_min = min(float(pc2_min), float(custom_scatter_y.min()))
            pc2_max = max(float(pc2_max), float(custom_scatter_y.max()))

        x_margin = 0.1 * (pc1_max - pc1_min if pc1_max > pc1_min else 1.0)
        y_margin = 0.1 * (pc2_max - pc2_min if pc2_max > pc2_min else 1.0)
        ax.set_xlim(pc1_min - x_margin, pc1_max + x_margin)
        ax.set_ylim(pc2_min - y_margin, pc2_max + y_margin)
    if xlim_arr is not None:
        ax.set_xlim(float(xlim_arr[0]), float(xlim_arr[1]))
    if ylim_arr is not None:
        ax.set_ylim(float(ylim_arr[0]), float(ylim_arr[1]))

    # Draw PC-space reference axes.
    ax.axhline(0.0, color="black", linestyle="--", linewidth=line_width, alpha=0.8, zorder=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=line_width, alpha=0.8, zorder=1)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=tick_label_fontsize, width=line_width)
    if grid:
        ax.grid(True, alpha=grid_alpha)
    else:
        ax.grid(False)
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(line_width)
        spine.set_color("black")

    vdf._draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label_text,
        y_label=y_label_text,
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=True,
        single_y_label=True,
        font_family=font_family,
    )
    vdf.maybe_apply_mixed_font_text(fig, enabled=use_symbol_for_greek)
    if show:
        plt.show()
    return fig



def plot_loading_factors_heatmap(
    loadings: pd.DataFrame,
    explained_variance: np.ndarray | list[float] | tuple[float, ...],
    df_pca_std: pd.DataFrame,
    n_components: int = 2,
    figure_size: tuple[float, float] | None = None,
    dpi: int = 300,
    label_top: str | None = None,
    cellsize: tuple[float, float] = (6.0, 4.0),
    font_family: str = DEFAULT_PCA_FONT_FAMILY,
    cmap: str | matplotlib.colors.Colormap = cm.vik,
    vmin: float = -1.0,
    vmax: float = 1.0,
    label_top_height_mm: float = 4.5,
    label_top_pad_mm: float = 2.0,
    label_top_bg_color: str = "#D7E3E0",
    label_top_text_color: str = "black",
    label_top_fontsize: float = 7.0,
    label_top_fontweight: str = "bold",
    x_label_offset_mm: float = 0.0,
    y_label_offset_mm: float = 18.0,
    cbar_label_offset_mm: float = 8.0,
    axis_label_fontsize: int = DEFAULT_PCA_AXIS_LABEL_FONTSIZE,
    tick_label_fontsize: int = DEFAULT_PCA_TICK_LABEL_FONTSIZE,
    colorbar_label: str = "Loading factor",
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 2.0,
    colorbar_length_mm: float | None = None,
    transparent: bool = True,
    use_symbol_for_greek: bool = False,
    show: bool = False,
) -> Figure:
    """Plot a fixed-cell loading-factor heatmap for the first PCs.

    `df_pca_std` is retained for compatibility with older callers; row labels
    now come directly from the loading matrix index.

    Returns:
        Matplotlib figure object for downstream save/export.
    """
    explained_var_arr: np.ndarray = np.asarray(explained_variance, dtype=float)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if explained_var_arr.shape[0] < n_components:
        raise ValueError("explained_variance has fewer elements than n_components.")
    if loadings.shape[1] < n_components:
        raise ValueError("loadings has fewer columns than n_components.")
    if dpi < 1:
        raise ValueError("dpi must be >= 1.")
    _ = df_pca_std
    if figure_size is not None:
        # Kept for backward-compatible callers. Geometry is now controlled by
        # `cellsize` so loading heatmaps match the visualdf heatmap family.
        _ = figure_size

    df_loadings: pd.DataFrame = loadings.iloc[:, :n_components].copy()
    x_labels: list[str] = [
        f"PC{i + 1}\n({explained_var_arr[i] * 100:.2f}%)"
        for i in range(n_components)
    ]
    df_loadings.columns = x_labels

    fig: Figure = vdf.plot_well_heatmap_df(
        df_value=df_loadings,
        df_p=None,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=label_top,
        font_family=font_family,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmode="sym",
        dpi=dpi,
        cellsize=cellsize,
        label_top_height_mm=label_top_height_mm,
        label_top_pad_mm=label_top_pad_mm,
        label_top_bg_color=label_top_bg_color,
        label_top_text_color=label_top_text_color,
        label_top_fontsize=label_top_fontsize,
        label_top_fontweight=label_top_fontweight,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        grid_lines=None,
        gline_width=0.6,
        gline_alpha=0.9,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        colorbar_label=colorbar_label,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        colorbar_length_mm=colorbar_length_mm,
        show_p=False,
        hide_ns=True,
        p_text_color="black",
        p_text_size=7,
        transparent=transparent,
    )
    vdf._ensure_strip_background_opaque(fig)
    vdf.maybe_apply_mixed_font_text(fig, enabled=use_symbol_for_greek)
    if show:
        plt.show()
    return fig
