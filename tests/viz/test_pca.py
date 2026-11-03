import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import src.viz.pca as pca_viz


def _sample_pca_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PC1": [-1.0, -0.7, 0.6, 0.9],
            "PC2": [-0.5, -0.2, 0.2, 0.5],
            "Phase": pd.Categorical(
                ["Pre-op", "Pre-op", "STN", "STN"],
                categories=["Pre-op", "STN"],
                ordered=True,
            ),
        }
    )


def test_visualize_pca_uses_requested_boxsize() -> None:
    boxsize = (42.0, 31.0)

    fig = pca_viz.visualize_PCA(
        _sample_pca_scores(),
        [0.6, 0.25],
        var1="Phase",
        var2="Phase",
        boxsize=boxsize,
        show=False,
    )

    ax = fig.axes[0]
    fig.canvas.draw()
    pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()

    assert pos.width * fig_w == pytest.approx(boxsize[0] / 25.4)
    assert pos.height * fig_h == pytest.approx(boxsize[1] / 25.4)
    plt.close(fig)


def test_visualize_pca_draws_custom_global_labels() -> None:
    fig = pca_viz.visualize_PCA(
        _sample_pca_scores(),
        [0.6, 0.25],
        var1="Phase",
        var2="Phase",
        x_label="Custom PC1",
        y_label="Custom PC2",
        axis_label_fontsize=9,
        show=False,
    )

    label_texts = {text.get_text(): text for text in fig.texts}

    assert "Custom PC1" in label_texts
    assert "Custom PC2" in label_texts
    assert label_texts["Custom PC1"].get_fontsize() == pytest.approx(9)
    assert label_texts["Custom PC2"].get_fontsize() == pytest.approx(9)
    plt.close(fig)


def test_visualize_pca_renders_custom_scatter_point() -> None:
    fig = pca_viz.visualize_PCA(
        _sample_pca_scores(),
        [0.6, 0.25],
        var1="Phase",
        var2="Phase",
        scatter_x=0.25,
        scatter_y=-0.5,
        show=False,
    )

    offsets = np.vstack(
        [
            np.asarray(collection.get_offsets())
            for collection in fig.axes[0].collections
            if hasattr(collection, "get_offsets") and len(collection.get_offsets()) > 0
        ]
    )

    assert np.any(np.all(np.isclose(offsets, [0.25, -0.5]), axis=1))
    plt.close(fig)


def test_visualize_pca_applies_center_edge_width() -> None:
    fig = pca_viz.visualize_PCA(
        _sample_pca_scores(),
        [0.6, 0.25],
        var1="Phase",
        var2="Phase",
        size_center=100,
        center_edge_width=0.2,
        show=False,
    )

    center_widths = [
        float(collection.get_linewidths()[0])
        for collection in fig.axes[0].collections
        if hasattr(collection, "get_sizes")
        and len(collection.get_sizes()) > 0
        and collection.get_sizes()[0] == pytest.approx(100)
    ]

    assert center_widths
    assert all(width == pytest.approx(0.2) for width in center_widths)
    plt.close(fig)


def test_visualize_pca_applies_legend_marker_size() -> None:
    fig = pca_viz.visualize_PCA(
        _sample_pca_scores(),
        [0.6, 0.25],
        var1="Phase",
        var2="Phase",
        size_sample=5,
        legend_marker_size=4,
        show=False,
    )

    legend = fig.axes[0].get_legend()
    if hasattr(legend, "legend_handles"):
        handles = legend.legend_handles
    else:
        handles = legend.legendHandles
    marker_sizes = [handle.get_markersize() for handle in handles]

    assert marker_sizes
    assert all(size == pytest.approx(4) for size in marker_sizes)
    plt.close(fig)


def test_visualize_pca_single_grouping_label_is_not_split() -> None:
    fig = pca_viz.visualize_PCA(
        _sample_pca_scores(),
        [0.6, 0.25],
        var1="Phase",
        var2="Phase",
        show=False,
    )

    legend = fig.axes[0].get_legend()
    labels = [text.get_text() for text in legend.get_texts()]

    assert labels == ["Pre-op", "STN"]
    plt.close(fig)
