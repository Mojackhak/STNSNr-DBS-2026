import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import src.viz.visualdf as vdf


def test_resolve_p_value_from_row_prefers_tukey_over_preferred() -> None:
    row = pd.Series({"p_tukey": 0.04, "p_across": 0.001, "q": 0.0001})

    assert vdf._resolve_p_value_from_row(row, preferred="q") == 0.04


def test_resolve_p_value_series_prefers_tukey_and_falls_back_per_row() -> None:
    df = pd.DataFrame(
        {
            "p_tukey": [0.02, np.nan],
            "p_across": [0.001, 0.03],
            "p.value": [0.5, 0.5],
        }
    )

    resolved = vdf._resolve_p_value_series(df)

    np.testing.assert_allclose(resolved.to_numpy(), np.array([0.02, 0.03]))


def test_pick_p_column_prefers_tukey_over_preferred_when_finite() -> None:
    df = pd.DataFrame({"p_tukey": [0.05], "q": [0.001]})

    assert vdf._pick_p_column(df, preferred="q") == "p_tukey"


def test_pick_p_column_falls_back_to_preferred_when_tukey_is_missing() -> None:
    df = pd.DataFrame({"p_tukey": [np.nan], "q": [0.001]})

    assert vdf._pick_p_column(df, preferred="q") == "q"


def test_categorical_summary_line_stats_mean_sem_and_band_order() -> None:
    df = pd.DataFrame(
        {
            "Band": ["Alpha", "Delta", "Delta", "Theta", "Delta", "Delta"],
            "Region": ["STN", "STN", "STN", "STN", "SNr", "SNr"],
            "Value": [5.0, 1.0, 3.0, 2.0, 10.0, 14.0],
        }
    )
    band_order = ["Delta", "Theta", "Alpha"]

    summary = vdf._categorical_summary_line_stats(
        df,
        x_var="Band",
        line_var="Region",
        value_col="Value",
        x_levels=band_order,
        line_levels=["STN", "SNr"],
    )

    stn_delta = summary[(summary["Region"] == "STN") & (summary["Band"] == "Delta")].iloc[0]
    assert stn_delta["mean"] == 2.0
    assert stn_delta["n"] == 2
    assert stn_delta["sem"] == 1.0

    stn_theta = summary[(summary["Region"] == "STN") & (summary["Band"] == "Theta")].iloc[0]
    assert stn_theta["mean"] == 2.0
    assert stn_theta["n"] == 1
    assert np.isnan(stn_theta["sem"])

    observed_order = summary[summary["Region"] == "STN"]["Band"].astype(str).tolist()
    assert observed_order == band_order


def test_plot_categorical_summary_line_creates_two_region_lines() -> None:
    df = pd.DataFrame(
        {
            "Band": ["Delta", "Theta", "Alpha", "Delta", "Theta", "Alpha"],
            "Region": ["STN", "STN", "STN", "SNr", "SNr", "SNr"],
            "Value": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        }
    )
    band_order = ["Delta", "Theta", "Alpha"]

    fig = vdf.plot_categorical_summary_line(
        df=df,
        x_var="Band",
        line_var="Region",
        value_col="Value",
        x_levels=band_order,
        line_levels=["STN", "SNr"],
        boxsize=(18.0, 12.0),
        dpi=100,
        legend_loc="none",
    )

    ax = fig.axes[0]
    tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert tick_labels == band_order
    plotted_labels = {line.get_label() for line in ax.lines}
    assert {"STN", "SNr"}.issubset(plotted_labels)

    plt.close(fig)


def test_plot_categorical_summary_line_handles_single_connection_region() -> None:
    df = pd.DataFrame(
        {
            "Band": ["Alpha", "Delta", "Theta", "Alpha", "Delta", "Theta"],
            "Region": ["SNr→STN"] * 6,
            "Value": [3.0, 1.0, 2.0, 6.0, 2.0, 4.0],
        }
    )
    band_order = ["Delta", "Theta", "Alpha"]

    fig = vdf.plot_categorical_summary_line(
        df=df,
        x_var="Band",
        line_var="Region",
        value_col="Value",
        x_levels=band_order,
        line_levels=["SNr→STN"],
        boxsize=(30.0, 25.0),
        xtick_rotation=45.0,
        dpi=100,
        legend_loc="none",
    )

    ax = fig.axes[0]
    tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert tick_labels == band_order
    plotted_labels = [line.get_label() for line in ax.lines if not line.get_label().startswith("_")]
    assert plotted_labels == ["SNr→STN"]

    plt.close(fig)


def test_plot_well_heatmap_df_label_top_is_optional() -> None:
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])

    fig = vdf.plot_well_heatmap_df(
        df_value=df,
        boxsize=(12.0, 10.0),
        colorbar_width_mm=0.0,
        dpi=100,
    )

    assert len(fig.axes) == 1

    plt.close(fig)


def test_plot_well_heatmap_df_adds_label_top_strip() -> None:
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])

    fig = vdf.plot_well_heatmap_df(
        df_value=df,
        boxsize=(12.0, 10.0),
        colorbar_width_mm=0.0,
        label_top="STN",
        label_top_height_mm=4.5,
        label_top_pad_mm=2.0,
        dpi=100,
    )

    assert len(fig.axes) == 2
    assert fig.axes[1].texts[0].get_text() == "STN"

    plt.close(fig)


def test_plot_well_heatmap_grid_df_adds_panel_strips_and_colorbar() -> None:
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[[df], [df]],
        boxsize=(12.0, 10.0),
        label_top=["Top"],
        label_right=["First", "Second"],
        label_top_height_mm=4.5,
        label_right_width_mm=4.5,
        strip_pad_mm=2.0,
        colorbar_label="Value",
        dpi=100,
    )

    assert len(fig.axes) == 6
    all_text = [text.get_text() for ax in fig.axes for text in ax.texts]
    assert "Top" in all_text
    assert "First" in all_text
    assert "Second" in all_text

    plt.close(fig)


def test_plot_well_heatmap_grid_df_can_show_x_ticklabels_on_bottom_only() -> None:
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[[df], [df]],
        boxsize=(12.0, 10.0),
        colorbar_width_mm=0.0,
        show_x_ticklabels="bottom",
        dpi=100,
    )

    top_ax, bottom_ax = fig.axes[:2]
    assert len(top_ax.get_xticks()) == 2
    assert [tick.get_text() for tick in top_ax.get_xticklabels()] == ["", ""]
    assert [tick.get_text() for tick in bottom_ax.get_xticklabels()] == ["x", "y"]

    plt.close(fig)


def test_plot_well_heatmap_grid_df_supports_multiple_top_strips_and_left_y_labels() -> None:
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[[df, df, df], [df, df, df]],
        boxsize=(12.0, 10.0),
        label_top=["zSTN-x", "zSTN-y", "zSTN-z"],
        label_right=["Stand", "Walk"],
        label_top_height_mm=4.5,
        label_right_width_mm=4.5,
        strip_pad_mm=2.0,
        colorbar_width_mm=0.0,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        dpi=100,
    )

    panel_axes = fig.axes[:6]
    assert [tick.get_text() for tick in panel_axes[0].get_yticklabels()] == ["A", "B"]
    assert [tick.get_text() for tick in panel_axes[1].get_yticklabels()] == ["", ""]
    assert [tick.get_text() for tick in panel_axes[3].get_xticklabels()] == ["x", "y"]
    assert [tick.get_text() for tick in panel_axes[0].get_xticklabels()] == ["", ""]

    all_text = [text.get_text() for ax in fig.axes for text in ax.texts]
    assert {"zSTN-x", "zSTN-y", "zSTN-z", "Stand", "Walk"}.issubset(all_text)

    plt.close(fig)


def test_plot_well_heatmap_grid_df_supports_variable_cellsize_panel_heights() -> None:
    tall = pd.DataFrame(
        [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]],
        index=["A", "B", "C"],
        columns=["x", "y"],
    )
    short = pd.DataFrame([[0.2, -0.1], [0.5, -0.4]], index=["D", "E"], columns=["x", "y"])

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[[tall, tall], [short, short]],
        cellsize=(3.0, 3.0),
        panel_gap_mm=(2.0, 2.0),
        label_top=["Between", "Within"],
        label_right=["SNr", "SNr-/→STN"],
        label_top_height_mm=4.5,
        label_right_width_mm=8.0,
        strip_pad_mm=2.0,
        colorbar_width_mm=0.0,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        dpi=100,
    )

    panel_axes = fig.axes[:4]
    assert np.isclose(
        panel_axes[0].get_position().height / panel_axes[2].get_position().height,
        1.5,
    )
    assert np.isclose(panel_axes[0].get_position().width, panel_axes[1].get_position().width)
    assert [tick.get_text() for tick in panel_axes[0].get_yticklabels()] == ["A", "B", "C"]
    assert [tick.get_text() for tick in panel_axes[1].get_yticklabels()] == ["", "", ""]
    assert [tick.get_text() for tick in panel_axes[0].get_xticklabels()] == ["", ""]
    assert [tick.get_text() for tick in panel_axes[2].get_xticklabels()] == ["x", "y"]

    all_text = [text.get_text() for ax in fig.axes for text in ax.texts]
    assert {"Between", "Within", "SNr", "SNr-/→STN"}.issubset(all_text)

    plt.close(fig)


def test_plot_well_heatmap_grid_df_rejects_variable_grid_row_mismatch() -> None:
    tall = pd.DataFrame(
        [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]],
        index=["A", "B", "C"],
        columns=["x", "y"],
    )
    short = pd.DataFrame([[0.2, -0.1], [0.5, -0.4]], index=["D", "E"], columns=["x", "y"])

    with pytest.raises(ValueError, match="same grid row"):
        vdf.plot_well_heatmap_grid_df(
            df_values=[[tall, short]],
            cellsize=(3.0, 3.0),
            colorbar_width_mm=0.0,
            dpi=100,
        )


def test_plot_well_heatmap_grid_df_rejects_variable_grid_column_mismatch() -> None:
    two_columns = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])
    three_columns = pd.DataFrame(
        [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]],
        index=["C", "D"],
        columns=["x", "y", "z"],
    )

    with pytest.raises(ValueError, match="same grid column"):
        vdf.plot_well_heatmap_grid_df(
            df_values=[[two_columns], [three_columns]],
            cellsize=(3.0, 3.0),
            colorbar_width_mm=0.0,
            dpi=100,
        )


def test_plot_well_heatmap_grid_df_aligns_p_values_for_significance_stars() -> None:
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])
    p_values = pd.DataFrame(
        [[0.8, 0.8], [0.8, 0.004]],
        index=["B", "A"],
        columns=["y", "x"],
    )

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[[df]],
        df_ps=[[p_values]],
        boxsize=(12.0, 10.0),
        colorbar_width_mm=0.0,
        show_p=True,
        hide_ns=True,
        dpi=100,
    )

    panel_text = [text.get_text() for text in fig.axes[0].texts]
    assert panel_text == ["**"]

    plt.close(fig)


def test_plot_well_heatmap_grid_df_applies_colorbar_ticks() -> None:
    df = pd.DataFrame([[0.1, -0.2], [0.3, 0.4]], index=["A", "B"], columns=["x", "y"])

    fig = vdf.plot_well_heatmap_grid_df(
        df_values=[[df], [df]],
        boxsize=(12.0, 10.0),
        colorbar_label="Value",
        colorbar_ticks=[-1.0, 0.0, 1.0],
        vmin=-1.0,
        vmax=1.0,
        dpi=100,
    )

    np.testing.assert_allclose(fig.axes[-1].get_yticks(), np.array([-1.0, 0.0, 1.0]))

    plt.close(fig)


def test_plot_well_heatmap_df_pdf_keeps_symbol_greek_disabled_by_default(tmp_path) -> None:
    df = pd.DataFrame([[0.1, -0.2]], index=["ΔUPDRS β"], columns=["ρ", "Low-β"])
    output_path = tmp_path / "default_font_heatmap.pdf"

    vdf.set_greek_symbol_font_enabled(False)
    fig = vdf.plot_well_heatmap_df(
        df_value=df,
        boxsize=(12.0, 8.0),
        colorbar_label="Spearman's ρ",
        label_top="Low-β",
        label_top_height_mm=4.5,
        label_top_pad_mm=2.0,
        dpi=100,
    )
    fig.savefig(output_path)
    plt.close(fig)

    pdf_bytes = output_path.read_bytes()
    assert b"Arial" in pdf_bytes
    assert b"Symbol" not in pdf_bytes


def test_plot_well_heatmap_df_pdf_uses_symbol_for_greek_text_when_enabled(tmp_path) -> None:
    df = pd.DataFrame([[0.1, -0.2]], index=["ΔUPDRS β"], columns=["ρ", "Low-β"])
    output_path = tmp_path / "mixed_font_heatmap.pdf"

    fig = vdf.plot_well_heatmap_df(
        df_value=df,
        boxsize=(12.0, 8.0),
        colorbar_label="Spearman's ρ",
        label_top="Low-β",
        label_top_height_mm=4.5,
        label_top_pad_mm=2.0,
        dpi=100,
    )
    vdf.apply_mixed_font_text(fig)
    fig.savefig(output_path)
    plt.close(fig)

    pdf_bytes = output_path.read_bytes()
    assert b"Arial" in pdf_bytes
    assert b"Symbol" in pdf_bytes
