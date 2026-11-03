# Rest LAH vs MAH Signed-LogP Heatmaps

## Purpose

This visualization summarizes resting-state `LAH vs MAH` Tukey comparisons from
the robust mixed-effects model results. It creates separate signed-logP heatmaps
for:

- `STN`
- `SNr`
- `SNr-/→STN`
- `Parameters`

The heatmaps are intended to show comparison direction and statistical evidence
in a compact matrix rather than one boxplot per feature.

## Inputs

The script reads Tukey CSV files under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest
```

Only `rlmer_model` CSV files are used for heatmap values and significance
stars. Files whose basename starts with `Region` are ignored.

Group-specific inputs are:

- `STN local`: `rlmer_model/Hemi-Region_tukey.csv`, `Region == "STN"`
- `SNr local`: `rlmer_model/Hemi-Region_tukey.csv`, `Region == "SNr"`
- `SNr-STN connectivity`: `rlmer_model/Hemi_tukey.csv`
- `Parameters`: `params-mean-scalar/<Parameter>/rlmer_model/Hemi-Region_tukey.csv`,
  with `Region` rows plotted as `SNr` and `STN`

Rows must encode the `LAH vs MAH` comparison through `group1/group2`,
`lhs/rhs`, or `contrast == "LAH - MAH"`.

## Matrix Order

The x-axis frequency-band order is:

```text
Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
```

The displayed x-axis labels are:

```text
δ, θ, α, Low-β, High-β, γ
```

The x-axis tick labels are drawn vertically to keep `Low-β` and `High-β`
legible in the compact cell layout.

The local y-axis metric order is:

```text
raw, period, burst-occupation, burst-rate, burst-duration, burst-mean
```

The compact display label for `burst-occupation` is `Burst occupation` with an
uppercase initial letter.

The connectivity y-axis metric order is:

```text
coh, wpli, ciplv, delta_net_gc, psi
```

Display labels are derived from `plugin/default/viz_defaults.py` using
`RAW_YLABEL_TPL`, then unit suffixes are removed for compact row labels.

The parameter heatmap x-axis order is:

```text
Exponent, Offset
```

The parameter heatmap y-axis order is:

```text
SNr, STN
```

The parameter heatmap uses the same fixed colorbar range as the other
signed-logP heatmaps: `[-4, 4]`.

Its top strip uses the compact label `Params`. The parameter panel uses the
same compact `4.0 x 4.0 mm` cell size as the other heatmaps and places the
direction labels farther from the compact colorbar to avoid label overlap.

## Encoding

Each cell is computed from the matching `rlmer_model` Tukey row:

```text
signed_logp = sign(estimate_LAH-MAH) * -log10(p_tukey)
```

Positive values mean `LAH > MAH`; negative values mean `LAH < MAH`.

The heatmap uses the `cm.vik` diverging colormap centered at zero. Displayed
values are clipped to the symmetric range `[-4, 4]`; the companion metadata CSV
keeps the unclipped signed-logP value.
Colorbar ticks are fixed at `-4`, `0`, and `4`.

The colorbar shows the central signed-logP label vertically and adds horizontal
direction labels near the colorbar ends. The direction-label text starts 1 mm
to the right of the main heatmap box. Relative to the previous endpoint
positions, `LAH > MAH` is moved upward by 2 mm and `LAH < MAH` is moved
downward by 2 mm. The vertical signed-logP colorbar label uses a 5 mm label
offset:

```text
LAH > MAH
Signed -log10(p_Tukey)  # vertical, 90 degrees
LAH < MAH
```

Stars use `rlmer p_tukey`:

```text
*   p_tukey < 0.05
**  p_tukey < 0.01
*** p_tukey < 0.001
```

Non-significant cells remain colored by signed-logP. Missing comparisons are
rendered as blank cells.

## Outputs

Outputs are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/corr
```

Generated heatmaps:

- `rest_stn_local_lah_vs_mah_signed_logp_heatmap.pdf`
- `rest_stn_local_lah_vs_mah_signed_logp_heatmap.png`
- `rest_snr_local_lah_vs_mah_signed_logp_heatmap.pdf`
- `rest_snr_local_lah_vs_mah_signed_logp_heatmap.png`
- `rest_connectivity_lah_vs_mah_signed_logp_heatmap.pdf`
- `rest_connectivity_lah_vs_mah_signed_logp_heatmap.png`
- `rest_params_lah_vs_mah_signed_logp_heatmap.pdf`
- `rest_params_lah_vs_mah_signed_logp_heatmap.png`

The companion table is:

- `rest_lah_vs_mah_signed_logp_heatmap_values.csv`

Top strips follow the `visualdf.py` facet-strip pattern: a short strip is placed
directly above the heatmap panel using the configured strip height, padding,
black bold text, and the `plugin/default/viz_defaults.py`
`label_top_bg_color` background. Exports preserve that strip background instead
of making the strip patch transparent.

Heatmap cells use a compact `4.0 x 4.0 mm` size. Direction labels use a
compact font in this layout to avoid overlap with the vertical colorbar label.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/rest/rest_lah_vs_mah_signed_logp_heatmap.py
```
