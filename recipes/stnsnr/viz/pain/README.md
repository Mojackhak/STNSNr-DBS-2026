# Pain Visualizations

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

This directory contains visualization workflows for pain-period STN/SNr scalar,
spectral, trace, raw time-frequency, coordinate, and volume summaries.

## Current plotting policy

Pain visualizations no longer generate Hemi-specific plots. Scripts skip plot
blocks where `Hemi` would be used as `x_var`, `panel_var`, or `facet_var`.

Figures that include `Region` as a panel or facet also emit one additional
figure per available region when `Region` is not the x variable. Scalar,
spectral, and coordinate figures where `Phase` is a panel or facet and not the
x variable also emit one additional figure per available phase and one
additional figure per available `Region * Phase` combination. Split figures are
written beside the combined figure by appending the selected level labels to the
existing file base, for example:

```text
LogForce-Phase-Region_norm.pdf
Phase-Region-STN_raw.pdf
LogForce-Phase-Region-SNr_norm.pdf
LogForce-Phase-Region-ONS_norm.pdf
LogForce-Phase-Region-STN-ONS_norm.pdf
zMX-Phase-Region-SNr-PST_norm.pdf
```

When `Region` or `Phase` is the x variable, that variable is not split because
the comparison is the plotted x-axis effect. Existing legacy Hemi-named figures
are not deleted by reruns; only same-name outputs are overwritten.

## Signed-LogP Layered Heatmaps

`pain_phase_pairwise_signed_logp_layered_heatmap.py` renders raw Phase pairwise
layered heatmaps from the stats intermediate table. It uses the
`visualdf.plot_well_heatmap_grid_df` backend, `3.0 x 3.0 mm` cells, black
solid grids, and writes only to the `corr/phase_pairwise` subfolder.

Panels are ordered `BL vs ONS`, `BL vs SUS`, `BL vs OFS`, `BL vs PST`,
`ONS vs SUS`, `ONS vs OFS`, `ONS vs PST`, `SUS vs OFS`, `SUS vs PST`,
`OFS vs PST`. For each panel label `a vs b`, positive values mean `b > a`.

The script renders individual heatmaps for `SNr`, `STN`, `SNr-/→STN`, and
Params, plus one combined summary heatmap with top strips `SNr`, `STN`, and
`SNr-/→STN`. The summary excludes Params. All local heatmaps omit the
`Aperiodic power` local metric from display while keeping the intermediate CSV
unchanged.

`pain_lat_ipsi_vs_contra_signed_logp_layered_heatmap.py` renders
BL-normalized Lat Ipsi-vs-Contra layered heatmaps from the stats intermediate
table. Panels are ordered `ONS`, `SUS`, `OFS`, `PST`; positive values mean the
BL-normalized `Ipsi` change is greater than the BL-normalized `Contra` change.
Metric and parameter labels are displayed with a leading `Δ`, for example
`ΔTotal power`. The script renders individual heatmaps for `SNr`, `STN`,
`SNr-/→STN`, and Params, plus one combined summary heatmap with top strips
`SNr`, `STN`, and `SNr-/→STN`. The summary excludes Params and all local
heatmaps omit `Aperiodic power` from display.

It also renders a Params-by-region heatmap with top strips `SNr` and `STN`,
y-axis rows `ΔExponent` and `ΔOffset`, and x-axis columns `ONS`, `SUS`, `OFS`,
and `PST`. `SNr-/→STN` is not included in that Params figure because the
source parameter Tukey rows are region-level only.

## MNI Standardized-Slope Layered Heatmaps

`pain_mni_standardized_slope_layered_heatmap.py` renders pain phase MNI spatial
gradient heatmaps from the standardized-slope stats table. It uses
`visualdf.plot_well_heatmap_grid_df`, `3.0 x 3.0 mm` cells, and writes only to
the `corr/mni_standardized_slope` subfolder.

Panels are ordered `ONS`, `SUS`, `OFS`, `PST`. Frequency figures include
single-coordinate heatmaps and combined `mni_xyz` heatmaps for `SNr` and `STN`
with top strips `zSNr-x/y/z` or `zSTN-x/y/z`. Parameter figures use
`MNI-x/y/z (z-scored)` rows and `ΔExponent`/`ΔOffset` columns, plus a combined
`SNr`/`STN` parameter heatmap and two per-region parameter phase-by-coordinate
heatmaps with x-axis phases. It writes 26 heatmap files. Frequency metric
labels are displayed with a leading `Δ` because the plotted quantities are
BL-relative changes.

`pain_logforce_standardized_slope_layered_heatmap.py` renders BL-normalized
LogForce standardized-slope heatmaps from the stats table. It uses the shared
`visualdf.plot_well_heatmap_grid_df` backend, `3.0 x 3.0 mm` cells, fixed
color limits `[-0.6, 0.6]`, and writes only to the
`corr/logforce_standardized_slope` subfolder. Metric and parameter labels are
displayed with a leading `Δ`. It writes 20 heatmap files, including an
additional Params phase-by-region figure with top strips `SNr` and `STN`.

`pain_lfp_kpps_3m_correlation_heatmap.py` renders pain LFP-KPPS Spearman
correlation heatmaps from the 3-month KPPS correlation workbooks. It writes the
original three-row heatmaps with `Pre-op KPPS` plus two KPPS improvement rows,
and additional `*_delta_percent_corr_heatmap` outputs that display only the
two KPPS improvement rows.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/pain/single_scalar.py
conda run -n LFP python recipes/stnsnr/viz/pain/double_scalar.py
conda run -n LFP python recipes/stnsnr/viz/pain/single_spectral.py
conda run -n LFP python recipes/stnsnr/viz/pain/double_spectral.py
conda run -n LFP python recipes/stnsnr/viz/pain/pain_phase_pairwise_signed_logp_layered_heatmap.py
conda run -n LFP python recipes/stnsnr/viz/pain/pain_lat_ipsi_vs_contra_signed_logp_layered_heatmap.py
conda run -n LFP python recipes/stnsnr/viz/pain/pain_mni_standardized_slope_layered_heatmap.py
conda run -n LFP python recipes/stnsnr/viz/pain/pain_logforce_standardized_slope_layered_heatmap.py
conda run -n LFP python recipes/stnsnr/viz/pain/pain_lfp_kpps_3m_correlation_heatmap.py
```
