# Pain Evoked Visualizations

This directory contains visualization workflows for pain-evoked scalar
summaries.

## Current plotting policy

Pain-evoked visualizations do not generate Hemi-specific plots.

Figures that include `Region` as a panel or facet also emit one additional
figure per available region when `Region` is not the x variable. LogForce fit
figures also emit one additional figure per available phase and one additional
figure per available `Region * Phase` combination. Split figures are written
beside the combined figure by appending the selected level labels to the
existing file base, for example:

```text
LogForce-Phase-Region_norm.pdf
Phase-Region-STN_raw.pdf
LogForce-Phase-Region-SNr_norm.pdf
LogForce-Phase-Region-ONS_norm.pdf
LogForce-Phase-Region-STN-ONS_norm.pdf
```

When `Region` is the x variable, no per-region split is generated. Phase as the
x variable remains an unsplit raw comparison. Existing legacy outputs are not
deleted by reruns; only same-name outputs are overwritten.

## LogForce Slope Heatmap

`pain_evoked_logforce_slope_heatmap.py` renders the pain-evoked LogForce slope
summary heatmap from the audit table. It uses
`visualdf.plot_well_heatmap_grid_df`, `3.0 x 3.0 mm` cells, top strips `ONS`
and `OFS`, right strips `SNr`, `STN`, and `SNr-/→STN`, and writes only to the
`corr/logforce_slope` subfolder.

## Phase Signed-LogP Heatmap

`pain_evoked_phase_ons_vs_ofs_signed_logp_heatmap.py` renders the pain-evoked
Phase `ONS - OFS` signed-logP summary heatmap from the stats intermediate
table. It uses `visualdf.plot_well_heatmap_grid_df`, `3.0 x 3.0 mm` cells,
one top strip `ONS vs OFS`, right strips `SNr`, `STN`, and `SNr-/→STN`, and
writes only to the `corr/phase_ons_vs_ofs` subfolder. The workflow also emits
a horizontal region-top companion heatmap with top strips `SNr`, `STN`, and
`SNr-/→STN`, no right strip, waveform parameters on the y-axis, and
electrophysiological metrics on the x-axis.

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain_evoked/pain_evoked_logforce_slope.py
conda run -n LFP python recipes/stnsnr/viz/pain_evoked/pain_evoked_logforce_slope_heatmap.py
conda run -n LFP python recipes/stnsnr/stats/pain_evoked/pain_evoked_phase_ons_vs_ofs_signed_logp.py
conda run -n LFP python recipes/stnsnr/viz/pain_evoked/pain_evoked_phase_ons_vs_ofs_signed_logp_heatmap.py
```
