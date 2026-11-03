# turn visualization

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

This directory renders turn-derived STN/SNr figures from the generated summary
and model outputs.

## Figure policy

- `Hemi` and `Task` are not used as plotted variables, panels, or facets.
- Single-region scalar and spectral scripts render combined `Lat-Region`
  figures and Region-only figures.
- Single-region `Lat-Region` figures also produce one PDF per Region level.
- Double-region connection scripts keep connection-level Region labeling with
  `append_region=True` and do not render Hemi-specific figures.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/turn/double_scalar.py
conda run -n LFP python recipes/stnsnr/viz/turn/double_spectral.py
conda run -n LFP python recipes/stnsnr/viz/turn/turn_lat_contra_vs_ipsi_signed_logp_heatmap.py
```

Generated files overwrite existing files with the same path. Legacy output
files, including older files whose names contain `Hemi`, are not removed by
these scripts.

## Signed-LogP Heatmaps

`turn_lat_contra_vs_ipsi_signed_logp_heatmap.py` renders the turn `Lat`
main-variable signed-logP heatmaps from the matching stats workflow. It writes
STN local, SNr local, SNr-/→STN connectivity, and Params heatmaps under
`corr/lat_contra_vs_ipsi`, using `3.0 x 3.0 mm` cells with black solid grid
lines. Positive signed-logP values mean `Contra < Ipsi`.
