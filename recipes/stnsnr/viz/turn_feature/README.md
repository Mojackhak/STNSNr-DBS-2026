# turn_feature visualization

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

This directory renders turn-feature derived STN/SNr figures from the generated
summary and model outputs.

## Figure policy

- `Hemi` and `Task` are not used as plotted variables, panels, or facets.
- Single-region scripts render combined Region-panel fit figures and
  Region-only figures.
- Single-region Region-panel fit figures also produce one PDF per Region
  level.
- Double-region connection scripts keep connection-level Region labeling with
  `append_region=True` and do not add Region to the fitted model.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/turn_feature/single_feature.py
conda run -n LFP python recipes/stnsnr/viz/turn_feature/turn_feature_duration_gradient_heatmap.py
```

Generated files overwrite existing files with the same path. Legacy output
files are not removed by these scripts.

## Duration Gradient Heatmaps

`turn_feature_duration_gradient_heatmap.py` renders standardized duration
gradient heatmaps from the stats audit table under
`corr/duration_standardized_slope`. Frequency figures use region or connection
top strips with `between` and `within` gradient subscripts. The params figure
uses `SNr` and `STN` top strips with `Between` and `Within` rows. All figures
use `3.0 x 3.0 mm` cells, q-value stars, and the shared `visualdf` well-heatmap
grid backend. Displayed LFP metric rows and aperiodic parameter columns use a
leading `Δ` because the plotted values are changes relative to Stand; the stats
audit table labels are unchanged. The additive frequency summary figure uses
top strips `Between` and `Within`, right strips `SNr`, `STN`, and
`SNr-/→STN`, and variable-height panels so local and connectivity metric groups
keep their native row counts.
