# STN/SNr FOG Visualizations

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

These scripts generate FOG figures from
`${MYLFP_PROJECT_ROOT}/summary/stats/fog`.

## Current Grouping Variables

`Hemi` is not used as a visualization grouping, panel, facet, or x-axis
variable in the FOG visualization recipes.

- `single_scalar.py` draws raw `Pose-Region` and normalized `Region` plots.
- `double_scalar.py` draws raw `Pose-Region` plots for connection metrics.
- `single_spectral.py` draws raw `Pose-Region` plots.
- `double_spectral.py` draws raw `Pose-Region` plots for connection metrics.
- `single_xyz.py` and `double_xyz.py` draw normalized coordinate-fit plots by
  `Region`.
- `fog_pose_pairwise_signed_logp_layered_heatmap.py` renders raw Pose pairwise
  signed-logP layered heatmaps with panels ordered `Stand vs Walk`,
  `Stand vs FoG`, `Walk vs FoG`. It includes STN local, SNr local,
  `SNr-/→STN` connectivity, and Params heatmaps, plus transposed variants. It
  also renders a Params-by-comparison heatmap with comparison top strips,
  `Exponent`/`Offset` rows, and `SNr`/`STN` columns. Combined summary heatmaps
  for `SNr`, `STN`, and `SNr-/→STN` exclude Params. All heatmaps omit the
  `Aperiodic power` local metric from display while keeping the intermediate
  CSV unchanged.

The scripts no longer generate `Pose-Hemi`, `Pose-Region-Hemi`, `Hemi`, or
`Hemi-Region` figures. Existing legacy files with those names are preserved
unless they are removed outside these scripts.

## Region Split Figures

For any figure involving `Region`, the scripts keep the combined overview
figure and also write split figures for each `Region` when `Region` is not the
x-axis variable.

For example:

```text
Pose-Region-STN_raw.pdf
Pose-Region-SNr_raw.pdf
zMX-Region-STN_norm.pdf
zMX-Region-SNr_norm.pdf
```

Connection metrics append a synthetic `Region` label such as `SNr-STN` or
`SNr-to-STN`, so their split outputs use that connection label.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/fog/single_scalar.py
conda run -n LFP python recipes/stnsnr/viz/fog/double_scalar.py
conda run -n LFP python recipes/stnsnr/viz/fog/double_spectral.py
conda run -n LFP python recipes/stnsnr/viz/fog/fog_pose_pairwise_signed_logp_layered_heatmap.py
```
