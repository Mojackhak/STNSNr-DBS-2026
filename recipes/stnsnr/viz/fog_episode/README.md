# STN/SNr FOG Episode Visualizations

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

These scripts generate FOG-episode figures from
`${MYLFP_PROJECT_ROOT}/summary/stats/fog_episode`.

## Current Grouping Variables

`Hemi` is not used as a visualization grouping, panel, facet, or x-axis
variable in the FOG-episode visualization recipes.

- `single_scalar.py` draws raw and normalized `Phase-Region` plots.
- `double_scalar.py` draws raw and normalized `Phase-Region` plots for
  connection metrics.
- `single_spectral.py` draws raw `Phase-Region` plots.
- `double_spectral.py` draws raw and normalized `Phase-Region` plots for
  connection metrics.
- `single_raw.py`, `double_raw.py`, `single_trace.py`, and `double_trace.py`
  draw normalized non-Hemi summary figures.
- `single_xyz.py` and `double_xyz.py` draw normalized coordinate-fit plots by
  `Phase` and `Region`.

The scripts no longer generate `Phase-Hemi`, `Phase-Region-Hemi`,
`Region-Hemi`, `Hemi`, `Hemi-Phase`, or `Hemi-Region-Phase` figures.
Existing legacy files with those names are preserved unless they are removed
outside these scripts.

## Region Split Figures

For any figure involving `Region`, the scripts keep the combined overview
figure and also write split figures for each `Region` when `Region` is not the
x-axis variable.

For example:

```text
Phase-Region-STN_raw.pdf
Phase-Region-SNr_norm.pdf
zMX-Phase-Region-STN_norm.pdf
zMX-Phase-Region-SNr_norm.pdf
```

Connection metrics append a synthetic `Region` label such as `SNr-STN` or
`SNr-to-STN`, so their split outputs use that connection label.

`fog_multi_exposure.py` and `fog_multi_exposure2.py` are interactive
IPython-style scripts and are not part of the batch rerun.

## Raw Figure Timing Marks

FOG-episode raw time-frequency figures use vertical reference lines at `1 s`
and `2 s`.

## SNr008 Motion-Only Multi-Exposure

`fog_snr008_multi_exposure_motion_only.py` creates the SNr008 video
multi-exposure image without the downstream MNE/raw-file analysis sections.
It estimates a median background from
`${MYLFP_PROJECT_ROOT}/summary/stats/fog/videos/fog_SNr008.mp4`,
renders the non-human region as a grayscale, lightly Gaussian-blurred
reference layer, and composites manually segmented `human` polygons exported
from X-AnyLabeling in
`${MYLFP_PROJECT_ROOT}/summary/stats/fog/videos/img`. The human core
stays in color, the polygon boundary is drawn with `#D7E3E0`, and only the
mask boundary is feathered into the grayscale background.

Run it with:

```bash
conda run -n LFP python recipes/stnsnr/viz/fog_episode/fog_snr008_multi_exposure_motion_only.py
```

The default output is:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/fog/videos/fog_SNr008_multi_exposure_manual_human_outline.png
```

## Signed-LogP Layered Heatmaps

`fog_episode_phase_pairwise_signed_logp_layered_heatmap.py` renders raw Phase
pairwise layered heatmaps from the stats intermediate table. It uses the
`visualdf.plot_well_heatmap_grid_df` backend, `3.0 x 3.0 mm` cells, black
solid grids, and writes only to the `corr/phase_pairwise` subfolder. The
default heatmaps use non-Delta bands; the additional `_with_delta` banded
heatmaps use bands ordered `Delta`, `Theta`, `Alpha`, `Beta_low`, `Beta_high`,
`Gamma`.

Panels are ordered `BL vs Pre`, `BL vs ONS`, `Pre vs ONS`. For each panel
label `a vs b`, positive values mean `b > a`.

The same script also renders combined summary heatmaps for `SNr`, `STN`, and
`SNr-/→STN`; these summary heatmaps exclude Params and keep missing
burst duration and burst amplitude cells blank. The `_with_delta` branch
renders STN local, SNr local, connectivity, and summary heatmaps, but does not
duplicate Params heatmaps because Params have no band axis. The default
non-Delta branch also renders a Params-by-comparison heatmap with comparison
top strips, `Exponent`/`Offset` rows, and `SNr`/`STN` columns. All heatmaps omit
the `Aperiodic power` local metric from display while keeping the intermediate
CSVs unchanged.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/fog_episode/single_raw.py
conda run -n LFP python recipes/stnsnr/viz/fog_episode/double_raw.py
conda run -n LFP python recipes/stnsnr/viz/fog_episode/single_trace.py
conda run -n LFP python recipes/stnsnr/viz/fog_episode/single_scalar.py
conda run -n LFP python recipes/stnsnr/viz/fog_episode/double_scalar.py
conda run -n LFP python recipes/stnsnr/viz/fog_episode/fog_episode_phase_pairwise_signed_logp_layered_heatmap.py
```
