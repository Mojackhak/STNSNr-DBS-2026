# STNSNr PCA Plot Layout

## Purpose

`src.viz.pca.visualize_PCA` creates PC1-PC2 scatter plots with group ellipses
for STNSNr clinical PCA outputs.

The PCA plot uses deterministic millimeter-based layout controls that follow
the visualization defaults in:

```text
plugin/default/viz_defaults.py
```

## Layout Defaults

The inner plotting box is controlled by `boxsize` in millimeters. The default
PCA layout mirrors the shared STNSNr defaults:

```text
boxsize = (30, 25)
dpi = 600
font_family = "Arial"
axis_label_fontsize = 7
tick_label_fontsize = 6
legend_fontsize = 7
line_width = 1
x_label_offset_mm = 5
y_label_offset_mm = 5
transparent = True
```

Axis labels are drawn at the figure level with millimeter offsets from the
inner plotting box. If no custom labels are supplied, the labels include the
explained variance:

```text
PC1 (<variance>%)
PC2 (<variance>%)
```

For the STNSNr clinic PCA recipes, individual sample dots and center/custom dots
are both drawn with Matplotlib scatter size `5`. Center/custom dot outlines
use linewidth `0.2`, independent from the axes and reference-line width. Legend
dots use an independent marker size of `4 pt`.

Zoomed PCA plots hide individual sample dots and keep only center/custom dots.

## Condition Palette

STNSNr clinic plots that use a single column for the five clinical conditions
must use this fixed condition order and palette:

```text
Pre-op                  #F2000E
STN (immediate)         #E87000
STN+SNr (immediate)     #884392
STN (3 m)               #0E6AAF
STN+SNr (3 m)           #0CA228
```

The fixed mapping applies to PCA `Phase` plots, all-condition radar plots using
`Protocol_Phase`, and scale-effect plots using `Condition`. Plots grouped by
separate `Protocol` and `Phase` columns do not use this five-condition palette.
The clinic volume and surface recipes are outside this palette update.
When only a subset of these condition labels is present, colors must still be
selected by condition label rather than by observed group position.
Scale-effect `Condition` plots should also draw only the condition labels
observed in the feature-specific data, using the same canonical order. These
scale-effect plots override the shared y-label offset to `5 mm`.
STNSNr clinic radar plot polygon lines use linewidth `1 pt`.

## Recipe Usage

The STNSNr clinic PCA recipes live in:

```text
recipes/stnsnr/viz/clinic/pca.py
recipes/stnsnr/viz/clinic/scale_pca.py
```

The STNSNr gait and 3-month scale PCA recipes pass shared plotting
configuration derived from `plugin.default.viz_defaults` into both the full
and zoomed PCA plots. The recipes should regenerate only these PCA figure
outputs when targeted validation is needed:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/clinic/pca/gait/pca.pdf
${MYLFP_PROJECT_ROOT}/summary/stats/clinic/pca/gait/pca_zoom.pdf
${MYLFP_PROJECT_ROOT}/summary/stats/clinic/pca/3m/pca.pdf
${MYLFP_PROJECT_ROOT}/summary/stats/clinic/pca/3m/pca_zoom.pdf
```
