# STN/SNr Visualization Recipes

The scripts in this directory generate STN/SNr manuscript and analysis figures.
Run Python recipes in the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/<subdir>/<script>.py
```

## Font Policy

By default, rendered figure text keeps the repository's original behavior:
Arial is used as the preferred font family and Greek characters are not
substituted with Symbol.

The optional mixed-font mode can be enabled for exports that need it. In that
mode, non-Greek text remains Arial, while Greek characters in labels, tick
labels, annotations, legends, and colorbar labels are drawn with Symbol at
render time.

Python callers can enable the mode per figure with helper arguments such as
`use_symbol_for_greek=True` where supported, by calling
`src.viz.visualdf.apply_mixed_font_text(fig)`, or globally for subsequent
`savefig` calls with `src.viz.visualdf.set_greek_symbol_font_enabled(True)`.
MATLAB surface exports use the `UseSymbolForGreek` name-value option for the
same opt-in behavior.

The underlying label strings are not rewritten. Unicode labels such as `Δ`,
`β`, and `ρ` remain unchanged in dataframe labels, matching logic, filenames,
and exported stats tables; only the figure text artist font changes.

## Label Text

Axis labels and colorbar labels keep a visible word boundary between a
frequency-band label and the metric name. For example, evoked metric labels use
`δ Periodic power` rather than `δPeriodic power`. Formatting helpers should
join optional label parts without leaving extra spaces when one part is empty.
Evoked slope labels use the concise phase names `rise` and `fall`.

For PDF verification with mixed-font mode enabled, render a representative
Greek-containing figure and inspect embedded font names. A valid mixed-font
export should expose both Arial and Symbol font entries. With the default mode,
Symbol is not expected unless another plot element independently uses it.

## Heatmap Significance Markers

Heatmaps that annotate significant cells with p-value or q-value asterisks draw
those asterisks in white for contrast against the diverging heatmap fills.

## Clinic Active-Contact Dots

`recipes/stnsnr/viz/clinic/show_dots.m` reads
`${MYLFP_PROJECT_ROOT}/summary/stats/clinic/coords/scale/scale_contact_only.xlsx`.
That workbook includes a boolean `ULF` column for active contacts. `ULF` is
derived from `${MYLFP_PROJECT_ROOT}/summary/cohort/subj/followup_stimulation.xlsx`
using `Contact Parameters` rows where `Phase` is `3m`, `Protocol` is
`STN+SNr`, and the contact-level `Frequency` is less than `50 Hz`.

The dot rendering uses red `#F2000E` for `ULF = true` contacts and blue
`#0E6AAF` for `ULF = false` contacts.

## Well Heatmap Backends

Single-panel compact heatmaps use `src.viz.visualdf.plot_well_heatmap_df`.
Layered heatmaps that need shared normalization and facet strips can use
`src.viz.visualdf.plot_well_heatmap_grid_df`, which reuses the same well-grid,
tick, spine, and significance-star drawing core. The grid backend supports
multi-row and multi-column panel layouts, including combined xyz heatmaps with
multiple top strips and optional right strips.
