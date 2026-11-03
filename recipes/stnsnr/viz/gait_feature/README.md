# gait_feature visualization

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

This directory renders gait-feature derived STN/SNr figures from the generated summary and model outputs.

## Figure policy

- Scripts do not create figures with `Hemi` or `Task` as plotted variables.
- Figures whose final layout includes `Region` and whose x variable is not `Region` also produce one PDF per Region level.
- For connection plots, `append_region=True` can add a connection-level Region layout; those figures also produce Region-level split PDFs.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/gait_feature/single_feature.py
conda run -n LFP python recipes/stnsnr/viz/gait_feature/gait_feature_gradient_heatmap.py
```

Generated files overwrite existing files with the same path.

## Gait Gradient Heatmaps

`gait_feature_gradient_heatmap.py` renders standardized gait-feature gradient
heatmaps from the stats audit tables under each root's
`corr/gait_standardized_slope` subfolder. Each gait feature is plotted
separately, frequency bands exclude `Delta`, and summary heatmaps use the shared
`visualdf` variable-height well-heatmap grid backend. The `gait_feature` root
keeps burst occupation, burst rate, burst duration, and burst amplitude;
`gait_feature_linear` keeps only burst occupation and burst amplitude. The
renderer also writes additive transposed frequency-summary heatmaps with
frequency rows and metric columns, plus params-by-gradient and params-by-region
heatmaps with top strips `Between`, `Within`.
