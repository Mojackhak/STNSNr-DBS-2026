# gait_phase visualization

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

This directory renders gait-phase aligned STN/SNr figures from the generated summary and model outputs.

## Figure policy

- Scripts do not create figures with `Hemi` or `Task` as plotted variables.
- Figures whose final layout includes `Region` and whose x variable is not `Region` also produce one PDF per Region level.
- For connection plots, `append_region=True` can add a connection-level Region layout; those figures also produce Region-level split PDFs.
- Raw and spectral scripts also produce `noDelta` PDFs that show 4-75 Hz while keeping the frequency reference styling from `plugin.default.viz_defaults`.
- Trace scripts are not part of the `noDelta` output set.
- Existing legacy files whose names contain `Hemi` are not removed by these scripts.
- `gait_phase_phase_swing_vs_stance_signed_logp_heatmap.py` renders Phase
  signed-logP heatmaps from the raw `rlmer_model` Tukey summary. It excludes
  `Delta`, uses `Ipsi` and `Contra` top strips, draws `3.0 x 3.0 mm` cells
  with black solid grid lines, includes a region-top Params heatmap with top
  strips `SNr`, `STN` and y rows `Ipsi`, `Contra`, includes a transposed
  Params heatmap with top strips `Ipsi`, `Contra`, y rows `Exponent`, `Offset`,
  and x columns `SNr`, `STN`, and includes a summary heatmap with right strips
  `SNr`, `STN`, `SNr-/→STN`.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/gait_phase/single_scalar.py
conda run -n LFP python recipes/stnsnr/viz/gait_phase/double_scalar.py
conda run -n LFP python recipes/stnsnr/viz/gait_phase/double_trace.py
conda run -n LFP python recipes/stnsnr/viz/gait_phase/gait_phase_phase_swing_vs_stance_signed_logp_heatmap.py
```

Generated files overwrite existing files with the same path.
