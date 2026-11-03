# gait_event statistics

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

This directory runs mixed-model statistics for gait-event aligned STN/SNr outputs.

## Model policy

- Scalar raw models test gait phase, region, and lateralized gait-event effects.
- Scalar normalized models test lateralized gait-event effects without hemisphere terms.
- `Hemi` and `Task` are not included as model predictors in these scripts.
- Existing legacy result files whose names contain `Hemi` are not removed by these scripts.
- `gait_event_phase_to_vs_hs_signed_logp.py` summarizes raw `rlmer_model`
  `Phase-Region-Lat` and `Phase-Lat` Tukey outputs for the Phase heatmaps.
  It excludes `Delta`, uses top-strip Lat levels `Ipsi` and `Contra`, and
  encodes positive signed-logP values as `TO > HS`.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/gait_event/single_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/gait_event/double_scalar.R
conda run -n LFP python recipes/stnsnr/stats/gait_event/gait_event_phase_to_vs_hs_signed_logp.py
```

Generated files overwrite existing files with the same path.
