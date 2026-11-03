# STN/SNr FOG Episode Statistics

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

These scripts fit mixed models for FOG-episode summary statistics under
`${MYLFP_PROJECT_ROOT}/summary/stats/fog_episode`.
They source shared helper code from the current MyLFP checkout at
`${MYLFP_REPO_ROOT}`.

## Current Model Factors

`Hemi` is not used as a model predictor or post-hoc grouping variable in the
FOG-episode statistics recipes.

- `single_scalar.R` fits raw and normalized scalar values with
  `Value ~ Region*Phase + (1|ID)`.
- `single_scalar.R` writes normalized EMM/Tukey files as
  `Phase-Region_norm_*`.
- `double_scalar.R` fits raw and normalized scalar values with
  `Value ~ Phase + (1|ID)`.
- `double_scalar.R` writes normalized EMM/Tukey files as `Phase_norm_*`.
- `single_xyz.R` and `double_xyz.r` keep the coordinate models:
  `Value ~ Region*Phase*(zMX + zMY + zMZ) + (1|ID)`.

The scalar scripts no longer write `Phase-Hemi`, `Phase-Region-Hemi`,
`Hemi-Phase`, or `Hemi-Region-Phase` EMM/Tukey outputs. Existing legacy files
with those names are preserved unless they are removed outside these scripts.

## Signed-LogP Heatmap Inputs

`fog_episode_phase_pairwise_signed_logp.py` builds raw Phase pairwise
signed-logP intermediate tables for layered heatmaps. It uses only non-Hemi
`rlmer_model` Tukey outputs and writes to the `corr/phase_pairwise` subfolder.
The default table excludes `Delta`; the `_with_delta` table includes bands in
the order `Delta`, `Theta`, `Alpha`, `Beta_low`, `Beta_high`, `Gamma`.

See `fog_episode_phase_pairwise_signed_logp.md` for the contrast direction,
row counts, and output schemas.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/fog_episode/single_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/fog_episode/double_scalar.R
conda run -n LFP python recipes/stnsnr/stats/fog_episode/fog_episode_phase_pairwise_signed_logp.py
```
