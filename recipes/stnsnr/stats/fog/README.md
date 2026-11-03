# STN/SNr FOG Statistics

This publication branch includes the figure-linked subset of the source workflow.
Use `figures/panel_code_map.tsv` for the retained entrypoints and selections.

These scripts fit mixed models for FOG summary statistics under
`${MYLFP_PROJECT_ROOT}/summary/stats/fog`.
They source shared helper code from the current MyLFP checkout at
`${MYLFP_REPO_ROOT}`.

## Current Model Factors

`Hemi` is not used as a model predictor or post-hoc grouping variable in the
FOG statistics recipes.

- `single_scalar.R` fits raw scalar values with
  `Value ~ Region*Pose + (1|ID)`.
- `single_scalar.R` fits normalized scalar values with
  `Value ~ Region + (1|ID)` and writes normalized EMM/Tukey files as
  `Region_norm_*`.
- `double_scalar.R` fits raw scalar values with `Value ~ Pose + (1|ID)`.
- `double_scalar.R` fits normalized scalar values with `Value ~ 1 + (1|ID)`
  for model diagnostics only; `model_omnibus_norm.csv` is written as an empty
  fixed-effect table and no normalized EMM/Tukey files are written.
- `single_xyz.R` and `double_xyz.r` keep the coordinate models:
  `Value ~ Region*(zMX + zMY + zMZ) + (1|ID)`.
- `fog_pose_pairwise_signed_logp.py` summarizes raw `rlmer_model`
  `Pose-Region_tukey.csv` and `Pose_tukey.csv` outputs for the pairwise Pose
  signed-logP heatmaps. It flips source `a - b` estimates so positive values
  mean `b > a` for each `a vs b` panel.

The scalar scripts no longer write `Pose-Hemi`, `Pose-Region-Hemi`, `Hemi`,
or `Hemi-Region` EMM/Tukey outputs. Existing legacy files with those names are
preserved unless they are removed outside these scripts.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/fog/single_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/fog/double_scalar.R
conda run -n LFP python recipes/stnsnr/stats/fog/fog_pose_pairwise_signed_logp.py
```
