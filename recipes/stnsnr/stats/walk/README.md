# STN/SNr Walk Statistics

These scripts fit walking-task mixed models for scalar and MNI-coordinate
features under `${MYLFP_PROJECT_ROOT}/summary/stats/walk`.

## Current Model Factors

`Hemi` is not used as a model predictor or post-hoc grouping variable in the
walk statistics recipes.

- `single_scalar.R` fits raw scalar values with
  `Value ~ Region*Pose*Task + (1|ID)`.
- `single_scalar.R` fits normalized scalar values with
  `Value ~ Region*Pose + (1|ID)` and writes normalized EMM/Tukey files with a
  `_norm` marker in the model result filename.
- `double_scalar.R` fits raw scalar values with `Value ~ Pose*Task + (1|ID)`.
- `double_scalar.R` fits normalized scalar values with
  `Value ~ Pose + (1|ID)` and writes normalized EMM/Tukey files with a
  `_norm` marker in the model result filename.
- `single_xyz.R` and `double_xyz.r` keep the existing coordinate models:
  `Value ~ Region*Pose*(zMX + zMY + zMZ) + (1|ID)`.

The scalar scripts no longer write `Hemi-Pose` or `Hemi-Pose-Region`
EMM/Tukey outputs. Existing legacy files with those names are preserved unless
they are removed outside these scripts.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/walk/single_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/walk/double_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/walk/single_xyz.R
conda run -n LFP Rscript recipes/stnsnr/stats/walk/double_xyz.r
```

## Signed-LogP Heatmap Inputs

`walk_pose_task_pairwise_signed_logp.py` converts raw Task-specific `Pose`
Tukey comparisons from `rlmer_model` outputs into a long signed-logP table for
layered heatmaps. It uses `Pose-Region-Task_tukey.csv` for STN/SNr local
metrics and parameters, and `Pose-Task_tukey.csv` for STN-SNr connectivity
metrics. See `walk_pose_task_pairwise_signed_logp.md` for contrast direction
and output details.

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_pose_task_pairwise_signed_logp.py
```

`walk_task_dual_vs_single_signed_logp.py` converts raw Pose-specific `Task`
Tukey comparisons from `rlmer_model` outputs into a long signed-logP table for
Task secondary-variable heatmaps. It uses `Task-Pose-Region_tukey.csv` for
STN/SNr local metrics and parameters, and `Task-Pose_tukey.csv` for SNr-/→STN
connectivity metrics.

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_task_dual_vs_single_signed_logp.py
```

## LFP-Gait PCA Correlation

`walk_lfp_gait_pca_3m_correlation.py` computes `Task=Single` correlations
between normalized walk LFP pose effects (`Stand/Walk/Turn - Sit`) and
3-month gait PCA distance effects. All outputs are written under
`corr/walk_lfp_gait_pca_3m`.

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_lfp_gait_pca_3m_correlation.py
```

## MNI Standardized Slope

`walk_mni_standardized_slope.py` converts existing `rlmer_model`
MNI-coordinate slope outputs for normalized walk pose effects into
pose-specific standardized slopes. All outputs are written under
`corr/mni_standardized_slope`.

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_mni_standardized_slope.py
```
