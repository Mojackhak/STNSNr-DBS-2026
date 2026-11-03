# gait_feature statistics

This directory runs mixed-model statistics for gait-feature derived STN/SNr outputs.

## Model policy

- Feature models test within-subject and between-subject gait-feature effects.
- Single-region outputs include `Region` as a model factor; double-region connection outputs use connection-level region labeling in visualization.
- `Hemi` and `Task` are not included as model predictors in these scripts.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/gait_feature/single_feature.R
conda run -n LFP Rscript recipes/stnsnr/stats/gait_feature/double_feature.R
conda run -n LFP python recipes/stnsnr/stats/gait_feature/gait_feature_gradient_standardized_slope.py
```

Generated files overwrite existing files with the same path.

## Burst Time Units

`gait_feature` burst duration and burst rate are derived on a 0-100% gait-cycle
axis, then converted with each epoch's `Cycle_duration` before statistics are
run. Burst duration is stored as `ln(seconds)`, and burst rate is stored as
`asinh(Hz)`. `Stride_length` and `Walk_speed` remain gait-feature predictors and
are not used for burst time-unit conversion.

To refresh only the gait-feature burst duration/rate outputs after changing the
percent-to-seconds conversion, use:

```bash
MYLFP_MERGE_CYCLE_STEMS=burst-duration-scalar,burst-rate-scalar \
  conda run -n LFP python pipeline/posetrack/merge_cycle.py

MYLFP_STATS_CFG=plugin.default.stats_posetrack_default \
MYLFP_STATS_STEMS=burst-duration-scalar,burst-rate-scalar \
MYLFP_STATS_SECTIONS=gait_feature \
  conda run -n LFP python pipeline/stats/stats.py

MYLFP_GAIT_FEATURE_SECTIONS=gait_feature \
MYLFP_GAIT_FEATURE_STEMS=burst-duration-scalar,burst-rate-scalar \
  conda run -n LFP Rscript recipes/stnsnr/stats/gait_feature/single_feature.R

MYLFP_GAIT_FEATURE_SECTIONS=gait_feature \
MYLFP_GAIT_FEATURE_STEMS=burst-duration-scalar,burst-rate-scalar \
  conda run -n LFP python recipes/stnsnr/viz/gait_feature/single_feature.py
```

## Gait Gradient Standardized Slopes

`gait_feature_gradient_standardized_slope.py` converts existing `rlmer_model`
gait-feature gradient slopes into standardized-slope audit tables under each
root's `corr/gait_standardized_slope` subfolder. It processes `gait_feature` and
`gait_feature_linear`, renders separate rows for `Walk_speed`, `Stride_length`,
and `Cycle_duration`, and excludes the `Delta` frequency band. The
`gait_feature` root keeps burst occupation, burst rate, burst duration, and
burst amplitude; `gait_feature_linear` keeps only burst occupation and burst
amplitude.
