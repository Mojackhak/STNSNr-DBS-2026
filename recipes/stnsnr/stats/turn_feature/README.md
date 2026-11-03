# turn_feature statistics

This directory runs mixed-model statistics for turn-feature derived STN/SNr
outputs.

## Model policy

- Feature models test within-subject and between-subject turn-feature effects.
- `Hemi` and `Task` are not included as model predictors in these scripts.
- Single-region outputs fit
  `Value ~ Region * (z_Feature_within + z_Feature_between) + (1|ID)`.
- Double-region connection outputs fit
  `Value ~ z_Feature_within + z_Feature_between + (1|ID)`.
- Single-region outputs include `Region` EMM/Tukey tables for Region-only
  display.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/turn_feature/single_feature.R
conda run -n LFP Rscript recipes/stnsnr/stats/turn_feature/double_feature.R
conda run -n LFP python recipes/stnsnr/stats/turn_feature/turn_feature_duration_standardized_slope.py
```

Generated files overwrite existing files with the same path. Legacy output
files are not removed by these scripts.

## Duration Gradient Standardized Slopes

`turn_feature_duration_standardized_slope.py` converts existing Duration
`rlmer_model` slope outputs into a standardized-slope audit table under
`corr/duration_standardized_slope`. It uses normalized `Value` SDs from the
matching contact tables, excludes `tfr/aperiod-mean-scalar`, and keeps
`Exponent` and `Offset` as parameter rows.
