# Gait Feature Gradient Standardized Slopes

## Purpose

This workflow converts existing gait-feature `rlmer_model` slope tables into
standardized-slope audit tables for:

- `${MYLFP_PROJECT_ROOT}/summary/stats/gait_feature`
- `${MYLFP_PROJECT_ROOT}/summary/stats/gait_feature_linear`

The workflow does not refit models. It reads only existing
`z_Feature_between*` and `z_Feature_within*` slope CSV files.

## Inputs

Each root is processed separately. Gait features are:

```text
Walk_speed, Stride_length, Cycle_duration
```

Frequency-resolved rows use only:

```text
Theta, Alpha, Beta_low, Beta_high, Gamma
```

The `Delta` frequency band is intentionally excluded.

Burst metric scope differs by root:

- `gait_feature`: `Burst occupation`, `asinh burst rate`,
  `ln burst duration`, and `ln burst amplitude` (`burst-mean`);
- `gait_feature_linear`: `Burst occupation` and
  `ln burst amplitude` (`burst-mean`).

Local metric and parameter SDs use `*_contact_normalized.csv`. Connectivity,
Granger causality, and PSI SDs use `*_contact_normalized_split.csv`.

## Standardization

Each model slope is standardized as:

```text
standardized_slope = slope / SD(Value)
```

SD groups are:

- local frequency metrics and parameters: `Feature x Band x Region`;
- connectivity, Granger causality, and PSI metrics: `Feature x Band`.

The unclipped standardized slope is kept for auditing. The heatmap display value
is clipped to `[-0.6, 0.6]`. This symmetric display range covers the current
gait-feature standardized-slope extrema, which are approximately `-0.538` and
`0.477`.

## Outputs

The script writes:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/gait_feature/corr/gait_standardized_slope/gait_feature_gradient_standardized_slope_values.csv
${MYLFP_PROJECT_ROOT}/summary/stats/gait_feature_linear/corr/gait_standardized_slope/gait_feature_linear_gradient_standardized_slope_values.csv
```

Expected row counts are:

- `gait_feature`: 534 rows;
- `gait_feature_linear`: 414 rows.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/gait_feature/gait_feature_gradient_standardized_slope.py
```
