# Turn Feature Duration Standardized Slopes

## Purpose

This workflow summarizes duration-gradient effects from existing turn-feature
`rlmer_model` slope tables. It does not refit models. The output is a long-form
audit table used by the corresponding two-panel heatmap renderer.

## Inputs

The workflow reads slope CSVs under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/turn_feature
```

Single-region local metrics and aperiodic parameters use:

```text
z_Feature_between-Region_Duration_slope.csv
z_Feature_within-Region_Duration_slope.csv
```

Connectivity, GC, and PSI metrics use:

```text
z_Feature_between_Duration_slope.csv
z_Feature_within_Duration_slope.csv
```

Value standard deviations are computed from normalized `Value` tables filtered
to `Feature == "Duration"`.

## Feature Scope

The frequency-resolved rows include:

```text
raw, period, burst-occupation, burst-rate, burst-duration, burst-mean,
coh, wpli, ciplv, delta_net_gc, psi
```

`tfr/aperiod-mean-scalar` is excluded. Aperiodic `Exponent` and `Offset` are
kept only in the parameter rows.

## Standardization

The displayed effect size is:

```text
standardized_slope = slope / SD(Value)
```

The SD grouping is:

- local metrics and params: `Band x Region`;
- connectivity, GC, and PSI metrics: `Band`.

`standardized_slope_display` is clipped to `[-1, 1]` for visualization, while
the unclipped `standardized_slope` is preserved for auditing.

## Output

The long-form table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/turn_feature/corr/duration_standardized_slope/turn_feature_duration_standardized_slope_values.csv
```

Expected row count is 212:

- 144 local frequency rows;
- 60 connectivity, GC, and PSI rows;
- 8 parameter rows.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/turn_feature/turn_feature_duration_standardized_slope.py
```
