# Walk MNI Standardized Slope

## Purpose

This workflow converts existing walk MNI-coordinate slope results into
standardized slopes for pose-specific LFP spatial-gradient heatmaps.

The source walk effect is the normalized pose effect stored in the
`contact_normalized` tables:

```text
Stand/Walk/Turn - Sit
```

The standardized slope is:

```text
standardized_slope = slope / SD(Value within Band, Region, Pose)
```

The coordinate predictors in the source models are already z-scored as `zMX`,
`zMY`, and `zMZ`, so the standardized slope is interpreted as:

```text
SD(Value) change per 1 SD MNI coordinate change
```

The script does not refit models. It reuses the existing `rlmer_model`
`zMX-Pose-Region_slope.csv`, `zMY-Pose-Region_slope.csv`, and
`zMZ-Pose-Region_slope.csv` outputs.

## Inputs

Source model outputs are under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk
```

The workflow uses the same 11 frequency-resolved metric families as the med
MNI standardized-slope workflow:

```text
raw, period, burst-occupation, burst-rate, burst-duration, burst-mean,
coh, wpli, ciplv, delta_net_gc, psi
```

It also includes two aperiodic parameters from `params-mean-scalar`:

```text
Exponent, Offset
```

The source `SD(Value)` table is matched to the statistics script that generated
the slope result:

- local metrics follow `single_xyz.R` and use `*_contact_normalized.csv`;
- connectivity, Granger causality, and PSI metrics follow `double_xyz.r` and
  use `*_contact_normalized_split.csv`;
- aperiodic parameters use `params-mean-scalar_contact_normalized.csv`.

The workflow filters `Task == "Single"` when a source value table contains a
`Task` column, matching the coordinate-model scripts. It intentionally excludes
`tfr/aperiod-mean-scalar` so the walk MNI summary matches the med MNI
standardized-slope feature scope.

## Output

All outputs are written under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/mni_standardized_slope
```

The standardized-slope table is:

```text
walk_mni_standardized_slope_values.csv
```

The output keeps raw model values and derived values:

```text
metric, metric_label, band, band_label, region, pose, coordinate, predictor,
slope, lower, upper, p, q, value_sd, value_n, standardized_slope,
standardized_slope_display, star, source_slope_csv, source_sd_csv, sd_source,
feature_group, parameter, parameter_label
```

Displayed standardized slopes are clipped to `[-1, 1]`; the unclipped
`standardized_slope` is preserved for auditing.

The expected table has 1224 rows: 1188 frequency-resolved metric rows and 36
aperiodic parameter rows.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_mni_standardized_slope.py
```
