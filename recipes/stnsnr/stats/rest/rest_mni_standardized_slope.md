# Rest MNI Standardized Slope

## Purpose

This workflow converts existing resting-state MNI-coordinate slope results into
standardized slopes that can be compared across metric families and aperiodic
parameter effects.

The standardized slope is:

```text
standardized_slope = slope / SD(Value within Band, Region)
```

The coordinate predictors in the source models are already z-scored as
`zMX`, `zMY`, and `zMZ`, so the standardized slope is interpreted as:

```text
SD(Value) change per 1 SD MNI coordinate change
```

The script does not refit models. It reuses the existing `rlmer_model`
`zMX-Region_slope.csv`, `zMY-Region_slope.csv`, and
`zMZ-Region_slope.csv` outputs.

## Inputs

Source model outputs are under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest
```

The workflow uses the same 11 frequency-resolved metric families as the rest
LAH vs MAH heatmaps:

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

- local metrics follow `single_xyz.R` and use `*_contact_raw.csv`;
- connectivity, Granger causality, and PSI metrics follow `double_xyz.r` and
  use `*_contact_split.csv`.
- aperiodic parameters use `params-mean-scalar_contact_raw.csv`.

Both sources compute `SD(Value)` by `Band` and `Region` with sample standard
deviation, matching R `sd()`. For aperiodic parameters, `Band` is the source
column that stores `Exponent` and `Offset`.

## Output

The standardized-slope table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/corr/rest_mni_standardized_slope_values.csv
```

The output keeps raw model values and derived values:

```text
metric, metric_label, band, band_label, region, coordinate, predictor,
slope, lower, upper, p, q, value_sd, value_n, standardized_slope,
standardized_slope_display, star, source_slope_csv, source_sd_csv, sd_source,
feature_group, parameter, parameter_label
```

Displayed standardized slopes are clipped to `[-1, 1]`; the unclipped
`standardized_slope` is preserved for auditing.

The expected table has 408 rows: 396 frequency-resolved metric rows and 12
aperiodic parameter rows.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/rest/rest_mni_standardized_slope.py
```
