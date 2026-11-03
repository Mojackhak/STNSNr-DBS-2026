# Pain MNI Standardized-Slope Intermediate Table

## Purpose

This workflow prepares the standardized MNI-coordinate slope table for pain
phase spatial-gradient heatmaps. It summarizes robust mixed-effects slope
outputs from existing `rlmer_model` coordinate models without refitting any
models.

## Inputs

Only `rlmer_model` slope CSV files are used. Local STN/SNr metrics and
aperiodic parameters read:

```text
zMX-Phase-Region_slope.csv
zMY-Phase-Region_slope.csv
zMZ-Phase-Region_slope.csv
```

Connectivity, GC, and PSI metrics read the same slope filenames from their
`contact_normalized_split` model outputs. The source value tables used for
standardization are:

- `*_contact_normalized.csv` for local metrics and aperiodic parameters;
- `*_contact_normalized_split.csv` for connectivity, GC, and PSI metrics.

## Encoding

The table preserves the raw model slope and adds standardized values:

```text
standardized_slope = slope / SD(Value within Band or parameter, Region, Phase)
```

The selected phases are:

```text
ONS, SUS, OFS, PST
```

Frequency bands are ordered as:

```text
Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
```

The workflow includes raw power, periodic power, four burst metrics,
connectivity, GC, PSI, and the aperiodic parameters `Exponent` and `Offset`.
`tfr/aperiod-mean-scalar` is excluded; aperiodic values are represented only by
the separate parameter rows.

`standardized_slope` is preserved unclipped. `standardized_slope_display` is
clipped to `[-0.7, 0.7]` for heatmap display. Stars are derived from the slope
`q` column:

```text
*   q < 0.05
**  q < 0.01
*** q < 0.001
```

## Output

The intermediate table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/pain/corr/mni_standardized_slope/pain_mni_standardized_slope_values.csv
```

The expected row count is 1632:

```text
Frequency metrics: 11 metrics x 6 bands x 3 coordinates x 2 regions x 4 phases = 1584
Params:             2 params  x 3 coordinates x 2 regions x 4 phases = 48
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain/pain_mni_standardized_slope.py
```
