# Pain LogForce Standardized-Slope Intermediate Table

## Purpose

This workflow prepares the standardized LogForce slope table for pain
phase-layered heatmaps. It summarizes robust mixed-effects slope outputs from
existing BL-normalized `rlmer_model` scalar models without refitting any models.

## Inputs

Only `rlmer_model` slope CSV files from the normalized `LogForce` analyses are
used.

Local STN/SNr metrics and aperiodic parameters read:

```text
LogForce-Phase-Region_slope.csv
```

Connectivity, GC, and PSI metrics read:

```text
LogForce-Phase_slope.csv
```

The source value tables used for standardization are the BL-normalized
`*_region_normalized.csv` tables.

## Encoding

The table preserves the raw model slope and adds standardized values:

```text
standardized_slope = slope / SD(BL-normalized Value)
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
clipped to `[-0.6, 0.6]` for heatmap display. Stars are derived from the slope
`q` column:

```text
*   q < 0.05
**  q < 0.01
*** q < 0.001
```

## Output

The intermediate table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/pain/corr/logforce_standardized_slope/pain_logforce_standardized_slope_values.csv
```

The expected row count is 424:

```text
SNr local:     6 metrics x 6 bands x 4 phases = 144
STN local:     6 metrics x 6 bands x 4 phases = 144
Connectivity:  5 metrics x 6 bands x 4 phases = 120
Params:        2 params x 2 regions x 4 phases = 16
```

Metric labels and parameter labels are stored without `Δ` in the audit table;
the visualization adds `Δ` as a display-only label because the modeled values
are BL-normalized changes.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain/pain_logforce_standardized_slope.py
```
