# Gait Event Phase TO vs HS Signed-LogP Intermediate Table

## Purpose

This workflow prepares the signed-logP intermediate table for the gait-event
`Phase` main-variable heatmaps. It summarizes robust mixed-effects Tukey
evidence for `TO` versus `HS` from raw gait-event outputs.

## Inputs

Only `rlmer_model` Tukey CSV files are used.

Local STN/SNr metrics and aperiodic parameters read
`Phase-Region-Lat_tukey.csv`. Connectivity metrics read
`Phase-Lat_tukey.csv`.

The workflow uses the main raw model outputs only. Legacy Hemi-specific files
are not used.

## Encoding

The source Tukey rows encode the contrast as:

```text
HS - TO
```

The heatmap direction is flipped before computing signed-logP:

```text
estimate_to_minus_hs = -source_estimate_hs_minus_to
signed_logp = sign(estimate_to_minus_hs) * -log10(p_tukey)
```

Positive values therefore mean `TO > HS`. The raw signed-logP is preserved in
the CSV. The display value is clipped to `[-3, 3]`.

Frequency bands exclude `Delta` and are ordered as:

```text
Theta, Alpha, Beta_low, Beta_high, Gamma
```

Stars are derived from `p_tukey`:

```text
*   p_tukey < 0.05
**  p_tukey < 0.01
*** p_tukey < 0.001
```

## Output

The intermediate table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/gait_event/corr/phase_to_vs_hs/gait_event_phase_to_vs_hs_signed_logp_values.csv
```

The expected row count is 158:

```text
STN local:     5 metrics x 5 bands x 2 Lat = 50
SNr local:     5 metrics x 5 bands x 2 Lat = 50
Connectivity:  5 metrics x 5 bands x 2 Lat = 50
Params:        2 params x 2 regions x 2 Lat = 8
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/gait_event/gait_event_phase_to_vs_hs_signed_logp.py
```
