# Pain Phase Pairwise Signed-LogP Intermediate Table

## Purpose

This workflow prepares the signed-logP intermediate table for raw pain `Phase`
pairwise heatmaps. It summarizes robust mixed-effects Tukey evidence from
`rlmer_model` outputs.

## Inputs

Only raw `rlmer_model` Tukey CSV files are used.

Local STN/SNr metrics and aperiodic parameters read `Phase-Region_tukey.csv`.
Connectivity metrics read `Phase_tukey.csv`.

The selected source contrasts are:

```text
BL - ONS
BL - SUS
BL - OFS
BL - PST
ONS - SUS
ONS - OFS
ONS - PST
SUS - OFS
SUS - PST
OFS - PST
```

`Phase-Region-Lat_tukey.csv` and `Phase-Lat_tukey.csv` are not used.
Normalized files are not used.

## Encoding

Panels are labeled in source order as `a vs b`, but the plotted signed value is
computed for `b - a`:

```text
estimate_second_minus_first = -source_estimate_first_minus_second
signed_logp = sign(estimate_second_minus_first) * -log10(p_tukey)
```

Positive values therefore mean the second phase in the panel label is greater
than the first. The raw signed-logP is preserved in the CSV. The display value
is clipped to `[-3, 3]`.

Frequency bands are ordered as:

```text
Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
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
${MYLFP_PROJECT_ROOT}/summary/stats/pain/corr/phase_pairwise/pain_phase_pairwise_signed_logp_values.csv
```

The expected row count is 1180:

```text
SNr local:     7 metrics x 6 bands x 10 comparisons = 420
STN local:     7 metrics x 6 bands x 10 comparisons = 420
Connectivity:  5 metrics x 6 bands x 10 comparisons = 300
Params:        2 params x 2 regions x 10 comparisons = 40
```

The CSV keeps `Aperiodic power` rows for traceability. The heatmap renderer
omits `Aperiodic power` from display.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain/pain_phase_pairwise_signed_logp.py
```
