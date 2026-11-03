# FOG Episode Phase Pairwise Signed-LogP Intermediate Table

## Purpose

This workflow prepares the signed-logP intermediate table for raw
FOG-episode `Phase` pairwise heatmaps. It summarizes robust mixed-effects
Tukey evidence from `rlmer_model` outputs.

## Inputs

Only raw non-Hemi `rlmer_model` Tukey CSV files are used.

Local STN/SNr metrics and aperiodic parameters read `Phase-Region_tukey.csv`.
Connectivity metrics read `Phase_tukey.csv`.

The selected source contrasts are:

```text
BL - Pre
BL - ONS
Pre - ONS
```

Legacy Hemi-specific files such as `Phase-Region-Hemi_tukey.csv` and
`Phase-Hemi_tukey.csv` are not used. Normalized files are not used.

## Encoding

Panels are labeled in source order as `a vs b`, but the plotted signed value is
computed for `b - a`:

```text
estimate_second_minus_first = -source_estimate_first_minus_second
signed_logp = sign(estimate_second_minus_first) * -log10(p_tukey)
```

Positive values therefore mean the second phase in the panel label is greater
than the first. For example, positive values in `BL vs ONS` mean `ONS > BL`.

The raw signed-logP is preserved in the CSV. The display value is clipped to
`[-3, 3]`.

The default frequency bands exclude `Delta` and are ordered as:

```text
Theta, Alpha, Beta_low, Beta_high, Gamma
```

The workflow also writes a second with-Delta branch for heatmaps that need the
full frequency axis. The with-Delta band order is:

```text
Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
```

Burst duration and burst amplitude do not have all bands in the source outputs.
Missing metric-band cells remain absent from the intermediate table and are
left blank in the heatmaps instead of being fabricated.

Stars are derived from `p_tukey`:

```text
*   p_tukey < 0.05
**  p_tukey < 0.01
*** p_tukey < 0.001
```

## Outputs

The default non-Delta intermediate table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/fog_episode/corr/phase_pairwise/fog_episode_phase_pairwise_signed_logp_values.csv
```

The expected row count is 273:

```text
STN local:     93
SNr local:     93
Connectivity:  5 metrics x 5 bands x 3 comparisons = 75
Params:        2 params x 2 regions x 3 comparisons = 12
```

The STN and SNr local totals include five non-Delta bands for metrics with
complete non-Delta output, plus the available non-Delta burst duration and
burst mean bands.

The with-Delta intermediate table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/fog_episode/corr/phase_pairwise/fog_episode_phase_pairwise_signed_logp_values_with_delta.csv
```

The expected with-Delta row count is 318:

```text
STN local:     108
SNr local:     108
Connectivity:  5 metrics x 6 bands x 3 comparisons = 90
Params:        2 params x 2 regions x 3 comparisons = 12
```

The with-Delta STN and SNr local totals include six bands for metrics with
complete output, plus the available burst duration and burst amplitude bands.
Params are included in the with-Delta CSV for traceability, but no separate
with-Delta Params heatmap is rendered because Params have no band axis.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/fog_episode/fog_episode_phase_pairwise_signed_logp.py
```
