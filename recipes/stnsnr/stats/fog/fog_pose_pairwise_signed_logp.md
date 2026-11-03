# FOG Pose Pairwise Signed-LogP Intermediate Table

## Purpose

This workflow prepares the signed-logP intermediate table for raw FOG `Pose`
pairwise heatmaps. It summarizes robust mixed-effects Tukey evidence from
`rlmer_model` outputs.

## Inputs

Only raw `rlmer_model` Tukey CSV files are used.

Local STN/SNr metrics and aperiodic parameters read `Pose-Region_tukey.csv`.
Connectivity metrics read `Pose_tukey.csv`.

The selected source contrasts are:

```text
Stand - Walk
Stand - FoG
Walk - FoG
```

Legacy Hemi-specific files such as `Pose-Region-Hemi_tukey.csv` and
`Pose-Hemi_tukey.csv` are not used.

## Encoding

Panels are labeled in source order as `a vs b`, but the plotted signed value is
computed for `b - a`:

```text
estimate_second_minus_first = -source_estimate_first_minus_second
signed_logp = sign(estimate_second_minus_first) * -log10(p_tukey)
```

Positive values therefore mean the second pose in the panel label is greater
than the first. For example, positive values in `Stand vs FoG` mean
`FoG > Stand`.

The raw signed-logP is preserved in the CSV. The display value is clipped to
`[-3, 3]`.

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
${MYLFP_PROJECT_ROOT}/summary/stats/fog/corr/pose_pairwise/fog_pose_pairwise_signed_logp_values.csv
```

The expected row count is 354:

```text
STN local:     7 metrics x 6 bands x 3 comparisons = 126
SNr local:     7 metrics x 6 bands x 3 comparisons = 126
Connectivity:  5 metrics x 6 bands x 3 comparisons = 90
Params:        2 params x 2 regions x 3 comparisons = 12
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/fog/fog_pose_pairwise_signed_logp.py
```
