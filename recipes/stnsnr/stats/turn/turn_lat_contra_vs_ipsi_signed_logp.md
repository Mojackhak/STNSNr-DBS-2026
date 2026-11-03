# Turn Lat Contra vs Ipsi Signed-LogP Intermediate Table

## Purpose

This workflow prepares the signed-logP intermediate table for the turn `Lat`
main-variable heatmaps. The table summarizes robust mixed-effects Tukey
evidence for `Contra` versus `Ipsi` from raw turn outputs.

## Inputs

Only `rlmer_model` Tukey CSV files are used.

Local STN/SNr metrics and aperiodic parameters read `Lat-Region_tukey.csv` and
filter by `Region`. Connectivity metrics read `Lat_tukey.csv`.

Legacy Hemi-specific files such as `Lat-Region-Hemi_tukey.csv` and
`Lat-Hemi_tukey.csv` are not used.

## Encoding

The source Tukey rows encode the contrast as:

```text
Ipsi - Contra
```

The heatmap direction follows the source Tukey estimate directly:

```text
signed_estimate = source_estimate_ipsi_minus_contra
signed_logp = sign(signed_estimate) * -log10(p_tukey)
```

Positive values therefore mean `Contra < Ipsi`. The raw `signed_logp` is
preserved in the CSV. The display value is clipped to `[-3, 3]`.

Stars are derived from `p_tukey`:

```text
*   p_tukey < 0.05
**  p_tukey < 0.01
*** p_tukey < 0.001
```

## Output

The intermediate table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/turn/corr/lat_contra_vs_ipsi/turn_lat_contra_vs_ipsi_signed_logp_values.csv
```

The expected row count is 106:

```text
STN local:     6 metrics x 6 bands = 36
SNr local:     6 metrics x 6 bands = 36
Connectivity:  5 metrics x 6 bands = 30
Params:        2 regions x 2 params = 4
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/turn/turn_lat_contra_vs_ipsi_signed_logp.py
```
