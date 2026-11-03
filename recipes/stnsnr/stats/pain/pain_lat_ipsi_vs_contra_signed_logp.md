# Pain Lat Ipsi-vs-Contra Signed-LogP Intermediate Table

## Purpose

This workflow prepares the signed-logP intermediate table for pain `Lat`
heatmaps. It summarizes robust mixed-effects Tukey evidence from `rlmer_model`
outputs across BL-normalized pain phases.

## Inputs

Only `rlmer_model` Tukey CSV files from the BL-normalized `Lat-Phase`
analysis are used. In the original R scripts these files are written in the
`NORM` block from `m_norm` / `d_norm`, even though the filenames do not include
`_norm`.

Local STN/SNr metrics and aperiodic parameters read
`Lat-Phase-Region_tukey.csv`. Connectivity metrics read
`Lat-Phase_tukey.csv`.

The selected source contrast is:

```text
Ipsi - Contra
```

The rendered phase panels are:

```text
ONS, SUS, OFS, PST
```

No-Phase Lat files and raw `Phase-Lat` files are not used.

## Encoding

The source Tukey estimate already has the intended direction and is not
flipped:

```text
signed_estimate = source_estimate_ipsi_minus_contra
signed_logp = sign(signed_estimate) * -log10(p_tukey)
```

Positive values mean the BL-normalized `Ipsi` change is greater than the
BL-normalized `Contra` change; negative values mean the opposite. The raw
signed-logP is preserved in the CSV. The display value is clipped to `[-3, 3]`.

Banded metric labels and parameter labels are written with a leading `Δ`, for
example `ΔTotal power`, because the modeled values are relative to BL.

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
${MYLFP_PROJECT_ROOT}/summary/stats/pain/corr/lat_ipsi_vs_contra/pain_lat_ipsi_vs_contra_signed_logp_values.csv
```

The expected row count is 472:

```text
SNr local:     7 metrics x 6 bands x 4 phases = 168
STN local:     7 metrics x 6 bands x 4 phases = 168
Connectivity:  5 metrics x 6 bands x 4 phases = 120
Params:        2 params x 2 regions x 4 phases = 16
```

The CSV keeps `Aperiodic power` rows for traceability. The heatmap renderer
omits `Aperiodic power` from display.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain/pain_lat_ipsi_vs_contra_signed_logp.py
```
