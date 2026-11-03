# MedON vs MedOFF Signed-LogP Intermediate Table

## Purpose

This workflow prepares a signed-logP intermediate table for medication-state
Tukey comparisons under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/med
```

The table is the only input expected by the companion heatmap renderer. It
summarizes robust mixed-effects model evidence for `MedON` versus `MedOFF` in a
compact long-form CSV.

## Inputs

Only `rlmer_model` Tukey CSV files are used.

Local STN/SNr metrics read `Med-Region_tukey.csv` and filter by `Region`:

```text
tfr/raw/raw-mean-scalar
tfr/period/period-mean-scalar
burst/burst/burst-occupation-scalar
burst/burst/burst-rate-scalar
burst/burst/burst-duration-scalar
burst/burst/burst-mean-scalar
```

Connectivity metrics read `Med_tukey.csv`:

```text
conn/coh/coh-mean-scalar
conn/wpli/wpli-mean-scalar
conn/ciplv/ciplv-mean-scalar
gc/delta_net_gc/delta_net_gc-mean-scalar
psi/psi/psi-mean-scalar
```

Aperiodic parameter metrics read `Med-Region_tukey.csv` from:

```text
params/params/params-mean-scalar/Exponent
params/params/params-mean-scalar/Offset
```

## Encoding

The source Tukey rows encode the contrast as:

```text
Off - On
```

For the heatmap convention, positive values must represent `MedON > MedOFF`.
The workflow therefore flips the source estimate before calculating signed-logP:

```text
estimate_on_minus_off = -estimate_off_minus_on
signed_logp = sign(estimate_on_minus_off) * -log10(p_tukey)
```

The raw `signed_logp` is preserved in the CSV. The display value is clipped to
the heatmap range `[-3, 3]`.

Stars are derived from `p_tukey`:

```text
*   p_tukey < 0.05
**  p_tukey < 0.01
*** p_tukey < 0.001
```

## Output

The intermediate table is written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/med/corr/med_medon_vs_medoff_signed_logp_values.csv
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
conda run -n LFP python recipes/stnsnr/stats/med/medon_vs_medoff_signed_logp.py
```
