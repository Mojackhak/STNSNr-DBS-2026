# STNSNr Rest Band Line Summary

## Purpose

`recipes/stnsnr/viz/rest/burst_summary_scalar.py` generates band-wise line
summaries for resting-state STN/SNr burst scalar features. Each figure shows the
subject-level mean `Value` with SEM error bars across frequency bands.

`recipes/stnsnr/viz/rest/psi_summary_scalar.py` generates the same band-wise
line summary for resting-state PSI.

## Inputs

The recipe reads the raw subject CSV files under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/burst/burst
```

Expected feature folders and files:

```text
burst-occupation-scalar/burst-occupation-scalar_subject_raw.csv
burst-rate-scalar/burst-rate-scalar_subject_raw.csv
burst-duration-scalar/burst-duration-scalar_subject_raw.csv
burst-mean-scalar/burst-mean-scalar_subject_raw.csv
```

Each CSV must be long-form and include `Band`, `Region`, and `Value`. The plot
groups rows by `Region` and `Band`, draws STN and SNr as separate lines, and
computes SEM from finite subject-level values only.

The PSI recipe reads:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/psi/psi/psi-mean-scalar/psi-mean-scalar_subject_raw.csv
```

The PSI CSV must also be long-form and include `Band`, `Region`, and `Value`.
The raw connection `Region` value is relabeled through the shared directed
connection helper so the line label is `SNr→STN`.

## Plot Layout

The x-axis uses the canonical frequency-band order:

```text
Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
```

The displayed tick labels use the shared labels from
`plugin/default/viz_defaults.py`. Figure geometry is deterministic and
boxsize-driven. The x-axis tick labels are rotated 45 degrees. The default inner
plotting box is fixed at:

```text
boxsize = (30, 25)
```

## Outputs

Each feature writes one PDF to its own `summary` folder:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/burst/burst/<feature>/summary/Band-Region_raw.pdf
```

The PSI plot writes:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/psi/psi/psi-mean-scalar/summary/Band-Region_raw.pdf
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/rest/burst_summary_scalar.py
conda run -n LFP python recipes/stnsnr/viz/rest/psi_summary_scalar.py
```
