# Pain LFP-KPPS 3-Month Correlation

## Purpose

This workflow computes subject-level Spearman correlations between pain LFP
changes and KPPS clinical variables.

The pain LFP side is defined as phase-related change relative to `BL`:

```text
ONS - BL
SUS - BL
OFS - BL
PST - BL
```

The KPPS side contains the preoperative KPPS score and the existing derived
3-month KPPS improvement variables used by the project.

## Inputs

Pain LFP feature inputs are normalized subject tables under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/pain
```

The workflow reads fixed scalar `*_subject_normalized.csv` files. The
normalized `Value` column is treated as the LFP phase effect:

```text
Phase - BL
```

Clinical inputs:

```text
${MYLFP_PROJECT_ROOT}/summary/cohort/subj/subj_scale.xlsx
${MYLFP_PROJECT_ROOT}/summary/stats/clinic/correlation/pain/scale_subject.xlsx
```

Preoperative KPPS is read from the `KPPS` column in `subj_scale.xlsx`.
The KPPS improvement variables are read from the existing derived pain
clinical correlation table. They are not recomputed in this workflow, so the
project's established zero-baseline handling and inclusion convention are
preserved.

Both sides are paired by `ID`. `ID` is retained in intermediate audit tables
but is not correlated.

## LFP Feature Order

Local STN and SNr features use frequency-band-first ordering:

```text
Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
```

Within each band, local features are ordered as:

```text
raw, period, burst_occupation, burst_rate, burst_duration, burst_mean
```

Band-level aperiodic power is excluded from this correlation workflow.
Aperiodic parameters are retained and appended after band features:

```text
Exponent, Offset
```

Connectivity features use the same band order and the following feature order
within each band:

```text
coh, wpli, ciplv, delta_net_gc, psi
```

## KPPS Variables

The y-axis KPPS variables are:

```text
Pre-op KPPS
KPPS Δ% (STN, 3m)
ΔKPPS Δ% (+SNr, 3m)
```

`Pre-op KPPS` is the raw preoperative KPPS score from `subj_scale.xlsx`.
`KPPS Δ% (STN, 3m)` and `ΔKPPS Δ% (+SNr, 3m)` are read directly from the
existing derived clinical table for pain correlations.

## Correlation

The statistical script computes one correlation matrix per analysis group and
phase. Each matrix uses `compute_correlation_matrix` with:

```text
method="spearman"
correction_mode="All"
correction_method="bh"
```

The exported workbooks store Spearman rho (`r`), corrected `p`, raw `p_raw`,
adjusted `p_adj`, and pairwise `n`.

## Outputs

All outputs are written to the dedicated subfolder:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/pain/corr/pain_lfp_kpps_3m
```

Statistical outputs:

- `pain_kpps_3m_subject_wide.xlsx`
- `pain_kpps_3m_subset_for_corr.xlsx`
- `pain_stn_local_subject_normalized_wide_by_phase.xlsx`
- `pain_snr_local_subject_normalized_wide_by_phase.xlsx`
- `pain_connectivity_subject_normalized_wide_by_phase.xlsx`
- `pain_stn_local_kpps_3m_corr_by_phase.xlsx`
- `pain_snr_local_kpps_3m_corr_by_phase.xlsx`
- `pain_connectivity_kpps_3m_corr_by_phase.xlsx`

Expected per-phase correlation matrix shapes:

```text
STN local:     3 x 38
SNr local:     3 x 38
Connectivity:  3 x 30
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain/pain_lfp_kpps_3m_correlation.py
```
