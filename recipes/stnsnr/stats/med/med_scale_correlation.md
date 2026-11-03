# Med LFP-Scale Delta Percent Correlation

## Purpose

This workflow computes subject-level Spearman correlations between medication
effects on LFP scalar features and medication-related clinical improvement
rates.

It creates three analysis groups:

- `STN local property` versus clinical improvement rates.
- `SNr local property` versus clinical improvement rates.
- `SNr-STN/SNr->STN connectivity` versus clinical improvement rates.

## Inputs

Medication LFP feature inputs are normalized subject tables under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/med
```

The workflow reads `*_subject_normalized.csv` scalar files. These normalized
tables contain one `Med == "On"` row for each available subject-feature
combination and encode the LFP medication effect as:

```text
Med-ON - Med-OFF
```

Clinical scale input:

```text
${MYLFP_PROJECT_ROOT}/summary/cohort/subj/subj_scale.xlsx
```

Both inputs are paired by `ID`. `ID` is used only for pairing and audit tables;
it is not included as a correlated variable.

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

## Clinical Delta Percent Variables

The y-axis clinical variables are:

```text
LCT improvement rate
MDS-UPDRS III Δ%
MDS-UPDRS III axial Δ%
MDS-UPDRS III rigidity Δ%
MDS-UPDRS III bradykinesia Δ%
MDS-UPDRS III rest tremor Δ%
MDS-UPDRS III action tremor Δ%
```

The workflow also creates a separate combined-tremor analysis with these
clinical variables:

```text
LCT improvement rate
MDS-UPDRS III Δ%
MDS-UPDRS III axial Δ%
MDS-UPDRS III rigidity Δ%
MDS-UPDRS III bradykinesia Δ%
MDS-UPDRS III tremor Δ%
```

The combined tremor score is calculated at the subject level before the
correlation analysis:

```text
Tremor Med-OFF = Rest tremor Med-OFF + Action tremor Med-OFF
Tremor Med-ON  = Rest tremor Med-ON  + Action tremor Med-ON
Tremor Δ%      = (Tremor Med-OFF - Tremor Med-ON) / Tremor Med-OFF * 100
```

The combined value is `NaN` when any source score is missing or when the
combined Med-OFF score is zero. The existing separate rest- and action-tremor
analysis remains unchanged.

`LCT improvement rate` is retained as stored in `subj_scale.xlsx`. The six
MDS-UPDRS III improvement rates are calculated as:

```text
(Med-OFF - Med-ON) / Med-OFF * 100
```

If the `Med-OFF` denominator is missing or zero, the calculated improvement
rate is set to `NaN`.

## Correlation

The statistical script uses `compute_correlation_matrix` with:

```text
method="spearman"
correction_mode="All"
correction_method="bh"
```

The correlation workbook stores `r`, corrected `p`, raw `p_raw`, adjusted
`p_adj`, and pairwise `n`.

## Outputs

All outputs are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/med/corr
```

Statistical outputs:

- `subj_scale_med_delta_percent_subject_wide.xlsx`
- `subj_scale_med_delta_percent_subset_for_corr.xlsx`
- `med_stn_local_subject_normalized_wide.xlsx`
- `med_snr_local_subject_normalized_wide.xlsx`
- `med_connectivity_subject_normalized_wide.xlsx`
- `med_stn_local_scale_delta_percent_corr.xlsx`
- `med_snr_local_scale_delta_percent_corr.xlsx`
- `med_connectivity_scale_delta_percent_corr.xlsx`
- `subj_scale_med_combined_tremor_delta_percent_subject_wide.xlsx`
- `subj_scale_med_combined_tremor_delta_percent_subset_for_corr.xlsx`
- `med_stn_local_scale_combined_tremor_delta_percent_corr.xlsx`
- `med_snr_local_scale_combined_tremor_delta_percent_corr.xlsx`
- `med_connectivity_scale_combined_tremor_delta_percent_corr.xlsx`

Expected correlation matrix shapes:

```text
STN local:     7 x 38
SNr local:     7 x 38
Connectivity:  7 x 30
```

Expected combined-tremor correlation matrix shapes:

```text
STN local:     6 x 38
SNr local:     6 x 38
Connectivity:  6 x 30
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/med/med_scale_correlation.py
```

To generate only the additional combined-tremor tables and correlation
workbooks without rewriting the existing separate-tremor outputs, use:

```bash
conda run -n LFP python recipes/stnsnr/stats/med/med_scale_correlation.py --combined-tremor-only
```
