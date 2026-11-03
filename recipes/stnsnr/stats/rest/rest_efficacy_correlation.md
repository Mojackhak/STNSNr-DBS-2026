# Rest LFP Efficacy Correlation

## Purpose

This workflow computes Spearman correlations between resting-state LFP scalar
features and subject-level clinical efficacy variables.

It creates three analysis groups:

- `STN local property` versus STN clinical improvement.
- `SNr local property` versus the additional improvement from adding SNr.
- `SNr-STN/SNr->STN connectivity` versus the additional improvement from
  adding SNr.

## Inputs

Resting-state LFP feature inputs and x-axis ordering are reused from
`rest_scale_correlation.py`.

Clinical efficacy input:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/clinic/coords/scale/scale_subject.xlsx
```

The efficacy workbook is a long table and must contain:

```text
ID, Region, Scale, Value
```

Only 3-month efficacy variables are used. Immediate efficacy variables are
excluded.

## Efficacy Variable Order

`STN local property` uses non-`Δ` STN improvement variables:

```text
UPDRS-III Δ% (STN, 3m)
UPDRS-III axial Δ% (STN, 3m)
FOG-Q Δ% (STN, 3m)
MADRS Δ% (STN, 3m)
KPPS Δ% (STN, 3m)
ADL Δ% (STN, 3m)
SE-ADL Δ% (STN, 3m)
PDQ-39 Δ% (STN, 3m)
```

`SNr local property` and `SNr-STN/SNr->STN connectivity` use `Δ` variables
that represent the additional improvement from adding SNr relative to STN:

```text
ΔUPDRS-III Δ% (+SNr, 3m)
ΔUPDRS-III axial Δ% (+SNr, 3m)
ΔFOG-Q Δ% (+SNr, 3m)
ΔMADRS Δ% (+SNr, 3m)
ΔKPPS Δ% (+SNr, 3m)
ΔADL Δ% (+SNr, 3m)
ΔSE-ADL Δ% (+SNr, 3m)
ΔPDQ-39 Δ% (+SNr, 3m)
```

Heatmaps use shorter row labels:

```text
UPDRS-III Δ%
UPDRS-III axial Δ%
FOG-Q Δ%
MADRS Δ%
KPPS Δ%
ADL Δ%
SE-ADL Δ%
PDQ-39 Δ%
```

The `Δ` efficacy heatmaps keep the leading `Δ` in the row label.

## Correlation and Visualization

The statistical script uses `compute_correlation_matrix` with:

```text
method="spearman"
correction_mode="All"
correction_method="bh"
```

The correlation workbook stores `r`, corrected `p`, raw `p_raw`, adjusted
`p_adj`, and pairwise `n`. Heatmap significance stars use raw `p_raw`.

Heatmaps reuse the rest scale-correlation visual style:

- x-axis order and labels from `rest_scale_correlation.py`;
- `low-β` and `high-β` band labels;
- top strip labels `STN`, `SNr`, and `SNr-STN / SNr→STN`;
- `cm.vik` colormap, black grid lines, and `Spearman's ρ` colorbar label.

## Outputs

All outputs are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/corr
```

Statistical outputs:

- `rest_stn_local_efficacy_corr.xlsx`
- `rest_snr_local_delta_efficacy_corr.xlsx`
- `rest_connectivity_delta_efficacy_corr.xlsx`
- `rest_stn_local_efficacy_subject_wide.xlsx`
- `rest_snr_local_delta_efficacy_subject_wide.xlsx`
- `rest_connectivity_delta_efficacy_subject_wide.xlsx`
- `scale_efficacy_subset_for_rest_corr.xlsx`

Visualization outputs:

- `rest_stn_local_efficacy_corr_heatmap.pdf`
- `rest_stn_local_efficacy_corr_heatmap.png`
- `rest_snr_local_delta_efficacy_corr_heatmap.pdf`
- `rest_snr_local_delta_efficacy_corr_heatmap.png`
- `rest_connectivity_delta_efficacy_corr_heatmap.pdf`
- `rest_connectivity_delta_efficacy_corr_heatmap.png`

## Run Commands

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/rest/rest_efficacy_correlation.py
conda run -n LFP python recipes/stnsnr/viz/rest/rest_efficacy_correlation.py
```
