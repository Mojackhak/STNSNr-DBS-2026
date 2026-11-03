# MER Spike-Scale Correlation Workflow

## Purpose

This workflow computes subject-level Spearman correlations between MER spike
features and clinical scale variables for the STNSNr cohort.

The statistics script lives under:

```text
recipes/stnsnr/stats/mer
```

The visualization script lives under:

```text
recipes/stnsnr/viz/mer
```

## Inputs

MER spike input:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/mer/spike/spike_subject_raw.pkl
```

Clinical scale input:

```text
${MYLFP_PROJECT_ROOT}/summary/cohort/subj/subj_scale.xlsx
```

Both inputs are paired by `ID`. `ID` is used only for pairing and audit tables;
it is not included as a correlated y-axis variable.

## Spike Feature Order

The MER spike table is widened from long format using:

```text
ID + Region + Label + Param
```

The x-axis/internal parameter order is:

```text
Firing_rate
Burst_index
LvR
Peak_trough_ratio
Peak_to_valley
Half_width
Repolarization_slope
Recovery_slope
```

The heatmap x-axis display labels are:

```text
Firing rate
Burst index
LvR
Peak trough ratio
Peak to valley
Half width
Repolarization slope
Recovery slope
```

## Clinical Variable Order

The y-axis clinical variables are:

```text
Age
BMI
Education years
Disease duration
LCT improvement rate
MDS-UPDRS I
MDS-UPDRS II
MDS-UPDRS III (Med-OFF)
MDS-UPDRS III axial (Med-OFF)
MDS-UPDRS III rigidity (Med-OFF)
MDS-UPDRS III bradykinesia (Med-OFF)
MDS-UPDRS III rest tremor (Med-OFF)
MDS-UPDRS III action tremor (Med-OFF)
MDS-UPDRS IV
FOG-Q
MoCA
MMSE
BDI
MADRS
HAMD
HAMA
MAES
ESS
RBDSQ
RBDSQ-HK
SCOPA-AUT
SDQ
EAT-10
DSFS
CCCS
FSS
KPPS
ODQ
ADL
SE-ADL
PDQ-39
```

All `Med-ON` columns and `QUIP-RS` are excluded from the correlation input.

The heatmap shortens selected y-axis labels for readability:

```text
LCT improvement rate -> LCT Δ%
MDS-UPDRS III (Med-OFF) -> MDS-UPDRS III
MDS-UPDRS III axial (Med-OFF) -> Axial
MDS-UPDRS III rigidity (Med-OFF) -> Rigidity
MDS-UPDRS III bradykinesia (Med-OFF) -> Bradykinesia
MDS-UPDRS III rest tremor (Med-OFF) -> Rest tremor
MDS-UPDRS III action tremor (Med-OFF) -> Action tremor
```

These display labels do not change the workbook row names. All displayed
MDS-UPDRS III total and subdomain labels refer to the medication-off state.

## Correlation and Visualization

The statistics script calls:

```python
compute_correlation_matrix(
    df_x=spike_wide_common,
    df_y=scale_subset_common,
    pair_col="ID",
    method="spearman",
    correction_mode="All",
    correction_method="bh",
)
```

The workbook stores raw and globally BH-FDR-adjusted p-values. The heatmap uses
Spearman rho as the color value and raw p-values for significance stars. The
colorbar label is `Spearman's ρ`.

The heatmap uses `plot_well_heatmap_df` with `cellsize` in millimeters so the
panel size is derived from the number of matrix rows and columns. The default
well heatmap colormap is `cm.vik`, and the default grid line color is black.
The well heatmap API supports `colorbar_length_mm`; the current MER corr
heatmap leaves it as `None`, so the colorbar spans the full panel height.

## Outputs

All workflow outputs are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/mer/spike/corr
```

Expected files:

```text
mer_spike_scale_corr.xlsx
mer_spike_subject_wide.xlsx
subj_scale_subset_for_corr.xlsx
mer_spike_scale_corr_heatmap.pdf
mer_spike_scale_corr_heatmap.png
```
