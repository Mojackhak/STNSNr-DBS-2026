# Rest Scalar Scale Correlation

## Purpose

This workflow computes Spearman correlations between resting-state scalar
features and the retained subject-level clinical scale variables.

The workflow creates three analysis groups:

- `STN local property`
- `SNr local property`
- `SNr-STN/SNr->STN connectivity`

## Inputs

Clinical scale input:

```text
${MYLFP_PROJECT_ROOT}/summary/cohort/subj/subj_scale.xlsx
```

Rest scalar inputs are read from:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest
```

Local property inputs:

- `tfr/raw/raw-mean-scalar/raw-mean-scalar_subject_raw.pkl`
- `tfr/period/period-mean-scalar/period-mean-scalar_subject_raw.pkl`
- `burst/burst/burst-occupation-scalar/burst-occupation-scalar_subject_raw.pkl`
- `burst/burst/burst-rate-scalar/burst-rate-scalar_subject_raw.pkl`
- `burst/burst/burst-duration-scalar/burst-duration-scalar_subject_raw.pkl`
- `burst/burst/burst-mean-scalar/burst-mean-scalar_subject_raw.pkl`
- `params/params/params-mean-scalar/params-mean-scalar_subject_raw.pkl`

Connectivity inputs:

- `conn/coh/coh-mean-scalar/coh-mean-scalar_subject_raw.pkl`
- `conn/wpli/wpli-mean-scalar/wpli-mean-scalar_subject_raw.pkl`
- `conn/ciplv/ciplv-mean-scalar/ciplv-mean-scalar_subject_raw.pkl`
- `gc/delta_net_gc/delta_net_gc-mean-scalar/delta_net_gc-mean-scalar_subject_raw.pkl`
- `psi/psi/psi-mean-scalar/psi-mean-scalar_subject_raw.pkl`

`tfr/aperiod` is intentionally excluded. The user-facing `wpi` name is treated
as the existing `wpli` input and is displayed as `wPLI`.

Each input pkl must contain `ID`, `Region`, `Band`, and `Value`, and the
resting-state context must be `Pose=Sit`, `Med=Off`, `Task=Single`, and
`Phase=All`.

## Feature Order

The frequency-band order is:

```text
Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
```

Local property x-axis order is band-first. Within each band the property order
is:

```text
Raw, Periodic, Burst occupation, Burst rate, Burst duration, Burst mean
```

`params` features do not expand by frequency band and are appended after the
banded features:

```text
Exponent, Offset
```

Connectivity x-axis order is also band-first. Within each band the metric order
is:

```text
COH, wPLI, ciPLV, GC, PSI
```

Connectivity features are filtered to the raw `Region` value `SNr-STN`. For
display and interpretation, `COH`, `wPLI`, and `ciPLV` are treated as
undirected `SNr-STN` metrics, while `GC` and `PSI` are treated as directed
`SNr->STN` metrics.

## Feature Display Labels

The correlation workbooks keep stable internal feature keys. Heatmap x-axis
labels are display-only and are generated from
`plugin/default/viz_defaults.py`:

- frequency-band names use `BAND_TPL`, such as `δ`, `θ`, `α`, `low-β`,
  `high-β`, and `γ`;
- `Beta_low` and `Beta_high` are displayed in modifier-first order as
  `low-β` and `high-β`;
- property names follow the full `RAW_YLABEL_TPL` style, with unit suffixes
  removed.

Local banded examples:

```text
Total δ power
Periodic δ power
δ burst occupation
arcsinh δ burst rate
ln δ burst duration
ln δ burst amplitude
```

Local params are displayed as:

```text
Exponent
Offset
```

Connectivity examples:

```text
Fisher z of sqrt(δ coherence)
logit(δ wPLI)
logit(δ ciPLV)
δ TRGC
δ PSI
```

## Clinical Scale Order

The y-axis uses the retained scale variables in the established order from the
MER spike correlation workflow:

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

Heatmaps use shorter display labels for selected rows:

```text
LCT improvement rate -> LCT Δ%
MDS-UPDRS III (Med-OFF) -> MDS-UPDRS III
MDS-UPDRS III axial (Med-OFF) -> Axial
MDS-UPDRS III rigidity (Med-OFF) -> Rigidity
MDS-UPDRS III bradykinesia (Med-OFF) -> Bradykinesia
MDS-UPDRS III rest tremor (Med-OFF) -> Rest tremor
MDS-UPDRS III action tremor (Med-OFF) -> Action tremor
```

The short heatmap labels do not change workbook row names. All displayed
MDS-UPDRS III labels refer to the Med-OFF state.

## Correlation and Visualization

The statistical script uses `compute_correlation_matrix` with:

```text
method="spearman"
correction_mode="All"
correction_method="bh"
```

The correlation workbook stores `r`, corrected `p`, raw `p_raw`, adjusted
`p_adj`, and pairwise `n`. Heatmap significance stars use raw `p_raw`.

Heatmaps use `plot_well_heatmap_df` with a symmetric color range around zero,
the default `cm.vik` colormap, black grid lines, and the colorbar label
`Spearman's ρ`.

Each heatmap uses a top strip label to identify the analysis group:

```text
STN local property -> STN
SNr local property -> SNr
Connectivity property -> SNr-STN / SNr→STN
```

## Outputs

All outputs are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/rest/corr
```

Statistical outputs:

- `rest_stn_local_scale_corr.xlsx`
- `rest_snr_local_scale_corr.xlsx`
- `rest_connectivity_scale_corr.xlsx`
- `rest_stn_local_subject_wide.xlsx`
- `rest_snr_local_subject_wide.xlsx`
- `rest_connectivity_subject_wide.xlsx`
- `subj_scale_subset_for_rest_corr.xlsx`

`subj_scale_subset_for_rest_corr.xlsx` contains one sheet per analysis group,
because the available paired IDs can differ across local and connectivity
features.

Visualization outputs:

- `rest_stn_local_scale_corr_heatmap.pdf`
- `rest_stn_local_scale_corr_heatmap.png`
- `rest_snr_local_scale_corr_heatmap.pdf`
- `rest_snr_local_scale_corr_heatmap.png`
- `rest_connectivity_scale_corr_heatmap.pdf`
- `rest_connectivity_scale_corr_heatmap.png`

## Run Commands

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/rest/rest_scale_correlation.py
conda run -n LFP python recipes/stnsnr/viz/rest/rest_scale_correlation.py
```
