# Walk LFP-Gait PCA 3-Month Correlation

## Purpose

This workflow computes subject-level Spearman correlations between walking
LFP changes and 3-month gait PCA distance effects.

The walking LFP side is defined as pose-related change relative to `Sit`:

```text
Stand - Sit
Walk - Sit
Turn - Sit
```

The gait side is defined from 3-month gait PCA distance values:

```text
STN vs Pre-op
STN+SNr vs Pre-op
STN+SNr vs STN
```

## Inputs

Walking LFP feature inputs are normalized subject tables under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk
```

The workflow reads fixed scalar `*_subject_normalized.csv` files. Rows are
filtered to:

```text
Med == "Off"
Task == "Single"
Phase == "All"
Pose in {"Stand", "Walk", "Turn"}
```

The normalized `Value` column is treated as the LFP pose effect:

```text
Pose - Sit
```

Gait PCA distance input:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/clinic/pca/gait/dist.xlsx
```

The workflow uses only `Phase == "3 m"` rows. Both sides are paired by `ID`.
`ID` is retained in intermediate audit tables but is not correlated.

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

## Gait PCA Variables

The y-axis gait PCA variables are:

```text
ΔGait PCA STN vs Pre-op
ΔGait PCA STN+SNr vs Pre-op
ΔGait PCA STN+SNr vs STN
```

`ΔGait PCA STN vs Pre-op` and `ΔGait PCA STN+SNr vs Pre-op` are read directly
from the 3-month `dist.xlsx` rows for `Protocol == "STN"` and
`Protocol == "STN+SNr"`.

`ΔGait PCA STN+SNr vs STN` is calculated per subject as:

```text
STN+SNr - STN
```

## Correlation

The statistical script computes one correlation matrix per analysis group and
pose. Each matrix uses `compute_correlation_matrix` with:

```text
method="spearman"
correction_mode="All"
correction_method="bh"
```

The exported workbooks store `r`, corrected `p`, raw `p_raw`, adjusted
`p_adj`, and pairwise `n`.

## Outputs

All outputs are written to the dedicated subfolder:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/walk_lfp_gait_pca_3m
```

Statistical outputs:

- `walk_gait_pca_3m_distance_subject_wide.xlsx`
- `walk_gait_pca_3m_distance_subset_for_corr.xlsx`
- `walk_single_stn_local_subject_normalized_wide_by_pose.xlsx`
- `walk_single_snr_local_subject_normalized_wide_by_pose.xlsx`
- `walk_single_connectivity_subject_normalized_wide_by_pose.xlsx`
- `walk_single_stn_local_gait_pca_3m_corr_by_pose.xlsx`
- `walk_single_snr_local_gait_pca_3m_corr_by_pose.xlsx`
- `walk_single_connectivity_gait_pca_3m_corr_by_pose.xlsx`

Expected per-pose correlation matrix shapes:

```text
STN local:     3 x 38
SNr local:     3 x 38
Connectivity:  3 x 30
```

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_lfp_gait_pca_3m_correlation.py
```
