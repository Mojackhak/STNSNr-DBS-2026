# Walk LFP-FOGQ 3-Month Correlation

## Purpose

This workflow computes subject-level Spearman correlations between walking
LFP pose effects and two established 3-month FOGQ improvement variables.

The walking LFP side is defined by the existing single-task, medication-off
pose-effect workflow:

```text
Stand - Sit
Walk - Sit
Turn - Sit
```

The clinical side separates the STN improvement from the incremental
improvement produced by adding SNr stimulation.

## Inputs

All live inputs are read from the mounted project root:

```text
${MYLFP_PROJECT_ROOT}
```

Clinical input:

```text
${MYLFP_PROJECT_ROOT}/summary/cohort/subj/subj_delta_effect.xlsx
```

The workflow reads these existing columns without recomputing them:

```text
FOG-Q Δ% (STN, 3m)
ΔFOG-Q Δ% (+SNr, 3m)
```

For FOGQ, lower scores are better. The upstream subject-effect workflow
defines the STN improvement as `(Baseline - STN at 3m) / Baseline * 100`.
The incremental SNr variable is the `STN+SNr` improvement minus the `STN`
improvement. Positive values therefore indicate clinical improvement.

Walking LFP inputs are the normalized scalar tables under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk
```

The workflow reuses the source definitions and feature order from the existing
walk LFP-gait PCA implementation and reads these scalar families directly:

- local `raw`, `period`, burst, `Exponent`, and `Offset` features;
- connectivity `coh`, `wpli`, `ciplv`, `delta_net_gc`, and `psi` features.

Rows are filtered directly to `Med == Off`, `Task == Single`, `Phase == All`,
and `Pose in {Stand, Walk, Turn}`. The normalized scalar `Value` is treated as
the pose effect relative to `Sit`. Reading the scalar sources directly keeps
the FOGQ cohort independent of any gait-PCA availability filter.

## Feature Order

Local STN and SNr matrices retain the existing 38-feature order: six bands
(`Delta`, `Theta`, `Alpha`, `Beta_low`, `Beta_high`, and `Gamma`) crossed with
`raw`, `period`, `burst_occupation`, `burst_rate`, `burst_duration`, and
`burst_mean`, followed by `Exponent` and `Offset`.

Connectivity matrices retain the existing 30-feature order: the same six
bands crossed with `coh`, `wpli`, `ciplv`, `delta_net_gc`, and `psi`.

## Correlation Contract

The statistical workflow computes one matrix for each analysis group and
pose using:

```text
method="spearman"
correction_mode="All"
correction_method="bh"
min_n=3
```

Benjamini-Hochberg correction is applied within each analysis-group-by-pose
matrix. Pairwise missingness is retained and the effective sample size for
every cell is exported in the `*_n` sheets.

Expected per-pose matrix shapes:

```text
STN local:     2 x 38
SNr local:     2 x 38
Connectivity:  2 x 30
```

## Outputs

All outputs are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/walk_lfp_fogq_3m
```

Audit tables:

- `walk_fogq_3m_subject_wide.xlsx`
- `walk_fogq_3m_subset_for_corr.xlsx`
- `walk_single_stn_local_subject_normalized_wide_by_pose.xlsx`
- `walk_single_snr_local_subject_normalized_wide_by_pose.xlsx`
- `walk_single_connectivity_subject_normalized_wide_by_pose.xlsx`

Correlation workbooks:

- `walk_single_stn_local_fogq_3m_corr_by_pose.xlsx`
- `walk_single_snr_local_fogq_3m_corr_by_pose.xlsx`
- `walk_single_connectivity_fogq_3m_corr_by_pose.xlsx`

Each correlation workbook contains pose-specific `r`, `p`, `p_raw`, `p_adj`,
and `n` sheets.

Rerunning the workflow rewrites only these dedicated FOGQ outputs. A change
to the clinical workbook invalidates the correlation outputs. A change to a
walking LFP scalar source invalidates its corresponding analysis group and
downstream heatmap.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_lfp_fogq_3m_correlation.py
```
