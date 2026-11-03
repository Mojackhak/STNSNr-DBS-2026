# STN/SNr Medication Statistics

These scripts fit medication-state mixed models for scalar and MNI-coordinate
features under `${MYLFP_PROJECT_ROOT}/summary/stats/med`.

## Current Model Factors

`Hemi` is not used as a model predictor or post-hoc grouping variable in the
medication statistics recipes.

- `single_scalar.R` fits raw scalar values with `Value ~ Med*Region + (1|ID)`.
- `single_scalar.R` fits normalized scalar values with `Value ~ Region + (1|ID)`.
- `double_scalar.R` fits raw scalar values with `Value ~ Med + (1|ID)`.
- `double_scalar.R` does not run normalized scalar post-hoc tests, because no
  non-`Hemi` predictor remains for those outputs.
- `single_xyz.R` and `double_xyz.R` keep the existing coordinate models:
  `Value ~ Region*(zMX + zMY + zMZ) + (1|ID)`.

The scalar scripts no longer write `Hemi`, `Med-Hemi`, `Hemi-Region`, or
`Med-Region-Hemi` EMM/Tukey outputs. Remove any legacy result files with those
names from the output tree before rerunning if the downstream analysis should
only see the current no-`Hemi` outputs.

## Runtime Helpers

The scalar scripts source EMM, tabular default, and data-frame schema helpers
from this `MyLFP` checkout. The R helper paths resolve through
`${MYLFP_REPO_ROOT}`, including `plugin/default/tabular_default.R`.

## LFP-Scale Delta Percent Correlation

`med_scale_correlation.py` correlates normalized medication effects on LFP
scalar features with clinical medication improvement rates. It writes
subject-wide audit tables, scale delta-percent intermediates, and correlation
workbooks to `${MYLFP_PROJECT_ROOT}/summary/stats/med/corr`. The
local STN/SNr feature set excludes band-level aperiodic power while retaining
`Exponent` and `Offset`.

## MNI Standardized Slope

`med_mni_standardized_slope.py` converts existing `rlmer_model`
MNI-coordinate slope outputs for normalized medication effects into
standardized slopes. It writes the long-form audit table used by the med MNI
standardized-slope heatmaps to
`${MYLFP_PROJECT_ROOT}/summary/stats/med/corr`.

## Signed-LogP Summary

`medon_vs_medoff_signed_logp.py` reads existing `rlmer_model` medication Tukey
outputs and writes the long-form signed-logP intermediate table for the
MedON-vs-MedOFF heatmaps. Positive signed-logP values mean `MedON > MedOFF`;
the script flips source `Off - On` estimates before computing the signed value.
