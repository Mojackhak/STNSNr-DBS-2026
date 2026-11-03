# Pain Evoked Statistics

This directory contains statistical workflows for pain-evoked scalar summaries.

## Current model policy

Pain-evoked models do not include `Hemi` as a fixed-effect predictor.

- `single_scalar.R` fits evoked scalar values with
  `Value ~ Region*Phase*LogForce_c + (1|ID)`.
- `double_scalar.R` fits evoked connection scalar values with
  `Value ~ Phase*LogForce_c + (1|ID)`.

The scripts generate phase and force-by-phase outputs only. Existing legacy
files outside those output names are not deleted by reruns.

## LogForce Slope Heatmap Inputs

`pain_evoked_logforce_slope.py` builds an audit table for pain-evoked
`LogForce_c` slope heatmaps from existing `rlmer_model` outputs. It uses raw
model slopes, not SD-standardized slopes, and writes to the
`corr/logforce_slope` subfolder.

See `pain_evoked_logforce_slope.md` for the feature scope, row count, and run
command.

## Phase Signed-LogP Heatmap Inputs

`pain_evoked_phase_ons_vs_ofs_signed_logp.py` builds the `ONS - OFS`
signed-logP intermediate table for pain-evoked Phase heatmaps from existing
`rlmer_model` Tukey outputs. It writes to the `corr/phase_ons_vs_ofs`
subfolder.

See `pain_evoked_phase_ons_vs_ofs_signed_logp.md` for the feature scope,
contrast direction, row count, and run command.
