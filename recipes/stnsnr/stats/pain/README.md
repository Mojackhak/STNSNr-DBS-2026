# Pain Statistics

This directory contains statistical workflows for pain-period STN/SNr scalar
and MNI-coordinate summaries.

## Current model policy

Pain scalar models do not include `Hemi` as a fixed-effect predictor. The
scalar scripts fit pain effects with phase, laterality, force, and region terms
only:

- `single_scalar.R` fits raw and normalized scalar values with
  `Value ~ Region*Phase*(Lat + LogForce_c) + (1|ID)`.
- `double_scalar.R` fits raw and normalized connection scalar values with
  `Value ~ Phase*(Lat + LogForce_c) + (1|ID)`.
- `single_xyz.R` and `double_xyz.r` fit coordinate effects with region, phase,
  and standardized MNI coordinates.

All scripts source their shared EMM and schema helpers from the local MyLFP
repository.

The scalar scripts no longer generate `Hemi-Phase` or
`Hemi-Phase-Region` EMM/Tukey outputs. Existing legacy files with those names
may remain in the output tree until manually removed; rerunning the current
scripts only overwrites outputs that are still generated.

## Signed-LogP Heatmap Inputs

`pain_phase_pairwise_signed_logp.py` builds the raw Phase pairwise signed-logP
intermediate table for layered heatmaps. It uses only `rlmer_model`
`Phase-Region_tukey.csv` and `Phase_tukey.csv` outputs, keeps all six bands,
and writes to the `corr/phase_pairwise` subfolder.

`pain_lat_ipsi_vs_contra_signed_logp.py` builds the BL-normalized Lat
signed-logP intermediate table for phase-layered heatmaps. It uses only
`rlmer_model` `Lat-Phase-Region_tukey.csv` and `Lat-Phase_tukey.csv` outputs,
keeps all six bands, writes `Δ`-prefixed metric and parameter labels, and
writes to the `corr/lat_ipsi_vs_contra` subfolder.

See `pain_phase_pairwise_signed_logp.md` for the contrast direction, row count,
and output schema. See `pain_lat_ipsi_vs_contra_signed_logp.md` for the Lat
direction, row count, and output schema.

## MNI Standardized-Slope Heatmap Inputs

`pain_mni_standardized_slope.py` builds the standardized MNI-coordinate slope
intermediate table for phase-layered spatial-gradient heatmaps. It uses only
existing `rlmer_model` `zMX/zMY/zMZ-Phase-Region_slope.csv` files, standardizes
each slope by the matching normalized `Value` SD within Band or parameter,
Region, and Phase, and writes to the `corr/mni_standardized_slope` subfolder.

See `pain_mni_standardized_slope.md` for the feature scope, row count, and
output schema.

## LogForce Standardized-Slope Heatmap Inputs

`pain_logforce_standardized_slope.py` builds the standardized LogForce slope
intermediate table for phase-layered heatmaps. It uses only existing
BL-normalized `rlmer_model` `LogForce-Phase-Region_slope.csv` and
`LogForce-Phase_slope.csv` files, standardizes each slope by the matching
BL-normalized `Value` SD, and writes to the `corr/logforce_standardized_slope`
subfolder.

See `pain_logforce_standardized_slope.md` for the feature scope, row count, and
output schema.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/pain/single_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/pain/double_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/pain/single_xyz.R
conda run -n LFP Rscript recipes/stnsnr/stats/pain/double_xyz.r
conda run -n LFP python recipes/stnsnr/stats/pain/pain_phase_pairwise_signed_logp.py
conda run -n LFP python recipes/stnsnr/stats/pain/pain_lat_ipsi_vs_contra_signed_logp.py
conda run -n LFP python recipes/stnsnr/stats/pain/pain_mni_standardized_slope.py
conda run -n LFP python recipes/stnsnr/stats/pain/pain_logforce_standardized_slope.py
```
