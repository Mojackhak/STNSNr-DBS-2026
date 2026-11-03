# turn statistics

This directory runs mixed-model statistics for turn-derived STN/SNr outputs.

## Model policy

- `Hemi` and `Task` are not included as model predictors in these scripts.
- Single-region scalar outputs fit `Value ~ Region*Lat + (1|ID)`.
- Double-region connection scalar outputs fit `Value ~ Lat + (1|ID)`.
- Single-region outputs include `Region` EMM/Tukey tables for Region-only display.

## Regeneration

Run the scripts from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/turn/single_scalar.R
conda run -n LFP Rscript recipes/stnsnr/stats/turn/double_scalar.R
conda run -n LFP python recipes/stnsnr/stats/turn/turn_lat_contra_vs_ipsi_signed_logp.py
```

Generated files overwrite existing files with the same path. Legacy output
files, including older files whose names contain `Hemi`, are not removed by
these scripts.

## Signed-LogP Heatmap Tables

`turn_lat_contra_vs_ipsi_signed_logp.py` builds the intermediate table for the
turn `Lat` main-variable heatmaps. It uses only `rlmer_model` Tukey rows with
source contrast `Ipsi - Contra`; positive signed-logP values mean
`Contra < Ipsi`. Outputs are written under
`${MYLFP_PROJECT_ROOT}/summary/stats/turn/corr/lat_contra_vs_ipsi`.
