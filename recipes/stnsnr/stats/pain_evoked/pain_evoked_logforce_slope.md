# Pain Evoked LogForce Slope Table

## Purpose

This workflow prepares an audit table for pain-evoked `LogForce_c` slope
heatmaps. It reads existing `rlmer_model` slope CSV files only and does not
refit any models.

## Input Models

Region-specific metrics use:

```text
LogForce-Phase-Region_slope.csv
```

Connectivity metrics use:

```text
LogForce-Phase_slope.csv
```

The plotted value is the raw `LogForce_c` slope because the normalized source
tables currently contain only `OFS`, while slope outputs contain both `ONS` and
`OFS`.

## Feature Scope

The table includes these pain-evoked metrics:

```text
duration, onset_time, offset_time, peak_time, peak_value, auc_rel_left,
mean_slope_rise, mean_slope_fall
```

The LFP feature scope is:

- `tfr/period`: `Delta`, `Alpha`, `Beta_low`, `Gamma`;
- `tfr/raw`: `Delta`, `Theta`;
- `conn/coh`: `Delta`;
- `conn/wpli`: `Delta`;
- `params/params`: `Exponent`, `Offset`.

## Output

The output table is:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/pain_evoked/corr/logforce_slope/pain_evoked_logforce_slope_values.csv
```

Expected row count is 288:

- 192 region-specific frequency rows;
- 32 connectivity rows;
- 64 parameter rows.

The audit table keeps all requested rows, including `SNr` `Periodic power`
`Gamma` and `wPLI` `OFS`; visualization-only display filtering is applied only
by the heatmap renderer.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain_evoked/pain_evoked_logforce_slope.py
```
