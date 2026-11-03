# Pain Evoked Phase ONS-vs-OFS Signed-LogP Table

## Purpose

This workflow prepares the signed-logP intermediate table for pain-evoked
`Phase` heatmaps. It reads existing `rlmer_model` Tukey CSV files only and does
not refit any models.

## Input Models

Region-specific metrics and aperiodic parameters use:

```text
Phase-Region_tukey.csv
```

Connectivity metrics use:

```text
Phase_tukey.csv
```

The selected source contrast is `ONS - OFS`. The source estimate already has
the intended direction and is not flipped.

## Feature Scope

The table includes these pain-evoked metrics:

```text
duration, onset_time, offset_time, peak_time, peak_value, auc_rel_left,
mean_slope_rise, mean_slope_fall
```

The LFP feature scope is:

- `SNr`: `tfr/raw` `Delta` and `Theta`;
- `STN`: `tfr/raw` `Delta` and `Theta`, plus `tfr/period` `Delta`, `Alpha`,
  `Beta_low`, and `Gamma`;
- `SNr-/→STN`: `conn/coh` `Delta` and `conn/wpli` `Delta`;
- `Params`: `Exponent` and `Offset` for `SNr` and `STN`.

## Encoding

The signed value is:

```text
signed_logp = sign(source_estimate_ons_minus_ofs) * -log10(p_tukey)
```

Positive values mean `ONS > OFS`; negative values mean `ONS < OFS`.
`signed_logp_display` is clipped to `[-3, 3]`. Stars are derived from
`p_tukey`.

## Output

The output table is:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/pain_evoked/corr/phase_ons_vs_ofs/pain_evoked_phase_ons_vs_ofs_signed_logp_values.csv
```

Expected row count is 112:

- 16 SNr local rows;
- 48 STN local rows;
- 16 connectivity rows;
- 32 parameter rows.

## Run Command

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/pain_evoked/pain_evoked_phase_ons_vs_ofs_signed_logp.py
```
