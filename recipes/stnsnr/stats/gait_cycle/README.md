# gait_cycle statistics

This directory runs pointwise mixed-model statistics on gait-cycle aligned STN/SNr traces.

Preprocessing is run from `pipeline/tabular/derive/gait_cycle`, while this directory keeps the R pointwise model and interval summary scripts.

## Inputs

Raw gait-cycle tables are read from:

```bash
${MYLFP_PROJECT_ROOT}/summary/table/raw/gait
```

The preprocessing step includes all available gait metric families:

- local metrics: `raw`, `period`, `aperiod`, `params`, `burst`
- connection metrics: `coh`, `ciplv`, `wpli`, `delta_net_gc`, `psi`

Only `Task == "Single"` rows are analyzed.

## Output layout

Outputs mirror the existing `gait_event` parameter hierarchy:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/gait_cycle/
  <param>/
    <subparam>/
      <subparam>-mean-trace/
        <subparam>-mean-trace_region_normalized.pkl
        <Band>/
          timepoint_input.csv
          timepoint_deviation.csv
          lmer_model/
            timepoint_deviation.csv
            timepoint_intervals.csv
            timepoint_interval_candidates.csv
            Region_norm.pdf
          rlmer_model/
            timepoint_deviation.csv
            timepoint_intervals.csv
            timepoint_interval_candidates.csv
            Region_norm.pdf
          joint_model/
            timepoint_intervals.csv
            timepoint_interval_candidates.csv
```

## Preprocessing policy

- Existing derive-style value transforms are applied before modeling.
- Each trace is centered by its own full-cycle mean.
- Original `0..100%` traces are resampled to 100 bin centers, `0.5..99.5%`, by adjacent-point averaging.
- Data are aggregated to Region level by averaging over `Contact` while retaining `ID`, `Subject`, `Region`, side-related columns, `Task`, `Epoch`, `Metric`, `Band`, `t_pct`, and `Value`.
- Retained side-related columns and `Epoch` are kept as within-subject repeated observations and are not default fixed-effect predictors.

## Models

For every gait-cycle time point:

- local metrics use `Value ~ Region + (1 | ID)`
- connection metrics use `Value ~ 1 + (1 | ID)`

Both `lmer` and `rlmer` are fit. Significant intervals are summarized separately for `lmer`, `rlmer`, and joint same-direction significance with `p < 0.05` and a minimum continuous span of `10%`.

## Regeneration

Run from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP python pipeline/tabular/derive/gait_cycle/export_region_long.py
conda run -n LFP Rscript recipes/stnsnr/stats/gait_cycle/run_pointwise_lmm.R
conda run -n LFP Rscript recipes/stnsnr/stats/gait_cycle/summarize_intervals.R
```

Generated outputs are written under the parameter and band hierarchy shown above.

For smoke testing one band, run:

```bash
conda run -n LFP Rscript recipes/stnsnr/stats/gait_cycle/run_pointwise_lmm.R --input-csv <Band>/timepoint_input.csv --jobs 1
conda run -n LFP Rscript recipes/stnsnr/stats/gait_cycle/summarize_intervals.R --deviation-csv <Band>/timepoint_deviation.csv
```

Default paths, trace schema, cycle-bin settings, model directory names, and figure filenames are configured in `plugin/default/stats_gait_cycle_default.py`.
