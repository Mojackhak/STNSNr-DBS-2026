# gait_cycle tabular derivation

This directory contains preprocessing entry points for gait-cycle pointwise statistics.

## Purpose

`export_region_long.py` converts raw gait-cycle summary tables into Region-level, cycle-normalized model inputs. It reads the default paths and schema from `plugin/default/stats_gait_cycle_default.py`.

## Command

Run from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP python pipeline/tabular/derive/gait_cycle/export_region_long.py
```

The command writes nested Region-level trace tables and band-level long input tables under:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/gait_cycle
```

Use `--input-root`, `--output-root`, and `--task` to override the plugin defaults.
