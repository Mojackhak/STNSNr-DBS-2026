# Walk Task Dual vs Single Signed-LogP Statistics

This workflow builds the intermediate table for raw walk `Task` signed-logP
heatmaps split by `Pose`.

## Scope

- Use only `rlmer_model` outputs.
- Use only raw walk model outputs; normalized outputs are excluded.
- Use Task-containing Tukey outputs:
  - STN/SNr local metrics and parameters:
    `Task-Pose-Region_tukey.csv`
  - SNr-/→STN connectivity metrics:
    `Task-Pose_tukey.csv`
- Omnibus `Task` heatmaps are deferred.

## Task Contrast

The source Tukey files store `Single - Dual`. The heatmap displays
`Dual vs Single`, so the estimate is flipped before computing signed-logP.

For each selected row:

```text
estimate_dual_minus_single = -source_estimate_single_minus_dual
signed_logp = sign(estimate_dual_minus_single) * -log10(p_tukey)
signed_logp_display = clip(signed_logp, -3, 3)
```

Positive heatmap values therefore mean `Dual > Single`.

Rows are grouped into pose panels in this fixed order:

1. `Sit`
2. `Stand`
3. `Walk`
4. `Turn`

## Output

The script writes:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/walk_task_dual_vs_single_signed_logp_values.csv
```

Expected rows:

```text
STN local:      6 metrics x 6 bands x 4 poses = 144
SNr local:      6 metrics x 6 bands x 4 poses = 144
Connectivity:  5 metrics x 6 bands x 4 poses = 120
Parameters:    2 params x 2 regions x 4 poses = 16
Total: 424
```

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_task_dual_vs_single_signed_logp.py
```

