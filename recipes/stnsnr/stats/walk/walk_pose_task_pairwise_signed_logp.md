# Walk Pose Task Pairwise Signed-LogP Statistics

This workflow builds the intermediate table for raw walk `Pose` pairwise
signed-logP heatmaps split by `Task`.

## Scope

- Use only `rlmer_model` outputs.
- Use only raw walk model outputs; normalized outputs are excluded.
- Use Task-containing Tukey outputs:
  - STN/SNr local metrics and parameters:
    `Pose-Region-Task_tukey.csv`
  - STN-SNr connectivity metrics:
    `Pose-Task_tukey.csv`
- Omnibus `Pose` heatmaps are deferred.

## Pose Contrasts

The source Tukey files store each contrast as `a - b`. The heatmap strip keeps
that source-order label as `a vs b`, but the signed value uses the flipped
estimate `b - a`.

| Source contrast `a - b` | Display panel `a vs b` | Signed estimate `b - a` |
| --- | --- | --- |
| `Sit - Stand` | `Sit vs Stand` | `Stand - Sit` |
| `Stand - Walk` | `Stand vs Walk` | `Walk - Stand` |
| `Walk - Turn` | `Walk vs Turn` | `Turn - Walk` |
| `Sit - Walk` | `Sit vs Walk` | `Walk - Sit` |
| `Stand - Turn` | `Stand vs Turn` | `Turn - Stand` |
| `Sit - Turn` | `Sit vs Turn` | `Turn - Sit` |

For each selected row:

```text
estimate_b_minus_a = -source_estimate
signed_logp = sign(estimate_b_minus_a) * -log10(p_tukey)
signed_logp_display = clip(signed_logp, -3, 3)
```

Positive heatmap values therefore mean that `b > a` for a display panel labeled
`a vs b`.

## Output

The script writes:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/walk_pose_task_pairwise_signed_logp_values.csv
```

The table includes source path, group, region, metric, band or parameter,
task, source contrast, source estimate, flipped estimate, `p_tukey`,
signed-logP, clipped display value, and significance star.

Expected rows:

```text
STN local:      6 metrics x 6 bands x 6 comparisons x 2 tasks = 432
SNr local:      6 metrics x 6 bands x 6 comparisons x 2 tasks = 432
Connectivity:  5 metrics x 6 bands x 6 comparisons x 2 tasks = 360
Parameters:    2 params x 2 regions x 6 comparisons x 2 tasks = 48
Total: 1272
```

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_pose_task_pairwise_signed_logp.py
```
