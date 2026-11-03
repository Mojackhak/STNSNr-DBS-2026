# Walk Task Dual vs Single Signed-LogP Layered Heatmaps

This workflow renders raw walk `Task` signed-logP heatmaps from the
Pose-specific intermediate table.

## Input

Run the statistics workflow first:

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_task_dual_vs_single_signed_logp.py
```

The visualization script reads:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/walk_task_dual_vs_single_signed_logp_values.csv
```

## Figure Layout

The script writes one layered heatmap per data group:

- STN local
- SNr local
- SNr-/→STN connectivity
- Parameters

Each figure has four vertically stacked pose panels in this fixed order:

1. `Sit`
2. `Stand`
3. `Walk`
4. `Turn`

Right-side strips label the pose. The shared colorbar is labeled signed
`-log10(p_Tukey)`, with endpoint labels `Dual > Single` and `Dual < Single`.
Display values are clipped to `[-3, 3]`.

Layered panels are rendered through the shared
`visualdf.plot_well_heatmap_grid_df` backend, matching the MNI standardized
slope heatmap workflow.

All heatmaps, including the Params heatmap, use `3.0 x 3.0 mm` cells.
Each cell is outlined with black solid grid lines matching the med
signed-logP heatmap style (`linewidth=0.6`, `alpha=0.9`).

In addition to the original orientation, the workflow writes a second
transposed output set. For STN local, SNr local, and SNr-/→STN connectivity,
the transposed heatmaps use frequency bands on the y-axis and
parameters/metrics on the x-axis. For the Params heatmap, the transposed
layout uses `Exponent` and `Offset` on the y-axis and regions on the x-axis.

## Outputs

Heatmap files are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/task_dual_vs_single
```

Output basenames:

```text
walk_task_stn_local_dual_vs_single_signed_logp_layered_heatmap
walk_task_snr_local_dual_vs_single_signed_logp_layered_heatmap
walk_task_connectivity_dual_vs_single_signed_logp_layered_heatmap
walk_task_params_dual_vs_single_signed_logp_layered_heatmap
```

Transposed output basenames:

```text
walk_task_stn_local_dual_vs_single_signed_logp_freq_by_param_layered_heatmap
walk_task_snr_local_dual_vs_single_signed_logp_freq_by_param_layered_heatmap
walk_task_connectivity_dual_vs_single_signed_logp_freq_by_param_layered_heatmap
walk_task_params_dual_vs_single_signed_logp_transposed_layered_heatmap
```

Each basename is saved as both `.pdf` and `.png`.
Rerunning the script rewrites these heatmap files in place and does not delete
other files from the output directory.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/walk/walk_task_dual_vs_single_signed_logp_layered_heatmap.py
```
