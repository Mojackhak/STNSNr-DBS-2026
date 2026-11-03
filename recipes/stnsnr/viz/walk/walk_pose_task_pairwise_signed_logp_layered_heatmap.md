# Walk Pose Task Pairwise Signed-LogP Layered Heatmaps

This workflow renders raw walk `Pose` pairwise signed-logP heatmaps from the
Task-specific intermediate table. The layered figures use the shared
`visualdf.plot_well_heatmap_grid_df` backend.

## Input

Run the statistics workflow first:

```bash
conda run -n LFP python recipes/stnsnr/stats/walk/walk_pose_task_pairwise_signed_logp.py
```

The visualization script reads:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/walk_pose_task_pairwise_signed_logp_values.csv
```

## Figure Layout

The script writes one layered heatmap per data group and task:

- STN local, `Single`
- STN local, `Dual`
- SNr local, `Single`
- SNr local, `Dual`
- SNr-/→STN connectivity, `Single`
- SNr-/→STN connectivity, `Dual`
- Parameters, `Single`
- Parameters, `Dual`

Each figure has six vertically stacked panels in this fixed order:

1. `Sit vs Stand`
2. `Stand vs Walk`
3. `Walk vs Turn`
4. `Sit vs Walk`
5. `Stand vs Turn`
6. `Sit vs Turn`

Right-side strips label the comparison. The strip geometry and style use
`cfg.strip_right_width_mm`, `cfg.label_right_bg_color`, `cfg.strip_pad_mm`,
`cfg.label_text_color`, and `cfg.label_fontweight`.

Layered panels are rendered through the shared
`visualdf.plot_well_heatmap_grid_df` backend, matching the MNI standardized
slope heatmap workflow.

All heatmaps, including the Params heatmaps, use `3.0 x 3.0 mm` cells.
Each cell is outlined with black solid grid lines matching the med
signed-logP heatmap style (`linewidth=0.6`, `alpha=0.9`).

In addition to the original orientation, the workflow writes a second
transposed output set. For STN local, SNr local, and SNr-/→STN connectivity,
the transposed heatmaps use frequency bands on the y-axis and
parameters/metrics on the x-axis. For Params heatmaps, the transposed layout
uses `Exponent` and `Offset` on the y-axis and regions on the x-axis.

The shared colorbar is labeled
`Signed (b-a) -log10(p_Tukey) (a vs b)`. Positive values mean `b > a` for the
right-strip comparison label `a vs b`. Display values are clipped to `[-3, 3]`.

## Outputs

Heatmap files are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/walk/corr/pose_pairwise
```

Output basenames:

```text
walk_pose_task_pairwise_stn_local_single_signed_logp_layered_heatmap
walk_pose_task_pairwise_stn_local_dual_signed_logp_layered_heatmap
walk_pose_task_pairwise_snr_local_single_signed_logp_layered_heatmap
walk_pose_task_pairwise_snr_local_dual_signed_logp_layered_heatmap
walk_pose_task_pairwise_connectivity_single_signed_logp_layered_heatmap
walk_pose_task_pairwise_connectivity_dual_signed_logp_layered_heatmap
walk_pose_task_pairwise_params_single_signed_logp_layered_heatmap
walk_pose_task_pairwise_params_dual_signed_logp_layered_heatmap
```

Transposed output basenames:

```text
walk_pose_task_pairwise_stn_local_single_signed_logp_freq_by_param_layered_heatmap
walk_pose_task_pairwise_stn_local_dual_signed_logp_freq_by_param_layered_heatmap
walk_pose_task_pairwise_snr_local_single_signed_logp_freq_by_param_layered_heatmap
walk_pose_task_pairwise_snr_local_dual_signed_logp_freq_by_param_layered_heatmap
walk_pose_task_pairwise_connectivity_single_signed_logp_freq_by_param_layered_heatmap
walk_pose_task_pairwise_connectivity_dual_signed_logp_freq_by_param_layered_heatmap
walk_pose_task_pairwise_params_single_signed_logp_transposed_layered_heatmap
walk_pose_task_pairwise_params_dual_signed_logp_transposed_layered_heatmap
```

Each basename is saved as both `.pdf` and `.png`.
Rerunning the script rewrites these heatmap files in place and does not delete
other files from the output directory.

## Runtime

Use the MyLFP default Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/walk/walk_pose_task_pairwise_signed_logp_layered_heatmap.py
```

## Compact parameter-panel labels

The two-by-two parameter panels use horizontal comparison labels in a 22-mm
right strip. This prevents the six long comparison labels from overlapping
adjacent panels. Metric panels retain their existing strips. Cell dimensions,
statistical tables, significance markers, and color limits remain unchanged.
