# gait_cycle visualization

This directory renders gait-cycle trace figures from the `gait_cycle` preprocessing and pointwise LMM outputs.

## Inputs

The visualization script reads each parameter-specific gait-cycle directory:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/gait_cycle/<param>/<subparam>/<subparam>-mean-trace/<Band>
```

It uses the nested Region-level normalized trace table and the interval tables saved under the band-level model directories.

## Regeneration

Run from the repository root with the `LFP` Conda environment:

```bash
conda run -n LFP python recipes/stnsnr/viz/gait_cycle/plot_basic.py
```

Figures are rendered through `plugin.default.viz_defaults.plot_series_wrapper()`, which calls `src.viz.visualdf` with the default gait trace style. The gait-cycle script disables default gait-event grey vertical shadows while keeping event reference vertical lines; only significant intervals with continuous span `>= 10%` are drawn as vertical shadows with model-specific asterisks.

Asterisks mark baseline-deviation direction: intervals significantly above baseline are labeled inside the top of the plot box, and intervals significantly below baseline are labeled inside the bottom with a tight bottom margin. Asterisk colors match the corresponding Region line. Multiple intervals use separate lanes within the plot box to avoid label overlap.

Vertical shadows use the corresponding Region curve color with alpha `0.15` and no patch outlines, so interval boundaries are not rendered as same-color edge lines.

Figures are written into the corresponding model subdirectories:

```text
<Band>/lmer_model/Region_norm.pdf
<Band>/rlmer_model/Region_norm.pdf
```
