# Synthetic demonstrations

The examples contain simulated observations only. They demonstrate the retained
statistical and plotting interfaces; they do not reproduce the manuscript's
effect sizes, P values, patient counts or figures.

Run commands from the repository root in the `LFP` environment:

```bash
export MYLFP_REPO_ROOT="$PWD"
export MYLFP_PROJECT_ROOT="$PWD/examples/output"
export MPLBACKEND=Agg
python examples/run_demo.py --output-dir examples/output
Rscript recipes/stnsnr/stats/gait_cycle/run_pointwise_lmm.R \
  --input-csv examples/output/timepoint_input.csv --jobs 1
```

The Python example uses the publication FOGQ correlation and heatmap functions
with twelve synthetic participants, two synthetic outcomes and three LFP
features across the supported pose labels. It writes input CSVs, correlation
matrices (including raw and BH-adjusted P values), a PDF and a PNG. It also
creates a synthetic long-form gait-cycle table with two sampled phase points.

The R command fits the existing local `Value ~ Region + (1 | ID)` models with
`lmer` and `rlmer`, producing `timepoint_deviation.csv` and model-specific
tables. Two sampled phase points are sufficient to exercise the model interface,
but are not a complete gait cycle and must not be interpreted as significant
gait-cycle intervals. No interval summary is run on this demonstration.

The expected output dimensions, measured runtime and validation results are
reported in `docs/validation.md` after execution. Demo outputs are ignored by
Git. Real study inputs must be obtained through the manuscript's data-access
procedure and arranged as described in `docs/data_format.md`.
