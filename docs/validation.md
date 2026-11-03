# Extraction validation

Validation date: 2 November 2026. Source commit:
`986358b112dcd2d032a55a425702049aa3a6c5a8`. Working branch: `nn2026`.

## Inventory and source preservation

- 38 portable YAML files contain 304 source records.
- Every record has exactly one row in `figures/panel_code_map.tsv`.
- Every mapped plotting, statistics and upstream file exists in this subset.
- The original 38 YAML files outside the repository were compared with their
  extraction-time contents and were unchanged.
- The original source-file inventory retains 478 files and excludes 223 tracked
  files from the initial main snapshot. New provenance, documentation,
  environment observations and the synthetic demo are additional files.
- The local dependency map contains 1,295 retained file relationships, including
  the declared dynamic statistics configurations and lazy LFP exports.
- Required local dependency paths exist. Legacy plugin import locations were
  corrected; no absolute local import remains unresolved in the scoped check.
- No cohort analysis, participant data export, manuscript edit or figure
  replacement was performed.

## Syntax and focused tests

The 302 retained source Python files parsed successfully. Three interactive
entrypoints contain IPython magic lines; those lines were excluded from the
Python syntax check and their interactive requirement is documented. The demo
itself executed successfully. All 40 retained R files parsed successfully.
MATLAB execution was not tested.

The following focused test selection covered 32 tests:

```bash
python -m pytest \
  tests/stats/test_gait_cycle.py \
  tests/stnsnr \
  tests/posetrack/test_merge_cycle_burst_time_conversion.py \
  tests/test_spikeinterface_adapter.py -q
```

Thirty-one tests passed in the initial sandboxed run. The remaining
SpikeInterface overwrite test failed because the sandbox blocked moving its
synthetic temporary output to macOS Trash. That single test was rerun with
permission to use Trash and passed. No code or test was changed to bypass the
file-preservation behavior. Thus all 32 selected tests passed; the complete
repository test suite was not run.

SpikeInterface emitted its existing warnings that synthetic in-memory recording
objects are not serializable. These do not establish archival provenance for
real recordings and were not suppressed by changing the analysis code.

## Synthetic demonstration

The Python demonstration used a fixed NumPy seed and twelve synthetic IDs. It
produced three 2-by-3 correlation matrices, with raw/adjusted P-value and sample
count outputs, plus a PDF and PNG. The rendered PNG was inspected for readable
labels, complete panels and a visible colorbar. The measured in-process runtime
was 0.354 seconds, excluding Python/import/font-cache startup.

The R pointwise demo consumed 144 synthetic rows: twelve participants, two
regions, two phase samples and three repeated epochs. Both `lmer` and `rlmer`
outputs contained four rows, all with status `ok` and finite estimates. The
measured command runtime, including Conda/R startup, was 2.39 seconds.
These are demonstration timings on the existing validation environment, not
full-study runtimes. The two sampled phase points were not used to infer
cycle-length significant intervals.

## Environment and limits

Checks used the existing LFP environment: Python 3.12.12, R 4.5.1, macOS 26.7.1
and Apple silicon. Dependency metadata is in `env/tested_python_packages.tsv`
and `env/tested_r_packages.tsv`. Raw pose-inference dependencies are not all
installed in this environment. A fresh installation was not performed or timed.
Fontconfig reported unavailable cache directories during initial startup;
Matplotlib completed the demo, and the actual image was inspected.

This validation establishes package structure and selected executable paths.
It does not certify full numerical reproduction of all manuscript panels,
resolve missing meta-analysis code, recreate manual artwork, or establish
historical environment identity. See `reproduction_gaps.md`.
