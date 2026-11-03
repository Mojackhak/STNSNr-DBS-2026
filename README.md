# STN SNr manuscript analysis code

This branch contains the MyLFP code associated with the STN-SNr manuscript's
main, Extended Data and Supplementary figures. It includes the selected plotting
and statistical recipes, their processing dependencies, portable figure
provenance and synthetic demonstrations. The extraction starts from MyLFP
commit `986358b112dcd2d032a55a425702049aa3a6c5a8`.

Start with [the panel index](figures/panel_code_map.tsv) to find an individual
figure's inputs and code. The index covers 38 figure YAML files and 304 source
records. A code mapping is not a certification that a historical figure was
executed with this exact commit. Read the [reproduction gaps](docs/reproduction_gaps.md)
for external analyses, manual composites and other limitations.

## Repository layout

| Location | Purpose |
| --- | --- |
| `figures/provenance/` | Portable copies of the supplied figure0930 YAML files |
| `figures/panel_code_map.tsv` | Figure/panel to plotting, statistics and upstream entrypoints |
| `figures/dependencies.json` | Local import, R-source and MATLAB-helper relationships |
| `recipes/stnsnr/stats/` | Study-specific models, contrasts and correlations |
| `recipes/stnsnr/viz/` | Statistical-result and signal-feature plotting |
| `pipeline/` | Interactive preprocessing, feature generation and table preparation |
| `plugin/default/` | Study settings, categories, transformations and plotting conventions |
| `src/` | Shared processing, statistics and visualization functions |
| `env/` | Source environment recipe and observed validation dependencies |
| `examples/` | Synthetic demonstration and its instructions |
| `tests/` and `src/lfp/tests/` | Retained relevant tests and synthetic signal fixtures |

## System and installation

Python, R and MATLAB serve different parts of the workflow. Two-dimensional
plotting and statistical analysis run on a CPU; raw pose inference and the
external anatomy/sorting tools have additional requirements.

The inherited environment recipe is `env/LFP.yml`. The extraction was checked
in an existing **LFP Conda environment**, on macOS 26.7.1 / Apple silicon, with
Python 3.12.12 and R 4.5.1. This installed environment differs from some versions
pinned in the inherited recipe. The observed package versions are recorded in
[the Python inventory](env/tested_python_packages.tsv) and
[the R inventory](env/tested_r_packages.tsv); they describe this validation
session, not a verified historical environment for every manuscript figure.

For a new installation, the inherited Python environment can be created with:

```bash
conda env create -f env/LFP.yml
conda activate LFP
python -m pip install -r env/additional_python_packages.txt
```

Install the required R packages using [the R instructions](env/r_dependencies.md).
See [MATLAB and external tools](env/matlab_dependencies.md) for Lead-DBS,
Wave_clus, anatomical assets and Illustrator. Raw video tracking additionally
needs the trained detector, pose model and camera calibration used for the
recordings. Anipose, MMPose, Ultralytics, PyTorch and OpenCV are not installed
and tested as a complete video-tracking environment in this extraction. Their
historical compatible versions must be supplied with the original pose setup.

A fresh environment installation was not timed or tested. Demo runtimes and
focused test results are in [validation](docs/validation.md). No GPU was used
for those checks.

## Set paths

Run from the repository root and select an analysis data tree explicitly:

```bash
export MYLFP_REPO_ROOT="$PWD"
export MYLFP_PROJECT_ROOT=/absolute/path/to/STNSNr
export PYTHONPATH="$MYLFP_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
```

`MYLFP_PROJECT_ROOT` contains the `summary/`, `derivatives/`, `rawdata/` and
`sourcedata/` subdirectories expected by the relevant stage. Scripts write their
outputs under this selected tree. The YAML `${MYLFP_PROJECT_ROOT}` notation is
an explanatory placeholder; the original plotting scripts consume their
configured file paths, not the YAML files.

Interactive subject-level scripts also require a study-approved, de-identified
`MYLFP_SUBJECT`. The FoG recording illustration uses `MYLFP_FOG_RAW` to select
its annotated FIF recording. Do not change scientific settings merely to make
an input fit the defaults. Task configuration, exclusions, camera calibration,
recording annotations and contact assignments are study inputs.

## Run the synthetic demo

The demo contains no patient data and does not reproduce manuscript results:

```bash
export MYLFP_REPO_ROOT="$PWD"
export MYLFP_PROJECT_ROOT="$PWD/examples/output"
export MPLBACKEND=Agg
python examples/run_demo.py --output-dir examples/output
Rscript recipes/stnsnr/stats/gait_cycle/run_pointwise_lmm.R \
  --input-csv examples/output/timepoint_input.csv --jobs 1
```

Expected outputs include three 2-by-3 correlation results and their raw/adjusted
P-value and sample-size matrices, a labelled synthetic heatmap PDF/PNG, and
pointwise mixed-model tables for two regions at two synthetic phase points.
See [the example guide](examples/README.md).

## Reproduce study outputs

There are three entry levels:

1. **Existing result tables to plots.** Use the plotting entrypoint listed for
   the panel. Correlation workbooks, model CSVs and normalized traces are often
   sufficient. See [figure reproduction](docs/figure_reproduction.md).
2. **Prepared analysis tables to statistical results.** Run the corresponding
   R or Python statistics scripts before plotting. Keep the reported model,
   correction scope, contrasts and analysis units. Some plotting recipes also
   assemble derived tables; the workflow notes describe these boundaries.
3. **Recordings to analysis tables.** Follow [the upstream workflow](docs/workflows.md)
   using the original annotations, anatomical reconstruction and preprocessing
   decisions. The interactive pipelines are not unattended batch commands.

For example, with the study's existing walking inputs:

```bash
python recipes/stnsnr/stats/walk/walk_lfp_fogq_3m_correlation.py
python recipes/stnsnr/viz/walk/walk_lfp_fogq_3m_correlation_heatmap.py
```

A gait-cycle analysis uses these sequential entrypoints:

```bash
python pipeline/tabular/derive/gait_cycle/export_region_long.py
Rscript recipes/stnsnr/stats/gait_cycle/run_pointwise_lmm.R --jobs 1
Rscript recipes/stnsnr/stats/gait_cycle/summarize_intervals.R
python recipes/stnsnr/viz/gait_cycle/plot_basic.py
```

These commands require study inputs. The two-point synthetic example is not a
substitute for a full gait cycle and is not passed to the interval summarizer.

## Figure assembly and data access

Many panels are selected views, crops or combinations of several code outputs.
The manual assembly instructions remain in the YAML notes and panel index.
The code does not automatically recreate the complete Illustrator layouts.
Fig.8 is a conceptual schematic; its eight provenance records are evidence
blocks, not eight generated statistical panels.

Original recordings, clinical workbooks, anatomy, identifiable videos, atlas
assets and manual source artwork are not distributed here. Their formats and
access boundaries are described in [data formats](docs/data_format.md).
Meta-analysis code is not present in the source snapshot; the affected panels
are explicitly listed in [reproduction gaps](docs/reproduction_gaps.md).

## Scope of changes and provenance

Numerical algorithms, model formulas, statistical thresholds and plotting
encodings are retained from the selected source code. Extraction changes are
limited to dependency selection, workstation-path removal, existing import
location repairs, documentation and the synthetic demonstration.

Input changes affect the stages that consume those inputs: clinical outcomes
affect their correlations; model tables affect their plots; raw signals and
annotations affect derived features and dependent analyses. This package adds
no automatic cache invalidation, full-cohort rerun or retry mechanism.

Use [the source-file inventory](docs/code_inventory.tsv),
[extraction notes](docs/extraction_notes.md), and [validation](docs/validation.md)
to inspect the extraction boundaries. A future submission archive should use
the selected branch's source files, excluding local data, demo outputs, `tmp/`
and `.git`. Branch history is inherited from MyLFP and is not a source-only
publication archive.

## License and citation

Original code retains the [MIT license](LICENSE), copyright Ankang Hu. External
software and anatomical/artwork resources retain their own licenses. The
[Code availability draft](docs/code_availability.md) records the information
needed for the manuscript. A public repository URL and persistent identifier
must be added after an actual deposit; none is claimed by this local extraction.
