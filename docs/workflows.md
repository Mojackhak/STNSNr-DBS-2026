# Analysis workflows

## LFP processing

The normal data flow is:

`recording and events -> preprocessing -> stitched recording -> tensors -> aligned tensors -> raw tables -> derived features -> analysis tables -> models -> figures`

| Stage | Entrypoint | Inputs and outputs |
| --- | --- | --- |
| Preprocessing | `pipeline/preproc/preproc.py` | CSV/FIF recordings, reference configuration, annotations and interactive artifact/ECG decisions; produces cleaned recording files |
| Stitching | `pipeline/preproc/stitch.py` | Reviewed recordings with compatible channels; produces a stitched FIF with boundary annotations |
| Tensor computation | `pipeline/lfp/tensors.py` | Stitched FIF and selected tensor configuration; produces nested TFR, periodic/aperiodic, burst and connectivity tensors |
| Alignment | `pipeline/lfp/warp.py` | Tensors plus original event annotations and warp settings; produces event/phase-aligned tensors |
| Cohort tables | `pipeline/tabular/concat.py` | Aligned tensors and contact/anatomy metadata; produces `summary/table/raw/` tables |
| Derived features | `pipeline/tabular/derive.py` | Raw tables, band definitions, transforms and feature settings; produces derived tables |
| Statistics inputs | `pipeline/stats/stats.py` | Derived tables and section configuration; produces raw and normalized regional/subject tables |
| Models and summaries | `recipes/stnsnr/stats/<section>/` | RDS/PKL/XLSX inputs; produces model CSVs, correlations and summary matrices |
| Plotting | `recipes/stnsnr/viz/<section>/` | The mapped analysis products; produces figure components |

Run preprocessing and other `#%%` pipelines cell by cell in IPython or a
compatible editor. `preproc.py`, `stitch.py` and the FoG illustration contain
IPython magic commands and are not plain `python file.py` batch scripts.
Review configuration cells and complete manual QC before advancing. A current
default configuration is not evidence of a historical participant's settings.

The generic statistics entrypoint supports `MYLFP_STATS_CFG`,
`MYLFP_STATS_SECTIONS`, `MYLFP_STATS_STEMS` and `MYLFP_STATS_JOBS` to select
existing configurations and limit an explicitly requested run. Walking and
pain R recipes have their existing stem selectors; the gait-feature connection
recipe also has its existing section selector. No scheduler or global rerun
wrapper is included.

## Gait and FoG

`pipeline/posetrack/gait_track.py` consumes synchronized videos, camera
calibration, a trained detector and pose model configuration. The subsequent
cycle/turn feature scripts derive kinematics. `merge_cycle.py` and
`merge_turn.py` join those features to electrophysiology; burst durations/rates
use the retained cycle-duration conversion functions.

Gait event and gait phase recipes are separate from the gait-cycle pointwise
workflow. For the latter, `export_region_long.py` creates per-band long tables,
`run_pointwise_lmm.R` fits the pointwise models, `summarize_intervals.R` describes
same-direction sequences, and `plot_basic.py` plots traces and annotations.
Stars based on a minimum pointwise P value are not interval-level tests.

FoG episode windows and alignment are governed by the retained configuration.
Publication schematic timelines and actual event axes must be reconciled
before claiming that a regenerated panel is identical to its illustrated
source. See `reproduction_gaps.md`.

## Mechanical stimulation

The pain workflow shares LFP processing, then uses the pain section's
baseline/phase/force definitions. `pipeline/tabular/derive/pain/evoked_peaks.py`
extracts the evoked waveform features consumed by `pain_evoked` statistics and
plots. Force is a physical stimulus parameter; these inputs do not constitute
trial-wise subjective pain ratings.

## Clinical outcomes

The clinical source workbooks and programming records are required inputs.
`subject_effect_tables.py` derives the established subject outcome tables.
`table_effect.py`, `table_scale_effect.py`, `table_pca.py` and
`table_scale_pca.py` prepare the clinical/gait/balance data products. The PCA
plotting recipes also compute/export PCA scores and projection distances;
their execution is an analytical step, not a purely cosmetic redraw.

Clinical scale PCA uses `scale_pca.py`; gait/balance use `pca.py` with its
feature selection cell. The corresponding R distance models and Python
`pca_dist.py`/`pca_scale_dist.py` plots depend on those outputs.
The effect R models and `scale_effect.py` plots require the prepared condition tables. Some
clinical plotting code rewrites derived comparison CSVs, so use the intended
analysis output tree rather than an immutable source-data deposit.

The active-contact workflow uses the actual programming JSON and contact
reconstruction. Its inherited entrypoint also exports a separately named
random-stimulation table. That synthetic table is not a clinical programming
record and is not evidence for Fig.1c.

## MER and anatomy

`pipeline/mer/wc_sorting.py` is an interactive NeuroOmega/NeuroExplorer and
Wave_clus workflow. Manual sorting/classification decisions and the recording
files are required. `pipeline/mer/coords.py` assembles unit and coordinate
tables for the MER statistics and waveform examples.

Lead-DBS reconstruction, transforms and atlas assets are inputs to the anatomy
pipeline. `lead_flip.m` calls the nonlinear left-to-right coordinate mapping;
it is not equivalent to changing the sign of MNI-x. Spatial model recipes,
volume generation and MATLAB rendering are distinct stages. The source file
list retains the local helpers; external Lead-DBS routines and atlases are not
vendored.
