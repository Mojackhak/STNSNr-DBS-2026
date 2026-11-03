# Study data formats

## Project tree

`MYLFP_PROJECT_ROOT` selects the data and analysis-output tree. Only the
subdirectories consumed by the selected workflow are needed.

```text
sourcedata/                         original acquisition inputs
rawdata/                           recording copies and supplied events
derivatives/mne/<subject>/         cleaned, stitched and tensor products
derivatives/spikeinterface/        MER processing products
summary/cohort/subj/                clinical workbooks and programming records
summary/cohort/lead/                contact/unit anatomy and coordinate tables
summary/table/raw/                  aligned cohort tables
summary/table/derive/               derived feature tables
summary/stats/<section>/            statistical inputs, models and plot components
```

Consult the actual entrypoint for stage-specific filenames. Paths in the
portable YAML refer to supplied historical outputs; normalization of their
root does not verify equivalence between old and current data trees.

## Analysis tables

Regional LFP tables commonly contain `ID`, `Subject`, `Region`, `Side`, `Hemi`,
`Lat`, `Task`, `Pose` or `Phase`, `Band`, and `Value`. Required columns vary by
model. Preserve the categorical values and condition ordering defined in
`plugin/default/tabular_default.py` and `.R`. Do not replace unavailable
observations with zeros.

PKL files can contain pandas tables with array-valued `Value` cells, rather
than a single rectangular numeric matrix. RDS inputs contain the model-ready
scalar tables. Use the retained preparation functions to construct them.
Only load PKL data from a trusted study source.

Mixed-model output CSVs include the relevant EMMs, contrasts, slopes, curves,
omnibus results and uncertainty columns. The panel YAML and index record
which tables are used. A figure's `statistics_sources` can be nested or can
list several tables for a composite.

## Correlation inputs

The walking FOGQ workflow reads `summary/cohort/subj/subj_delta_effect.xlsx`
with `ID`, `FOG-Q Δ% (STN, 3m)` and `ΔFOG-Q Δ% (+SNr, 3m)`. Those source columns
are used as stored; the incremental SNr column is not an SNr-only protocol.

Gait-PCA associations read
`summary/stats/clinic/pca/gait/dist.xlsx`. Pain-KPPS associations read both
`summary/cohort/subj/subj_scale.xlsx` and the existing clinical long table at
`summary/stats/clinic/correlation/pain/scale_subject.xlsx`. Supply the original
clinical tables or documented exports; the extraction does not reconstruct
missing patient values.

Correlation workbooks use sheets such as `Walk_r`, `Walk_p_raw`, `Walk_p_adj`
and `Walk_n`, with analogous sheets for the other supported poses or phases.
The plotting code retains the source choice of raw or adjusted P values.
The presence of an adjusted sheet does not mean that the figure used it.

## Gait-cycle inputs

`timepoint_input.csv` contains at least `ID`, `Metric`, `MetricClass`, `Band`,
`Region`, `t_pct` and `Value`, plus the context/epoch fields supplied by the
exporter. Keep repeated observations within participant identifiers.
`MetricClass` is `local` or `connectivity`. The normal study grid and region
trace preparation are defined in `plugin/default/stats_gait_cycle_default.py`.
The synthetic demo deliberately uses only two phase points and is not a
complete-cycle input.

## External or restricted inputs

Original clinical records, identifiable video, MRI/reconstruction data,
annotation decisions, detector/pose model weights, camera calibration and
third-party atlas files are not included. Data access must follow the study's
consent, ethics and availability arrangements. The code availability statement
must not imply that these data are publicly downloadable from this branch.

Manual schematics and Illustrator composites require the separately maintained
artwork and assembly instructions. Meta-analysis requires the actual extracted
study data and analysis project/code, which are absent from this source snapshot.
