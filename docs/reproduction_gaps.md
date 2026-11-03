# Reproduction gaps and boundaries

## Missing meta-analysis implementation

The source snapshot contains no located meta-analysis/forest-plot implementation.
Twenty-one panel records are affected: Fig.1j,k; Supplementary Fig.2b-m; and
Supplementary Fig.3a-g. Supplementary Fig.2a is the manually prepared study-flow
schematic. Supply the original extraction dataset and analysis project/code
before claiming that this package reproduces the pooled results.

The current manuscript's provisional SMD wording is not implemented by this
extraction. No pooled effect, study weight or forest plot was recalculated.

## Manual and external sources

Twenty-six YAML records have no `image_source`; this includes the 21 statistical
records above and five manually prepared/anatomical records. Other populated
sources also include manual composites, screenshots and conceptual artwork.
These are classified in the panel index rather than being assigned invented
plotting code.

Fig.7b names
`pain_phase_pairwise_local_params_signed_logp_layered_heatmap_bl_comparisons.pdf`.
The current selected renderer has local-only and connectivity subset exports;
it does not expose that exact historical local-plus-parameter export name.
Its source summary CSV and related code are retained. The historical composite
must be supplied or its manual assembly documented before claiming exact output
reproduction.

Fig.1 clinical PCA panels also use assembled `pca_combine`/`radar_combine`
exports. The component renderers and analytical steps are retained, while the
final assembly remains manual. Supplementary Fig.25's Walk-only composition is
specified in its YAML notes.

## Historical data and settings

The supplied YAML predominantly references the old VAL data tree. Current main
has a mixture of later project roots and interactive example selections.
Portable paths make the code relocatable; they do not prove that old and current
inputs, settings or outputs are identical. Preserve the original publication
inputs and recorded analysis settings when executing a reproduction.

The copied provenance retains source notes, including wording that may predate
later manuscript edits. In particular, Fig.1c's legacy frequency descriptions
need reconciliation with the final stimulation labels. The FoG timeline artwork
also needs reconciliation with the confirmed epoch boundaries. This package
changes neither scientific figures nor their source YAML outside the repository.

## Environment and external workflows

The inherited `env/LFP.yml` pins some versions differently from the current
installed LFP environment used for focused checks. Both records are provided.
The local `cmcrameri` installation reports distribution version `0.0.0`, so that
metadata value is not used as a PyPI installation pin. A clean installation,
full historical environment reconstruction, raw video inference, manual MER
sorting, MATLAB/Lead-DBS rendering and full-cohort execution have not been
performed for this extraction.

Pose model weights/calibration, anatomical atlases/reconstructions, original
annotations and restricted participant data remain external requirements.
Package metadata lists available installed versions; listing a dependency is
not evidence that its entire workflow was executed.

## Publication deposition

A public repository URL, persistent identifier and the final data-access route
are not assigned by creating a local branch. Complete those details after the
actual deposit. The source-only archive should exclude Git history, local data,
run outputs and temporary receipts. No remote publication was performed.
