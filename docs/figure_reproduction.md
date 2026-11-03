# Figure reproduction guide

## Locate a panel

Open `figures/panel_code_map.tsv` and select its `figure` and `panel`. Read
`image_sources`, `input_paths`, the script columns and the selection/assembly
notes together. Execute dependencies in the order described in `workflows.md`.
The table is an evidence-based source map, not an automatic execution manifest.

`mapping_evidence` distinguishes an explicit YAML script reference from a match
to an output directory, metric family or save-name pattern. Historical execution
and numerical equivalence have not been certified by filename matching.

## Plot components from existing results

For ordinary raw, spectral, trace and scalar plots, select the matching
`single_*` or `double_*` recipe. Local features use `single_*`; coupling uses
`double_*`. Scalar recipes read model tables. The `raw` output type denotes a
time-frequency representation and should not be confused with unprocessed
acquisition data.

Heatmap recipes read their corresponding summary CSVs or workbooks. For example:

```bash
python recipes/stnsnr/stats/walk/walk_pose_task_pairwise_signed_logp.py
python recipes/stnsnr/viz/walk/walk_pose_task_pairwise_signed_logp_layered_heatmap.py
```

The first command summarizes already fitted contrasts; it does not replace the
underlying R fits. Follow the per-panel statistics column when those fits are
not available. For correlation heatmaps, the Python statistical recipe performs
the correlation analysis itself.

## Statistical chains

| Output family | Existing entrypoint sequence |
| --- | --- |
| Clinical effects | Clinical table preparation -> `stats/clinic/scale_effect.R` -> `viz/clinic/scale_effect.py` |
| Clinical/gait/balance PCA | Corresponding table preparation -> `viz/clinic/scale_pca.py` or `pca.py` -> matching R distance model -> matching distance plot |
| Local/coupling scalar contrasts | `stats/<section>/single_scalar.R` or `double_scalar.R` -> matching scalar plot |
| Spatial slopes and surfaces | Coordinate model -> selected volume/heatmap script -> MATLAB surface renderer if applicable |
| Gait/turn feature associations | `stats/<section>/single_feature.R` or `double_feature.R` -> gradient summary or feature-fit plots |
| Gait-cycle traces | Region-long exporter -> pointwise R models -> interval summarizer -> `viz/gait_cycle/plot_basic.py` |
| Walking clinical associations | FOGQ/Gait-PCA correlation recipe -> corresponding heatmap recipe |
| Mechanical evoked features | Evoked-peak extraction -> `pain_evoked` models -> summary heatmap/scalar plots |

Paths in this table are under `recipes/stnsnr/` unless an upstream entrypoint is
named. Detailed filenames and condition selections are in the panel index.

## Manual assembly

- Supplementary Fig.25 combines only the **Walk** panels from the FOGQ and
  compact Gait-PCA source heatmaps, using the exact rows recorded in its YAML.
  Stand/Turn blocks and the unused Gait-PCA comparison are excluded during
  assembly. Labels and a shared colorbar were rearranged manually.
- Sources with `copy` or `combine` in their filenames may be selected or assembled
  exports. Retain their YAML notes and do not equate a similarly named base
  export with the final panel.
- Surface panels require the specified atlas, geometry, coordinates and view.
  Code availability does not supply or license these external resources.
- Example recordings and videos require original selections and, where stated,
  manually drawn masks. No participant video is included in the demonstration.
- Fig.8 is manually authored conceptual artwork. Its provenance links to
  supporting results; it has no independent statistical generation step.
- Final multi-panel layouts remain an Illustrator assembly step. The optional
  `tools/export_ai_to_layered_pdf.jsx` exports existing AI artwork; it does not
  generate scientific data or reconstruct absent artwork.
