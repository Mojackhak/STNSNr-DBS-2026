# Selected surface rendering workflows

The publication subset retains surface entrypoints for MER, walking and
mechanical-stimulation spatial panels. The corresponding volume and coordinate
model scripts are listed for each panel in `figures/panel_code_map.tsv`.

Set the repository and project environment variables, initialize Lead-DBS and
its required atlas, then run the appropriate MATLAB entrypoint:

```matlab
run(fullfile(getenv('MYLFP_REPO_ROOT'), 'recipes', 'stnsnr', 'viz', 'mer', 'surface_mer.m'));
run(fullfile(getenv('MYLFP_REPO_ROOT'), 'recipes', 'stnsnr', 'viz', 'walk', 'surface_walk.m'));
run(fullfile(getenv('MYLFP_REPO_ROOT'), 'recipes', 'stnsnr', 'viz', 'pain', 'surface_pain.m'));
```

These are separate selected-section commands, not a recommendation to run all
three for every panel. The scripts use shared code in `src/viz/surface/batch/`,
`batch_helper/` and `render/`. Preserve the supplied mask, interpolation,
smoothing, color limits and camera/view settings when reproducing a source
panel. The original view or manual crop remains part of figure assembly.

The MER coverage figure has its own `spike_topomap.py` and `spike_topomap.m`
chain. Pain contact coverage uses `volume_cover.py` and `volume_cover.m`.
These depend on coordinate tables and external atlas files; those resources
are not distributed in this code subset. MATLAB rendering was not rerun during
extraction. See `env/matlab_dependencies.md` and `docs/reproduction_gaps.md`.
