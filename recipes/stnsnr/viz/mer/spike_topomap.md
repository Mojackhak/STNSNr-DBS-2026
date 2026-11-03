# MER Spike Coverage Topomap

This recipe builds a coordinate coverage topomap for MER spike units in the STNSNr
project. It mirrors the `volume_cover.py` workflow: each coordinate is assigned a
constant plotting value and then voxelized into an SNr mask for surface rendering.

## Input

The coordinate source is:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/mer/spike/spike_unit_coords.pkl
```

The script keeps rows with valid `MNI_x_flip`, `MNI_y_flip`, and `MNI_z_flip`
coordinates. The source file currently contains only `SNr` rows, so the rendered
coverage map is SNr-only.

## Value Assignment

The coverage map does not encode spike feature magnitude. Every retained coordinate
gets:

```text
Value = 1.0
```

This makes the NIfTI and rendered surface show where MER spike units are present.

## Outputs

Outputs are written to:

```text
${MYLFP_PROJECT_ROOT}/summary/stats/mer/spike/topomap/cover/SNr
```

Expected files:

```text
spike_unit_cover.csv
spike_unit_cover.pkl
spike_unit_cover_log.pkl
spike_unit_cover.nii.gz
cover1.png
cover2.png
```

## Run

Generate the voxelized coverage map:

```sh
conda run -n LFP python ${MYLFP_REPO_ROOT}/recipes/stnsnr/viz/mer/spike_topomap.py
```

Render the two SNr surface views:

```sh
matlab -batch "run('${MYLFP_REPO_ROOT}/recipes/stnsnr/viz/mer/spike_topomap.m')"
```
