# LFPscope Time-Frequency Refactor Bundle

This bundle splits the original single-file implementation into a maintainable
package layout:

- `lfp/tfr/`             : TFR computation + (optional) SpecParam decomposition
- `lfp/connectivity/`    : time-frequency connectivity aligned to the TFR time grid
- `lfp/burst/`           : burst-amplitude tensors (bandpass+Hilbert+threshold)
- `lfp/psi/`             : Phase Slope Index (PSI) tensors over time
- `lfp/mask/`            : annotation-based time-axis masking (set outside segments to NaN)
- `lfp/warp/`            : warpers (gait anchored-linear; pad+concat) + metadata
- `lfp/common/`          : shared helpers (freq grid, decimation, Morlet cycles)
- `lfp/pipelines/`       : Raw -> tensor -> mask -> warp pipelines
  - `lfp.pipelines.masking.get_custom_warped_tensors(...)` now provides the
    high-level "mask, warp, QC, and save" helper for annotation-specific
    warped tensor exports.

## Quick usage

```python
import mne
from lfp import gait_warper, mask_by_annotations, tfr_grid

raw = ...  # mne.io.Raw
cycles, warp_fn = gait_warper(raw)

tfr_out = tfr_grid(raw, method="morlet", time_resolution_s=0.10, hop_s=0.05, average=False)
tensor = tfr_out["power"][0]  # (n_channels, n_freqs, n_times)
times = tfr_out["metadata"]["axes"]["times"]

# Keep only time bins covered by these annotations (everything else -> NaN).
tensor_masked, time_mask, mask_info = mask_by_annotations(raw, tensor, times, keep=["sit", "gait"])

# Then warp (example: 0..100% gait cycle axis).
sr_time = 1.0 / (times[1] - times[0])
warped, pct, meta = warp_fn(tensor_masked, sr=sr_time, n_samples=501)
```

If you need a single call:

```python
from lfp.pipelines import compute_raw_tfr_and_warp

out = compute_raw_tfr_and_warp(raw, warp_fn=warp_fn, n_samples=501, keep_annotations=["sit", "gait"])
warped_tensor = out["tensor_warped"]
warped_meta = out["metadata_warped"]
```


Additional pipelines:

```python
from lfp.pipelines import compute_raw_burst_and_warp, compute_raw_psi_and_warp

# Burst
out_burst = compute_raw_burst_and_warp(
    raw,
    bands={"beta": (13, 30)},
    warp_fn=warp_fn,
    keep_annotations=["sit", "gait"],
)

# PSI
out_psi = compute_raw_psi_and_warp(
    raw,
    bands={"beta": (13, 30)},
    time_resolution_s=0.10,
    warp_fn=warp_fn,
    keep_annotations=["sit", "gait"],
    ordered_pairs=True,
)
```

## Connectivity GC padding

For multivariate GC calls in `lfp.connectivity.grid.grid(...)`:

- `gc_pad_scale` now defaults to `3.0`.
- The GC padding target is `ceil(gc_pad_scale * gc_min_freqs)`.
- If the available full-grid size is still smaller than the estimator hard
  minimum `gc_min_freqs`, the runtime still raises an error.
- Otherwise, if the scaled target is larger than the available full-grid size,
  the runtime caps the padding target to the maximum available frequency-bin
  count and pads as far as the current grid allows.

## Connectivity window grouping

Connectivity window grouping in `lfp.connectivity.grid.grid(...)` is controlled
by two parameters:

- `group_by_samples=True` groups neighboring frequencies when their effective
  window lengths collapse to the same integer number of samples.
- `group_by_samples=False` instead rounds each effective window duration onto a
  millisecond grid controlled by `round_ms`.
- `round_ms` is only used when `group_by_samples=False`; it must be strictly
  positive in that mode.

Changing `round_ms` alone therefore has no effect unless the caller also
disables sample-count grouping.

## Warp annotation matching

The warpers under `lfp.warp` now expose separate controls for positive-match
and drop-match behavior:

- `concat_warper(...)`, `stack_warper(...)`, and `pad_warper(...)` accept
  `mode` for selecting kept annotations and `drop_mode` for `anno_drop`
  filtering, with both modes supporting case-insensitive `exact` and
  `substring` matching.
- `linear_warper(...)` accepts `anno_drop` with its own `drop_mode`, so
  candidate event-anchored epochs can be rejected when they overlap blocked
  annotations without changing how anchor events are matched.
- `linear_warper(...)` now keeps stable per-anchor metadata keys even when the
  same annotation token appears at multiple target percentages, and it collects
  every valid epoch candidate instead of stopping after the first match.
