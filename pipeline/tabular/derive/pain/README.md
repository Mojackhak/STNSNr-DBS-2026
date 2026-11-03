# Pain-Evoked Peak Derivation

`evoked_peaks.py` extracts transient peak/trough features from pain-locked
feature traces and writes long-format pain-evoked tables for downstream
statistics and visualization.

## Current QC Policy

Peak detection uses the fixed event windows defined in the detector config:

- ONS: 3.5-5.5 s, reported relative to 4.0 s.
- OFS: 13.5-15.5 s, reported relative to 14.0 s.

The detector keeps the existing LOWESS smoothing behavior:
`lowess_frac = 0.05`. For the 1001-sample, 20-s pain epoch, this corresponds
to an effective smoothing scale of about 1 s.

The post-detection QC is phase-specific:

- ONS uses only ONS boundary QC columns and a BL-based amplitude threshold.
- OFS uses only OFS boundary QC columns and a SUS/right-baseline amplitude
  threshold.

The default peak amplitude threshold is:

```text
peak_value > baseline_mean + 1.0 * baseline_sd
```

This replaces the previous linked ONS/OFS QC, where both event windows had to
pass boundary QC before either phase could be retained.

## Retention Reference

An in-memory audit on the current traces showed the following retention rates
with the phase-specific QC and OFS SUS/right-baseline amplitude threshold:

| Threshold | ONS kept | ONS % | OFS kept | OFS % |
|---:|---:|---:|---:|---:|
| 1.0 SD | 5893 / 14437 | 40.8% | 8251 / 14437 | 57.2% |
| 1.5 SD | 4463 / 14437 | 30.9% | 2288 / 14437 | 15.8% |
| 2.0 SD | 3376 / 14437 | 23.4% | 338 / 14437 | 2.3% |

The implemented default is 1.0 SD because stricter thresholds sharply reduce
OFS retention.
