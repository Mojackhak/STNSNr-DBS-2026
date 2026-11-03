# `src/lfp/tests` — Synthetic ground-truth unit & integration tests

This folder contains **research-grade, deterministic tests** for the DBS-LFP feature pipeline in `src/lfp/`.

These tests do **not** rely on real recordings (real data rarely has a clean, unambiguous “correct answer”).
Instead, they generate small **synthetic MNE `Raw` objects with known ground truth** and verify that the
pipeline recovers the expected structure and patterns after masking, dropping artifacts, and warping.

## 1) High-level testing strategy

The LFP pipeline produces high-dimensional tensors (e.g., `channel × freq × time` or `epoch × channel × freq × percent`),
so “correctness” is best tested with signals where the expected outputs are known in advance.

The strategy is:

1. **Generate** a compact `mne.io.RawArray` using `synthetic_raw.py`:
   - Known spectral content (beta/theta/gamma components)
   - Known inter-channel coupling and directionality (delay-based ground truth)
   - Rich annotations required by the pipeline (`sit`, `gait`, `walk`, `pain`, gait events, pain labels, and random `BAD_*`)

2. **Compute** tensors with the same building blocks used by the real pipeline:
   - TFR (`src/lfp/tfr/grid.py`)
   - Connectivity (`src/lfp/connectivity/grid.py`)
   - PSI (`src/lfp/psi/grid.py`)
   - Burst (`src/lfp/burst/grid.py`)

3. **Mask** and **drop** data using annotation-aware utilities:
   - `src/lfp/mask/annotations.py`
   - `src/lfp/pipelines/masking.py` (`mask_tensors`, `drop_tensors`)

4. **Warp** tensors using paradigm-specific warpers:
   - `concat_warper` for concatenation paradigms (`sit`, etc.)
   - `pad_warper` for pad-style events (`pain`, etc.)
   - `gait_warper` for event-anchored gait cycles

5. **Assert** that outputs match ground truth in a robust way:
   - Prefer **relative comparisons** and **shape/trend checks** over brittle absolute thresholds
   - Example: “beta power early > beta power late” rather than “beta power equals X”

This approach is designed to catch **silent bugs** where array shapes look fine but semantics are wrong.

## 2) How to run

From the project root (the directory that contains `src/`):

```bash
# Run all LFP tests
pytest -q src/lfp/tests

# Run the integration-style warp tests only
pytest -q src/lfp/tests/test_pipeline_warp_synthetic.py

# Run connectivity + directionality tests (requires mne-connectivity)
pytest -q src/lfp/tests/test_synthetic_connectivity_metrics.py
```

### Dependencies

Required:
- `pytest`
- `numpy`
- `scipy`
- `mne`

Optional:
- `mne-connectivity`

Tests that require `mne-connectivity` use `pytest.importorskip("mne_connectivity")`,
so they will be skipped automatically if the package is not installed.

## 3) What the synthetic Raw contains (ground truth)

`src/lfp/tests/synthetic_raw.py` returns:

- `raw`: an `mne.io.RawArray` with annotations
- `truth`: a small dataclass describing the known generative parameters

### 3.1 Signal components (known spectral structure)

The synthetic signal is constructed by adding sinusoids with simple envelopes plus noise:

- **Beta** (default 20 Hz): present throughout, but the amplitude depends on the paradigm
  - `sit`: strong, constant beta amplitude
  - `pain`: beta differs across pad-left / mid / pad-right regions
  - `gait`: beta ramps linearly across the full gait interval
- **Theta** (default 6 Hz): injected during `gait`
- **Gamma** (default 60 Hz): injected during `pain`

Internally the signal is generated in a dimensionless form and then scaled to microvolt-order **Volts**
to match typical MNE conventions.

### 3.2 Channel coupling & directionality (known connectivity structure)

The generator creates a controlled relationship between channels:

- Channel 1 is a **causally delayed** version of Channel 0 (no circular wrap-around)
  - This yields strong undirected coupling (e.g., coherence/PLV/PLI/ciPLV/wPLI) in beta
  - Directed metrics (e.g., GC / PSI) should prefer the true delay direction
- Additional channels (if present) are mostly independent noise (low coupling)

### 3.3 Annotations (known behavioral structure)

The `raw.annotations` include at least:

- `sit`, `gait`, `walk`, `pain` intervals
- A gait event cycle with:
  - `strike_L`, `strike_R`, `off_L`, `off_R` repeated across cycles
- Pain labels (e.g., `L26 pain`) and optional extra pain label markers
- Random `BAD_*` annotations (deterministic under a fixed RNG seed)

These are the minimum vocabulary elements required by the masking/dropping/warping utilities.

## 4) What is covered (by test file)

Below is a practical map from **tests → modules → what is being validated**.

### 4.1 `test_synthetic_raw_factory.py`

Depends on:
- `src/lfp/tests/synthetic_raw.py`

Validates:
- Channel count/names/sfreq and realistic microvolt-order scale
- Required annotation vocabulary exists (`sit`, `gait`, `walk`, `pain`, gait events, `BAD_*`)
- Spectral signatures are consistent with the paradigm
  - Sit: beta dominates
  - Pain: gamma is strongly elevated
  - Gait: theta is elevated relative to sit

### 4.2 `test_mask_annotations.py`

Covers:
- `src/lfp/mask/annotations.py`

Validates:
- `time_mask_by_annotations()` produces a correct boolean mask
- `mask_by_annotations()` respects `overlap_policy="split"`

### 4.3 `test_burst_grid.py`

Covers:
- `src/lfp/burst/grid.py`

Validates:
- Output shapes match the time grid contract
- Metadata axes (especially `time`) are consistent and monotonic

### 4.4 `test_psi_grid.py` (optional dependency)

Covers:
- `src/lfp/psi/grid.py`

Validates:
- Output tensor shape matches metadata axes
- Pair/channel axis records the requested pairs

### 4.5 `test_warp_pad_concat.py`

Covers:
- `src/lfp/warp/pad_concat.py`

Validates:
- `pad_warper()` identifies labeled pain epochs and returns a working `warp_fn`
- `warp_fn` contract: returns `(warped, meta_epochs)` with consistent epoch counts

### 4.6 `test_warp_event_anchored.py`

Covers:
- `src/lfp/warp/event_anchored_linear.py`

Validates:
- `gait_warper()` detects gait cycles from strike/off events
- The returned `warp_fn` produces warped output with consistent epoch metadata

### 4.7 `test_synthetic_connectivity_metrics.py` (optional dependency)

Covers:
- `src/lfp/connectivity/grid.py` (coh/plv/pli/ciplv/wpli/gc)
- `src/lfp/psi/grid.py` (PSI)

Validates:
- Coupled pair (ch0–ch1) shows stronger coupling than an unrelated pair (ch0–ch2) in beta
- Directed metrics reflect the known delay direction (A→B differs from B→A)

### 4.8 `test_pipeline_warp_synthetic.py` (integration-style)

Covers:
- `src/lfp/tfr/grid.py`
- `src/lfp/pipelines/masking.py` (`mask_tensors`, `drop_tensors`)
- `src/lfp/pipelines/warping.py` (`warp_tensors`)
- `src/lfp/warp/concat.py`
- `src/lfp/warp/pad_concat.py`
- `src/lfp/warp/event_anchored_linear.py`

Validates (semantic ground truth):
- **Concat warping (sit)**: warped beta level is consistent with sit-only beta level (within tolerance)
- **Pad warping (pain)**: warped beta shows the intended left-vs-right contrast
- **Gait warping**: warped beta power follows the expected ramp trend after event-anchored normalization

## 5) What “passing tests” means

Think of correctness in two layers:

### 5.1 Structural correctness

Passing tests provide evidence that:

- Tensor shapes, axis ordering, and metadata axes match the documented contract
- Annotation-to-time alignment works (mask/drop are consistent with tensor time axes)
- Warpers return `warp_fn` functions compatible with `warp_tensors`

### 5.2 Semantic correctness (under controlled ground truth)

Under the synthetic ground truth assumptions, passing tests provide evidence that:

- Masking/dropping/warping does not destroy core spectral patterns
- Warping is not only “a successful resample”, but aligns the intended within-paradigm structure
- Connectivity/directionality methods recover expected *relative* relationships in a controlled case

**Important:** This does not prove the full pipeline is “always correct” on real DBS-LFP.
It shows that core logic works on a deterministic dataset with known answers.

## 6) What is NOT covered (do not over-interpret)

These tests are not intended to fully validate:

- Real stimulation artifacts (harmonics, saturation, non-stationary artifacts)
- SpecParam / FOOOF-like decomposition correctness (`src/lfp/tfr/decompose.py`)
- Performance, parallel determinism, or memory limits on large datasets
- I/O correctness (BIDS layout, `.fif`/`.pkl` round-trips, stitching across sessions)
- Full coverage of complex or conflicting annotations (overlaps, missing events, irregular gait)

## 7) Debugging tips

- If a gait warp test shows “most epochs look fine, but one epoch is inverted/negative trend”,
  suspect **non-monotonic anchors** or **incorrect anchor bookkeeping** (e.g., a repeated key overwriting another anchor).
- In failures, inspect:
  - `warped["..."]["meta"]["warp_epochs"]` (start/end times and anchor times)
  - Whether anchor times are strictly increasing in sample indices before interpolation

## 8) How to extend this test suite

When adding new tests:

1. Encode the new ground truth into `synthetic_raw.py` (envelope/phase/delay changes).
2. Add assertions based on **trend / contrast / direction**, not fragile absolute values.
3. Keep tests small and deterministic (fixed RNG seeds).
