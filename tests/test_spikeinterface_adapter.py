from pathlib import Path

import numpy as np
import pytest
import spikeinterface.full as si
from probeinterface import generate_linear_probe

from src.mer.spikeinterface_adapter import save_spikeinterface_objects


def _make_sorting_analyzer(base: str) -> object:
    """Build a minimal in-memory SortingAnalyzer for save tests."""
    traces = np.arange(200, dtype="float32").reshape(-1, 1)
    recording = si.NumpyRecording([traces], sampling_frequency=1000.0)
    probe = generate_linear_probe(
        num_elec=1,
        ypitch=20.0,
        contact_shapes="circle",
        contact_shape_params={"radius": 5.0},
    )
    probe.set_device_channel_indices(np.array([0]))
    recording = recording.set_probe(probe)
    recording.annotate(base=base)

    sorting = si.NumpySorting.from_samples_and_labels(
        samples_list=[np.array([10, 50, 120], dtype="int64")],
        labels_list=[np.array([1, 1, 1], dtype="int64")],
        sampling_frequency=1000.0,
    )
    return si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
        sparse=False,
    )


def test_save_spikeinterface_objects_uses_full_segment_chunks_for_filtered_recordings(
    tmp_path: Path,
) -> None:
    """Filtered recordings with a tiny final chunk should still be saved."""
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((1005, 1)).astype("float32")

    recording = si.NumpyRecording([traces], sampling_frequency=1000.0)
    recording = si.bandpass_filter(recording, freq_min=1.0, freq_max=200.0)
    recording = si.notch_filter(recording, freq=50.0)
    recording.annotate(base="L01D0F")

    save_spikeinterface_objects([recording], tmp_path)

    saved_folder = tmp_path / "L01D0F"
    loaded_recording = si.load(saved_folder)

    assert saved_folder.exists()
    assert loaded_recording.get_num_frames(segment_index=0) == 1005


def test_save_spikeinterface_objects_overwrites_existing_sorting_analyzer_zarr(
    tmp_path: Path,
) -> None:
    """Saving the same analyzer twice should replace the existing zarr output."""
    analyzer = _make_sorting_analyzer(base="LT1D-2.912F001")

    save_spikeinterface_objects([analyzer], tmp_path)
    save_spikeinterface_objects([analyzer], tmp_path)

    assert (tmp_path / "LT1D-2.912F001.zarr").exists()


def test_save_spikeinterface_objects_can_disable_overwrite(tmp_path: Path) -> None:
    """overwrite=False should fail fast when the target already exists."""
    analyzer = _make_sorting_analyzer(base="RT1D-2.500F001")

    save_spikeinterface_objects([analyzer], tmp_path)

    with pytest.raises(FileExistsError, match="Save target already exists"):
        save_spikeinterface_objects([analyzer], tmp_path, overwrite=False)
