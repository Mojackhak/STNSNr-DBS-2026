from pathlib import Path

import numpy as np
import pandas as pd

import plugin.default.stats_gait_cycle_default as gait_cfg
from src.io.pkl_io import save_pkl
from src.stats.gait_cycle import (
    adjacent_mean_resample_101_to_100,
    build_region_trace_frame_for_path,
    trace_frame_to_long_frame,
)


def _params_value(offset_shift: float = 0.0) -> pd.DataFrame:
    time = np.arange(101, dtype=float)
    return pd.DataFrame(
        [time + offset_shift, (time * 2.0) + offset_shift],
        index=["offset", "exponent"],
        columns=time,
    )


def test_adjacent_mean_resample_101_to_100_centers() -> None:
    values = np.arange(101, dtype=float)
    out = adjacent_mean_resample_101_to_100(values)

    assert len(out) == 100
    np.testing.assert_allclose(out, gait_cfg.CYCLE_BIN_CENTERS)


def test_region_trace_frame_aggregates_contact_and_retains_repeated_columns(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw" / "gait"
    raw_path = input_root / "params" / "params" / "params-na-raw.pkl"
    raw_path.parent.mkdir(parents=True)

    rows = [
        {
            "ID": "SNr001",
            "Subject": "Sub-One",
            "Contact": "0_1",
            "Region": "SNr",
            "Side": "Left",
            "Side_cycle": "Right",
            "Hemi": "LAH",
            "Lat": "Contra",
            "Task": "Single",
            "Epoch": 0,
            "Value": _params_value(0.0),
        },
        {
            "ID": "SNr001",
            "Subject": "Sub-One",
            "Contact": "1_2",
            "Region": "SNr",
            "Side": "Left",
            "Side_cycle": "Right",
            "Hemi": "LAH",
            "Lat": "Contra",
            "Task": "Single",
            "Epoch": 0,
            "Value": _params_value(10.0),
        },
        {
            "ID": "SNr001",
            "Subject": "Sub-One",
            "Contact": "2_3",
            "Region": "SNr",
            "Side": "Left",
            "Side_cycle": "Right",
            "Hemi": "LAH",
            "Lat": "Contra",
            "Task": "Dual",
            "Epoch": 0,
            "Value": _params_value(100.0),
        },
    ]
    save_pkl(pd.DataFrame(rows), raw_path)

    trace_frame = build_region_trace_frame_for_path(raw_path, input_root=input_root)

    assert len(trace_frame) == 2
    assert "Contact" not in trace_frame.columns
    assert set(trace_frame["Band"]) == {"Offset", "Exponent"}
    assert set(trace_frame["Epoch"].astype(str)) == {"0"}
    assert set(trace_frame["Side"]) == {"Left"}
    assert set(trace_frame["Side_cycle"]) == {"Right"}

    offset_trace = trace_frame.loc[trace_frame["Band"] == "Offset", "Value"].iloc[0]
    np.testing.assert_allclose(offset_trace.index.to_numpy(dtype=float), gait_cfg.CYCLE_BIN_CENTERS)
    np.testing.assert_allclose(offset_trace.to_numpy(dtype=float), gait_cfg.CYCLE_BIN_CENTERS - 50.0)

    long_frame = trace_frame_to_long_frame(trace_frame)
    assert len(long_frame) == 200
    assert long_frame["t_pct"].nunique() == 100
    assert set(long_frame["Band"]) == {"Offset", "Exponent"}


def test_trace_discovery_excludes_appledouble_sidecars(tmp_path: Path) -> None:
    from recipes.stnsnr.viz.gait_cycle.plot_basic import discover_trace_paths

    trace = tmp_path / "raw-mean-trace_region_normalized.pkl"
    sidecar = tmp_path / "._raw-mean-trace_region_normalized.pkl"
    trace.write_bytes(b"trace placeholder")
    sidecar.write_bytes(b"\x00AppleDouble metadata")
    assert discover_trace_paths(tmp_path) == (trace,)
