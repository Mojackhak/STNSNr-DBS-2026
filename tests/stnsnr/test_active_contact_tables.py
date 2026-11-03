from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


CLINIC_STATS_DIR = (
    Path(__file__).resolve().parents[2] / "recipes" / "stnsnr" / "stats" / "clinic"
)
sys.path.insert(0, str(CLINIC_STATS_DIR))

import active_contact_tables  # noqa: E402


def test_build_scale_contact_only_marks_ulf_from_frequency() -> None:
    active_contacts = pd.DataFrame(
        {
            "ID": ["SNr001", "SNr001", "SNr002"],
            "Subject": ["Sub-Alpha", "Sub-Alpha", "Sub-Beta"],
            "Contact": [0, 4, 8],
            "MNI_x_flip": [1.0, 2.0, 3.0],
            "MNI_y_flip": [-1.0, -2.0, -3.0],
            "MNI_z_flip": [-4.0, -5.0, -6.0],
        }
    )
    contact_parameters = pd.DataFrame(
        {
            "ID": ["SNr001", "SNr001", "SNr002", "SNr001"],
            "Phase": ["3m", "3m", "3m", "immediate"],
            "Protocol": ["STN+SNr", "STN+SNr", "STN+SNr", "STN+SNr"],
            "Contact": [0, 4, 8, 0],
            "Target": ["SNr", "STN", "SNr", "SNr"],
            "Frequency": [30, 125, 130, 30],
        }
    )

    result = active_contact_tables.build_scale_contact_only(
        active_contacts,
        contact_parameters,
    )

    assert list(result.columns) == [
        "Subject",
        "MNI_x_flip",
        "MNI_y_flip",
        "MNI_z_flip",
        "ULF",
    ]
    assert result["ULF"].tolist() == [True, False, False]


def test_build_scale_contact_only_rejects_missing_contact_parameter() -> None:
    active_contacts = pd.DataFrame(
        {
            "ID": ["SNr001"],
            "Subject": ["Sub-Alpha"],
            "Contact": [0],
            "MNI_x_flip": [1.0],
            "MNI_y_flip": [-1.0],
            "MNI_z_flip": [-4.0],
        }
    )
    contact_parameters = pd.DataFrame(
        {
            "ID": ["SNr001"],
            "Phase": ["3m"],
            "Protocol": ["STN"],
            "Contact": [0],
            "Frequency": [130],
        }
    )

    with pytest.raises(ValueError, match="Missing follow-up stimulation rows"):
        active_contact_tables.build_scale_contact_only(
            active_contacts,
            contact_parameters,
        )


def test_build_scale_contact_only_rejects_duplicate_contact_parameter() -> None:
    active_contacts = pd.DataFrame(
        {
            "ID": ["SNr001"],
            "Subject": ["Sub-Alpha"],
            "Contact": [0],
            "MNI_x_flip": [1.0],
            "MNI_y_flip": [-1.0],
            "MNI_z_flip": [-4.0],
        }
    )
    contact_parameters = pd.DataFrame(
        {
            "ID": ["SNr001", "SNr001"],
            "Phase": ["3m", "3m"],
            "Protocol": ["STN+SNr", "STN+SNr"],
            "Contact": [0, 0],
            "Frequency": [30, 45],
        }
    )

    with pytest.raises(ValueError, match="Duplicate follow-up stimulation rows"):
        active_contact_tables.build_scale_contact_only(
            active_contacts,
            contact_parameters,
        )
