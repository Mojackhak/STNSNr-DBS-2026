from __future__ import annotations

import os

from datetime import datetime
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from programming_contacts import PROGRAM_PERIOD_3M, get_active_contacts


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
SUMMARY_DIR = PROJECT / "summary"
SCALE_DIR = SUMMARY_DIR / "stats" / "clinic" / "coords" / "scale"
LEAD_DIR = SUMMARY_DIR / "cohort" / "lead"
SUBJ_DIR = SUMMARY_DIR / "cohort" / "subj"
CONTACT_DATASET_DIR = LEAD_DIR / "contact_activation_dataset"

PROGRAMMING_PATH = SUBJ_DIR / "programming.json"
LEAD_COORDS_PATH = LEAD_DIR / "contact_reco_space_locs.pkl"
SUBJECT_COORDS_CONFIG_PATH = LEAD_DIR / "subject_coords_config.json"

SCALE_CONTACT_ONLY_PATH = SCALE_DIR / "scale_contact_only.xlsx"
ACTIVE_CONTACTS_PATH = CONTACT_DATASET_DIR / "active_contacts.csv"
ACTIVE_CONTACTS_INFO_PATH = CONTACT_DATASET_DIR / "active_contacts_info.json"
RANDOM_STIMULATION_PATH = CONTACT_DATASET_DIR / "random_stimulation_parameters.csv"
RANDOM_STIMULATION_INFO_PATH = CONTACT_DATASET_DIR / "random_stimulation_parameters_info.json"
FOLLOWUP_STIMULATION_PATH = SUBJ_DIR / "followup_stimulation.xlsx"
FOLLOWUP_CONTACT_PARAMETERS_SHEET = "Contact Parameters"

RANDOM_SEED = 20260624
ULF_FREQUENCY_THRESHOLD_HZ = 50
VOLTAGE_RANGE = (2.0, 4.0)
PULSE_WIDTH_RANGE = (50, 90)
FREQUENCY_RANGES = {
    "SNr": (15, 40),
    "STN": (120, 180),
}
SCALE_CONTACT_ONLY_COLUMNS = ["Subject", "MNI_x_flip", "MNI_y_flip", "MNI_z_flip"]


def main() -> None:
    CONTACT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SCALE_DIR.mkdir(parents=True, exist_ok=True)

    programming = _load_json(PROGRAMMING_PATH)
    subject_config = _load_json(SUBJECT_COORDS_CONFIG_PATH)
    lead_coords = pd.read_pickle(LEAD_COORDS_PATH)
    followup_contact_parameters = pd.read_excel(
        FOLLOWUP_STIMULATION_PATH,
        sheet_name=FOLLOWUP_CONTACT_PARAMETERS_SHEET,
    )

    active_contacts = build_active_contacts(lead_coords, programming, subject_config)
    scale_contact_only = build_scale_contact_only(
        active_contacts,
        followup_contact_parameters,
    )

    scale_contact_only.to_excel(SCALE_CONTACT_ONLY_PATH, index=False)
    active_contacts.to_csv(ACTIVE_CONTACTS_PATH, index=False)

    random_stimulation = build_random_stimulation_parameters(active_contacts)
    random_stimulation.to_csv(RANDOM_STIMULATION_PATH, index=False)

    _write_json(ACTIVE_CONTACTS_INFO_PATH, build_active_contacts_info(active_contacts))
    _write_json(
        RANDOM_STIMULATION_INFO_PATH,
        build_random_stimulation_info(active_contacts, random_stimulation),
    )

    print(f"Saved {SCALE_CONTACT_ONLY_PATH}")
    print(f"Saved {ACTIVE_CONTACTS_PATH}")
    print(f"Saved {RANDOM_STIMULATION_PATH}")


def build_active_contacts(
    lead_coords: pd.DataFrame,
    programming: dict[str, object],
    subject_config: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subjects = set(programming.keys())

    for _, row in lead_coords.iterrows():
        subject_id = str(row["ID"])
        if subject_id not in subjects:
            continue
        contact = int(row["Contact"])
        active_stn_snr = get_active_contacts(
            programming,
            subject_id,
            "STN+SNr",
            period=PROGRAM_PERIOD_3M,
        )
        if contact not in active_stn_snr:
            continue

        active_snr = get_active_contacts(
            programming,
            subject_id,
            "SNr",
            period=PROGRAM_PERIOD_3M,
        )
        region_programming = "SNr" if contact in active_snr else "STN"
        subject_key = _normalize_subject_key(row["Subject"])
        side_code = _side_code(row["Side"])
        lead_metadata = _lead_metadata(subject_config, subject_key, side_code)

        output_row = row.to_dict()
        output_row["subject_key"] = subject_key
        output_row["region_programming"] = region_programming
        output_row["lead_model"] = lead_metadata["lead"]
        output_row["lead_dist_mm"] = json.dumps(lead_metadata["dist"])
        output_row["lead_offset_mm"] = json.dumps(lead_metadata["offset"])
        output_row["units"] = subject_config.get("units", "mm")
        output_row["active_scale_row"] = len(rows)
        output_row["match_distance_mm"] = 0.0
        rows.append(output_row)

    if not rows:
        raise ValueError("No active contacts were selected from programming.json.")

    table = pd.DataFrame(rows)
    return table.loc[:, _active_contact_columns(table)]


def build_scale_contact_only(
    active_contacts: pd.DataFrame,
    followup_contact_parameters: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = [
        column
        for column in SCALE_CONTACT_ONLY_COLUMNS
        if column not in active_contacts.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing active contact columns: {missing_columns}")

    scale_contact_only = active_contacts.loc[:, SCALE_CONTACT_ONLY_COLUMNS].copy()
    scale_contact_only["ULF"] = derive_ulf_flags(
        active_contacts,
        followup_contact_parameters,
    ).to_numpy()
    return scale_contact_only


def derive_ulf_flags(
    active_contacts: pd.DataFrame,
    followup_contact_parameters: pd.DataFrame,
) -> pd.Series:
    required_active_columns = ["ID", "Contact"]
    missing_active = [
        column
        for column in required_active_columns
        if column not in active_contacts.columns
    ]
    if missing_active:
        raise ValueError(f"Missing active contact columns: {missing_active}")

    required_followup_columns = ["ID", "Phase", "Protocol", "Contact", "Frequency"]
    missing_followup = [
        column
        for column in required_followup_columns
        if column not in followup_contact_parameters.columns
    ]
    if missing_followup:
        raise ValueError(f"Missing follow-up stimulation columns: {missing_followup}")

    followup = followup_contact_parameters.loc[
        followup_contact_parameters["Phase"].astype(str).eq(PROGRAM_PERIOD_3M)
        & followup_contact_parameters["Protocol"].astype(str).eq("STN+SNr"),
        ["ID", "Contact", "Frequency"],
    ].copy()
    followup["ID"] = followup["ID"].astype(str)
    followup["Contact"] = _contact_series_to_int(
        followup["Contact"],
        "follow-up stimulation",
    )
    followup["Frequency"] = pd.to_numeric(followup["Frequency"], errors="coerce")

    duplicate_mask = followup.duplicated(["ID", "Contact"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = _format_contact_keys(
            followup.loc[duplicate_mask, ["ID", "Contact"]]
        )
        raise ValueError(
            "Duplicate follow-up stimulation rows for ID/Contact: "
            f"{duplicate_keys}"
        )

    active_keys = active_contacts.loc[:, ["ID", "Contact"]].copy()
    active_keys["ID"] = active_keys["ID"].astype(str)
    active_keys["Contact"] = _contact_series_to_int(
        active_keys["Contact"],
        "active contacts",
    )
    active_keys["active_row"] = np.arange(active_keys.shape[0])

    merged = active_keys.merge(
        followup,
        on=["ID", "Contact"],
        how="left",
        validate="many_to_one",
    ).sort_values("active_row")
    missing_mask = merged["Frequency"].isna()
    if missing_mask.any():
        missing_keys = _format_contact_keys(
            merged.loc[missing_mask, ["ID", "Contact"]]
        )
        raise ValueError(
            "Missing follow-up stimulation rows or frequencies for ID/Contact: "
            f"{missing_keys}"
        )

    return (
        (merged["Frequency"] < ULF_FREQUENCY_THRESHOLD_HZ)
        .astype(bool)
        .reset_index(drop=True)
    )


def build_random_stimulation_parameters(active_contacts: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []

    for active_contacts_row, row in active_contacts.reset_index(drop=True).iterrows():
        frequency_group = str(row["region_programming"])
        if frequency_group not in FREQUENCY_RANGES:
            raise ValueError(f"Unsupported frequency group: {frequency_group}")
        frequency_low, frequency_high = FREQUENCY_RANGES[frequency_group]

        side_code = _side_code(row["Side"])
        rows.append(
            {
                "ID": row["ID"],
                "Subject": row["Subject"],
                "subject_key": row["subject_key"],
                "Contact": int(row["Contact"]),
                "lead_contact": int(row["Contact"]) + 1,
                "Region": row["Region"],
                "region_programming": row["region_programming"],
                "Side": row["Side"],
                "side_code": side_code,
                "Hemi": row["Hemi"],
                "Lead_idx": int(row["Lead_idx"]),
                "Lead_num": int(row["Lead_num"]),
                "MNI_x_flip": float(row["MNI_x_flip"]),
                "MNI_y_flip": float(row["MNI_y_flip"]),
                "MNI_z_flip": float(row["MNI_z_flip"]),
                "frequency_group": frequency_group,
                "frequency_rule": "region_programming",
                "voltage_V": round(float(rng.uniform(*VOLTAGE_RANGE)), 1),
                "pulse_width_us": int(rng.integers(PULSE_WIDTH_RANGE[0], PULSE_WIDTH_RANGE[1] + 1)),
                "frequency_Hz": int(rng.integers(frequency_low, frequency_high + 1)),
                "unit": "V",
                "cathode": True,
                "anode": "case",
                "random_seed": RANDOM_SEED,
                "active_contacts_row": int(active_contacts_row),
            }
        )

    return pd.DataFrame(rows)


def build_active_contacts_info(active_contacts: pd.DataFrame) -> dict[str, object]:
    return {
        "name": "STN/SNr active contact coordinate table",
        "generated_at": _timestamp(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "inputs": {
            "scale_contact_only_xlsx": _file_info(SCALE_CONTACT_ONLY_PATH),
            "contact_reco_space_locs_pkl": _file_info(LEAD_COORDS_PATH),
            "subject_coords_config_json": _file_info(SUBJECT_COORDS_CONFIG_PATH),
            "programming_json": _file_info(PROGRAMMING_PATH),
        },
        "outputs": {
            "active_contacts_csv": str(ACTIVE_CONTACTS_PATH),
            "active_contacts_info_json": str(ACTIVE_CONTACTS_INFO_PATH),
        },
        "qc": {
            "active_contacts": int(active_contacts.shape[0]),
            "subjects": int(active_contacts["ID"].nunique()),
            "region_counts": _value_counts(active_contacts["Region"]),
            "region_programming_counts": _value_counts(active_contacts["region_programming"]),
            "period": PROGRAM_PERIOD_3M,
            "program_key": "STN+SNr",
        },
        "schema": {
            "ID": "Clinical/reconstruction subject ID.",
            "Subject": "Subject name from contact reconstruction.",
            "subject_key": "Normalized subject identifier used for configuration joins.",
            "Contact": "Source zero-based contact index.",
            "Region": "Anatomical atlas/reconstruction label.",
            "region_programming": "Programming region; SNr if the contact is in the 3m SNr list, otherwise STN.",
            "MNI_x_flip/MNI_y_flip/MNI_z_flip": "Right-flipped MNI coordinates in millimeters.",
            "active_scale_row": "Zero-based row index in scale_contact_only.xlsx.",
            "match_distance_mm": "Exact reconstruction match distance in millimeters.",
        },
        "notes": [
            "This dataset includes contacts in programming.json[ID]['3m']['STN+SNr'].",
            "Contacts in the 3m SNr list are labeled SNr; remaining STN+SNr contacts are labeled STN.",
        ],
    }


def build_random_stimulation_info(
    active_contacts: pd.DataFrame,
    random_stimulation: pd.DataFrame,
) -> dict[str, object]:
    return {
        "name": "STN/SNr random test stimulation parameter table",
        "generated_at": _timestamp(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "random_seed": RANDOM_SEED,
        "inputs": {
            "active_contacts_csv": _file_info(ACTIVE_CONTACTS_PATH),
        },
        "outputs": {
            "random_stimulation_parameters_csv": str(RANDOM_STIMULATION_PATH),
            "random_stimulation_parameters_info_json": str(RANDOM_STIMULATION_INFO_PATH),
        },
        "randomization": {
            "voltage_V": {
                "distribution": "uniform",
                "minimum": VOLTAGE_RANGE[0],
                "maximum": VOLTAGE_RANGE[1],
                "round_decimals": 1,
            },
            "pulse_width_us": {
                "distribution": "discrete_uniform_integer",
                "minimum": PULSE_WIDTH_RANGE[0],
                "maximum": PULSE_WIDTH_RANGE[1],
            },
            "frequency_Hz": {
                "SNr": {
                    "distribution": "discrete_uniform_integer",
                    "minimum": FREQUENCY_RANGES["SNr"][0],
                    "maximum": FREQUENCY_RANGES["SNr"][1],
                },
                "STN": {
                    "distribution": "discrete_uniform_integer",
                    "minimum": FREQUENCY_RANGES["STN"][0],
                    "maximum": FREQUENCY_RANGES["STN"][1],
                },
            },
            "frequency_group_rule": "Frequency is assigned from active_contacts.csv region_programming.",
        },
        "qc": {
            "active_contacts_rows": int(active_contacts.shape[0]),
            "stimulation_rows": int(random_stimulation.shape[0]),
            "subjects": int(random_stimulation["ID"].nunique()),
            "region_counts": _value_counts(random_stimulation["Region"]),
            "region_programming_counts": _value_counts(random_stimulation["region_programming"]),
            "frequency_group_counts": _value_counts(random_stimulation["frequency_group"]),
            "frequency_rule_counts": _value_counts(random_stimulation["frequency_rule"]),
            "voltage_range_V": [
                float(random_stimulation["voltage_V"].min()),
                float(random_stimulation["voltage_V"].max()),
            ],
            "pulse_width_range_us": [
                int(random_stimulation["pulse_width_us"].min()),
                int(random_stimulation["pulse_width_us"].max()),
            ],
            "frequency_range_Hz": [
                int(random_stimulation["frequency_Hz"].min()),
                int(random_stimulation["frequency_Hz"].max()),
            ],
        },
        "schema": {
            "ID": "Clinical/reconstruction subject ID from active_contacts.csv.",
            "Subject": "Subject name from active_contacts.csv.",
            "subject_key": "Normalized subject identifier used for joins.",
            "Contact": "Source zero-based contact index from active_contacts.csv.",
            "lead_contact": "One-based contact index for Lead-DBS stimulation helpers.",
            "Region": "Anatomical atlas/reconstruction label from active_contacts.csv.",
            "region_programming": "Programming region from active_contacts.csv.",
            "frequency_group": "Target group used to choose the frequency range.",
            "frequency_rule": "Rule used to assign frequency_group.",
            "voltage_V": "Random test voltage in volts.",
            "pulse_width_us": "Random test pulse width in microseconds.",
            "frequency_Hz": "Random test frequency in hertz.",
            "active_contacts_row": "Zero-based row index in active_contacts.csv.",
        },
        "notes": [
            "This table is for testing only and is not a clinical programming record.",
            "lead_contact is one-based and can be passed to helpers that require Lead-DBS contact numbering.",
            "Contact preserves the source zero-based contact index from active_contacts.csv.",
        ],
    }


def _active_contact_columns(table: pd.DataFrame) -> list[str]:
    columns = [
        "ID",
        "Subject",
        "subject_key",
        "Contact",
        "Region",
        "region_programming",
        "Side",
        "Hemi",
        "Lead_idx",
        "Lead_num",
        "MNI_x",
        "MNI_y",
        "MNI_z",
        "MNI_x_flip",
        "MNI_y_flip",
        "MNI_z_flip",
        "SNr_in",
        "SNr_x",
        "SNr_y",
        "SNr_z",
        "STN_in",
        "STN_x",
        "STN_y",
        "STN_z",
        "lead_model",
        "lead_dist_mm",
        "lead_offset_mm",
        "units",
        "active_scale_row",
        "match_distance_mm",
    ]
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing active contact columns: {missing}")
    return columns


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _normalize_subject_key(subject: object) -> str:
    text = str(subject)
    if text.startswith("Sub-"):
        return "sub-" + text.removeprefix("Sub-")
    return text


def _side_code(side: object) -> str:
    text = str(side).strip().lower()
    if text.startswith("l"):
        return "L"
    if text.startswith("r"):
        return "R"
    raise ValueError(f"Unsupported side label: {side}")


def _contact_series_to_int(series: pd.Series, context: str) -> pd.Series:
    contacts = pd.to_numeric(series, errors="coerce")
    invalid_mask = contacts.isna() | contacts.mod(1).ne(0)
    if invalid_mask.any():
        invalid_values = series.loc[invalid_mask].astype(str).tolist()
        raise ValueError(f"Invalid {context} contact values: {invalid_values}")
    return contacts.astype("int64")


def _format_contact_keys(keys: pd.DataFrame) -> str:
    unique_keys = keys.drop_duplicates().sort_values(["ID", "Contact"])
    return ", ".join(
        f"{row.ID}|{int(row.Contact)}" for row in unique_keys.itertuples(index=False)
    )


def _lead_metadata(
    subject_config: dict[str, object],
    subject_key: str,
    side_code: str,
) -> dict[str, object]:
    subjects = subject_config.get("subjects", {})
    if not isinstance(subjects, dict):
        raise ValueError("subject_coords_config.json has no subjects mapping.")
    subject_entry = subjects.get(subject_key)
    if not isinstance(subject_entry, dict):
        raise ValueError(f"Missing subject configuration for {subject_key}.")
    side_entry = subject_entry.get(side_code)
    if not isinstance(side_entry, dict):
        raise ValueError(f"Missing {side_code} lead configuration for {subject_key}.")
    return {
        "lead": side_entry.get("lead"),
        "dist": side_entry.get("dist", []),
        "offset": side_entry.get("offset", []),
    }


def _timestamp() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _file_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).replace(microsecond=0).isoformat(),
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


if __name__ == "__main__":
    main()
