from __future__ import annotations

import os

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import shutil

import numpy as np
import pandas as pd


# Confirmed column labels in the recovered backup clinical workbook.
BACKUP_SCALE_COLUMN_NAMES = {
    "EducationYears": "Education years",
    "DiseaseDuration": "Disease duration",
    "LCTImprovementRate": "LCT improvement rate",
    "AS": "MAES",
    "Cleveland": "CCCS",
    "Schwab-England": "SE-ADL",
    "UPDRS III-OFF": "MDS-UPDRS III (Med-OFF)",
    "UPDRS III-ON": "MDS-UPDRS III (Med-ON)",
    "Axial-OFF": "MDS-UPDRS III axial (Med-OFF)",
    "Axial-ON": "MDS-UPDRS III axial (Med-ON)",
    "Rigidity-OFF": "MDS-UPDRS III rigidity (Med-OFF)",
    "Rigidity-ON": "MDS-UPDRS III rigidity (Med-ON)",
    "Bradykinesia-OFF": "MDS-UPDRS III bradykinesia (Med-OFF)",
    "Bradykinesia-ON": "MDS-UPDRS III bradykinesia (Med-ON)",
    "RestTremor-OFF": "MDS-UPDRS III rest tremor (Med-OFF)",
    "RestTremor-ON": "MDS-UPDRS III rest tremor (Med-ON)",
    "ActionTremor-OFF": "MDS-UPDRS III action tremor (Med-OFF)",
    "ActionTremor-ON": "MDS-UPDRS III action tremor (Med-ON)"
}


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
SUBJECT_DIR = PROJECT / "summary" / "cohort" / "subj"
ORIGIN_PATH = SUBJECT_DIR / "subject_effect_origin.xlsx"
EFFECT_PATH = SUBJECT_DIR / "subj_effect.xlsx"
DELTA_EFFECT_PATH = SUBJECT_DIR / "subj_delta_effect.xlsx"

ID_COL = "ID"
METADATA_COLUMNS = [ID_COL, "NameEn", "NameZh", "LCT Δ%"]
PROTOCOL_STN = "STN"
PROTOCOL_STN_SNR = "STN+SNr"
PHASE_IMMEDIATE = "immediate"
PHASE_3M = "3m"
HIGHER_IS_BETTER_SCALES = {"SE-ADL score (%)"}

SCALE_LABELS: OrderedDict[str, str] = OrderedDict(
    [
        ("MDS-UPDRS III score", "UPDRS-III"),
        ("MDS-UPDRS III axial score", "UPDRS-III axial"),
        ("FOGQ score", "FOG-Q"),
        ("PDQ39 score", "PDQ-39"),
        ("KPPS score", "KPPS"),
        ("MADRS score", "MADRS"),
        ("ADL", "ADL"),
        ("SE-ADL score (%)", "SE-ADL"),
        ("BDI score", "BDI"),
        ("Bradykinesia subscore (Med OFF)", "Bradykinesia subscore (Med OFF)"),
        ("CCCS", "CCCS"),
        ("DSFS", "DSFS"),
        ("EAT-10", "EAT-10"),
        ("ESS", "ESS"),
        ("FSS", "FSS"),
        ("HAMA", "HAMA"),
        ("HAMD", "HAMD"),
        ("MAES", "MAES"),
        ("MDS-UDPRS II", "MDS-UDPRS II"),
        ("MDS-UPDRS I", "MDS-UPDRS I"),
        ("MDS-UPDRS IV", "MDS-UPDRS IV"),
        ("ODQ", "ODQ"),
        ("RBDSQ", "RBDSQ"),
        ("RBDSQ-HK", "RBDSQ-HK"),
        ("Rigidity subscore (Med OFF)", "Rigidity subscore (Med OFF)"),
        ("SCOPA-AUT", "SCOPA-AUT"),
        ("SDQ", "SDQ"),
        ("Tremor subscore (Med OFF)", "Tremor subscore (Med OFF)"),
    ]
)

LEGACY_EFFECT_COLUMNS = [
    ID_COL,
    "NameEn",
    "NameZh",
    "LCT Δ%",
    "UPDRS-III Δ% (STN, immed)",
    "UPDRS-III Δ% (STN, 3m)",
    "UPDRS-III Δ% (STN+SNr, immed)",
    "UPDRS-III Δ% (STN+SNr, 3m)",
    "UPDRS-III axial Δ% (STN, immed)",
    "UPDRS-III axial Δ% (STN, 3m)",
    "UPDRS-III axial Δ% (STN+SNr, immed)",
    "UPDRS-III axial Δ% (STN+SNr, 3m)",
    "FOG-Q Δ% (STN, 3m)",
    "FOG-Q Δ% (STN+SNr, 3m)",
    "PDQ-39 Δ% (STN, 3m)",
    "PDQ-39 Δ% (STN+SNr, 3m)",
    "KPPS Δ% (STN, 3m)",
    "KPPS Δ% (STN+SNr, 3m)",
    "MADRS Δ% (STN, 3m)",
    "MADRS Δ% (STN+SNr, 3m)",
    "ADL Δ% (STN, 3m)",
    "ADL Δ% (STN+SNr, 3m)",
    "SE-ADL Δ% (STN, 3m)",
    "SE-ADL Δ% (STN+SNr, 3m)",
]

LEGACY_DELTA_EFFECT_COLUMNS = [
    ID_COL,
    "UPDRS-III Δ% (STN, immed)",
    "ΔUPDRS-III Δ% (+SNr, immed)",
    "UPDRS-III Δ% (STN, 3m)",
    "ΔUPDRS-III Δ% (+SNr, 3m)",
    "UPDRS-III axial Δ% (STN, immed)",
    "ΔUPDRS-III axial Δ% (+SNr, immed)",
    "UPDRS-III axial Δ% (STN, 3m)",
    "ΔUPDRS-III axial Δ% (+SNr, 3m)",
    "FOG-Q Δ% (STN, 3m)",
    "ΔFOG-Q Δ% (+SNr, 3m)",
    "PDQ-39 Δ% (STN, 3m)",
    "ΔPDQ-39 Δ% (+SNr, 3m)",
    "KPPS Δ% (STN, 3m)",
    "ΔKPPS Δ% (+SNr, 3m)",
    "MADRS Δ% (STN, 3m)",
    "ΔMADRS Δ% (+SNr, 3m)",
    "ADL Δ% (STN, 3m)",
    "ΔADL Δ% (+SNr, 3m)",
    "SE-ADL Δ% (STN, 3m)",
    "ΔSE-ADL Δ% (+SNr, 3m)",
]


@dataclass(frozen=True)
class TotalEffectSpec:
    column: str
    scale: str
    protocol: str
    phase: str


@dataclass(frozen=True)
class DeltaEffectSpec:
    column: str
    scale: str
    phase: str


def expected_effect_columns(*, include_metadata: bool = True) -> list[str]:
    columns = list(LEGACY_EFFECT_COLUMNS if include_metadata else LEGACY_EFFECT_COLUMNS[4:])
    seen = set(columns)
    for scale in SCALE_LABELS:
        for protocol in (PROTOCOL_STN, PROTOCOL_STN_SNR):
            column = total_effect_column(scale, protocol, PHASE_3M)
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def expected_delta_effect_columns() -> list[str]:
    columns = list(LEGACY_DELTA_EFFECT_COLUMNS)
    seen = set(columns)
    for scale in SCALE_LABELS:
        for column in (
            total_effect_column(scale, PROTOCOL_STN, PHASE_3M),
            delta_effect_column(scale, PHASE_3M),
        ):
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def build_effect_tables(
    origin: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = _effect_matrix(origin)
    subject_order = _subject_order(origin, metadata)

    effect_table = _metadata_frame(subject_order, metadata)
    for spec in _total_effect_specs():
        effect_table[spec.column] = [
            _lookup_total(effects, subject_id, spec.scale, spec.protocol, spec.phase)
            for subject_id in subject_order
        ]
    effect_table = effect_table.loc[:, expected_effect_columns(include_metadata=True)]

    delta_table = pd.DataFrame({ID_COL: subject_order})
    for spec_or_column in _delta_specs_in_output_order():
        if isinstance(spec_or_column, TotalEffectSpec):
            delta_table[spec_or_column.column] = [
                _lookup_total(
                    effects,
                    subject_id,
                    spec_or_column.scale,
                    spec_or_column.protocol,
                    spec_or_column.phase,
                )
                for subject_id in subject_order
            ]
        else:
            delta_table[spec_or_column.column] = [
                _lookup_delta(effects, subject_id, spec_or_column.scale, spec_or_column.phase)
                for subject_id in subject_order
            ]
    delta_table = delta_table.loc[:, expected_delta_effect_columns()]

    return effect_table, delta_table


def assert_shared_columns_equal(
    old_table: pd.DataFrame,
    refreshed_table: pd.DataFrame,
    *,
    atol: float = 1e-6,
) -> None:
    if ID_COL not in old_table.columns or ID_COL not in refreshed_table.columns:
        raise AssertionError("Both tables must contain an ID column.")

    shared_columns = [column for column in old_table.columns if column in refreshed_table.columns]
    missing_columns = [column for column in old_table.columns if column not in refreshed_table.columns]
    if missing_columns:
        raise AssertionError(f"Refreshed table is missing old columns: {missing_columns}")

    old_indexed = old_table.set_index(ID_COL, drop=False)
    refreshed_indexed = refreshed_table.set_index(ID_COL, drop=False)
    missing_ids = [subject_id for subject_id in old_indexed.index if subject_id not in refreshed_indexed.index]
    if missing_ids:
        raise AssertionError(f"Refreshed table is missing old IDs: {missing_ids}")

    mismatches: list[str] = []
    for subject_id in old_indexed.index:
        for column in shared_columns:
            old_value = old_indexed.at[subject_id, column]
            new_value = refreshed_indexed.at[subject_id, column]
            if not _values_equal(old_value, new_value, atol=atol):
                mismatches.append(
                    f"{subject_id} {column}: old={old_value!r}, refreshed={new_value!r}"
                )
                if len(mismatches) >= 10:
                    raise AssertionError("Shared column mismatch: " + "; ".join(mismatches))
    if mismatches:
        raise AssertionError("Shared column mismatch: " + "; ".join(mismatches))


def refresh_workbooks(
    *,
    origin_path: Path = ORIGIN_PATH,
    effect_path: Path = EFFECT_PATH,
    delta_effect_path: Path = DELTA_EFFECT_PATH,
    backup_to_trash: bool = True,
) -> dict[str, object]:
    origin = pd.read_excel(origin_path)
    old_effect = pd.read_excel(effect_path)
    old_delta = pd.read_excel(delta_effect_path)

    effect_table, delta_table = build_effect_tables(
        origin,
        old_effect.loc[:, [column for column in METADATA_COLUMNS if column in old_effect.columns]],
    )

    assert_shared_columns_equal(old_effect, effect_table)
    assert_shared_columns_equal(old_delta, delta_table)

    backups: list[Path] = []
    if backup_to_trash:
        backups.append(_move_to_trash(effect_path))
        backups.append(_move_to_trash(delta_effect_path))

    effect_table.to_excel(effect_path, index=False)
    delta_table.to_excel(delta_effect_path, index=False)

    return {
        "effect_path": effect_path,
        "delta_effect_path": delta_effect_path,
        "effect_shape": effect_table.shape,
        "delta_effect_shape": delta_table.shape,
        "backups": backups,
    }


def total_effect_column(scale: str, protocol: str, phase: str) -> str:
    return f"{_scale_label(scale)} Δ% ({protocol}, {_phase_label(phase)})"


def delta_effect_column(scale: str, phase: str) -> str:
    return f"Δ{_scale_label(scale)} Δ% (+SNr, {_phase_label(phase)})"


def _total_effect_specs() -> list[TotalEffectSpec]:
    specs: list[TotalEffectSpec] = []
    seen: set[str] = set()

    for column in LEGACY_EFFECT_COLUMNS[4:]:
        spec = _total_spec_for_legacy_column(column)
        specs.append(spec)
        seen.add(spec.column)

    for scale in SCALE_LABELS:
        for protocol in (PROTOCOL_STN, PROTOCOL_STN_SNR):
            spec = TotalEffectSpec(
                column=total_effect_column(scale, protocol, PHASE_3M),
                scale=scale,
                protocol=protocol,
                phase=PHASE_3M,
            )
            if spec.column not in seen:
                specs.append(spec)
                seen.add(spec.column)

    return specs


def _delta_specs_in_output_order() -> list[TotalEffectSpec | DeltaEffectSpec]:
    specs: list[TotalEffectSpec | DeltaEffectSpec] = []
    seen: set[str] = set()

    for column in LEGACY_DELTA_EFFECT_COLUMNS[1:]:
        spec = _delta_spec_for_legacy_column(column)
        specs.append(spec)
        seen.add(column)

    for scale in SCALE_LABELS:
        total_spec = TotalEffectSpec(
            column=total_effect_column(scale, PROTOCOL_STN, PHASE_3M),
            scale=scale,
            protocol=PROTOCOL_STN,
            phase=PHASE_3M,
        )
        delta_spec = DeltaEffectSpec(
            column=delta_effect_column(scale, PHASE_3M),
            scale=scale,
            phase=PHASE_3M,
        )
        for spec in (total_spec, delta_spec):
            if spec.column not in seen:
                specs.append(spec)
                seen.add(spec.column)

    return specs


def _effect_matrix(origin: pd.DataFrame) -> pd.DataFrame:
    required = {ID_COL, "Scale", "Protocol", "Phase", "Value", "Baseline"}
    missing = sorted(required.difference(origin.columns))
    if missing:
        raise ValueError(f"Origin table is missing required columns: {missing}")

    table = origin.loc[:, [ID_COL, "Scale", "Protocol", "Phase", "Value", "Baseline"]].copy()
    unknown_scales = sorted(set(table["Scale"].dropna().astype(str)).difference(SCALE_LABELS))
    if unknown_scales:
        raise ValueError(f"Origin table contains unknown scales: {unknown_scales}")

    table["Effect"] = table.apply(_improvement_rate, axis=1)
    return table.pivot_table(
        index=ID_COL,
        columns=["Scale", "Protocol", "Phase"],
        values="Effect",
        aggfunc="mean",
        dropna=False,
    )


def _improvement_rate(row: pd.Series) -> float:
    baseline = _as_float(row["Baseline"])
    value = _as_float(row["Value"])
    if not math.isfinite(baseline) or baseline == 0 or not math.isfinite(value):
        return np.nan

    scale = str(row["Scale"])
    if scale in HIGHER_IS_BETTER_SCALES:
        return (value - baseline) / baseline * 100.0
    return (baseline - value) / baseline * 100.0


def _lookup_total(
    effects: pd.DataFrame,
    subject_id: str,
    scale: str,
    protocol: str,
    phase: str,
) -> float:
    key = (scale, protocol, phase)
    if subject_id not in effects.index or key not in effects.columns:
        return np.nan
    return effects.at[subject_id, key]


def _lookup_delta(
    effects: pd.DataFrame,
    subject_id: str,
    scale: str,
    phase: str,
) -> float:
    stn = _lookup_total(effects, subject_id, scale, PROTOCOL_STN, phase)
    stn_snr = _lookup_total(effects, subject_id, scale, PROTOCOL_STN_SNR, phase)
    if not math.isfinite(_as_float(stn)) or not math.isfinite(_as_float(stn_snr)):
        return np.nan
    return float(stn_snr) - float(stn)


def _subject_order(origin: pd.DataFrame, metadata: pd.DataFrame | None) -> list[str]:
    if metadata is not None and ID_COL in metadata.columns:
        return metadata[ID_COL].astype(str).tolist()
    return origin[ID_COL].drop_duplicates().astype(str).tolist()


def _metadata_frame(subject_order: list[str], metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame({ID_COL: subject_order})

    frame = pd.DataFrame({ID_COL: subject_order})
    available_metadata_columns = [
        column for column in METADATA_COLUMNS if column != ID_COL and column in metadata.columns
    ]
    if available_metadata_columns:
        metadata_indexed = metadata.drop_duplicates(ID_COL).set_index(ID_COL)
        for column in available_metadata_columns:
            frame[column] = frame[ID_COL].map(metadata_indexed[column])

    for column in METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, METADATA_COLUMNS]


def _total_spec_for_legacy_column(column: str) -> TotalEffectSpec:
    for scale, label in SCALE_LABELS.items():
        for protocol in (PROTOCOL_STN, PROTOCOL_STN_SNR):
            for phase in (PHASE_IMMEDIATE, PHASE_3M):
                if column == f"{label} Δ% ({protocol}, {_phase_label(phase)})":
                    return TotalEffectSpec(column, scale, protocol, phase)
    raise ValueError(f"Cannot map legacy effect column: {column}")


def _delta_spec_for_legacy_column(column: str) -> TotalEffectSpec | DeltaEffectSpec:
    for scale, label in SCALE_LABELS.items():
        for phase in (PHASE_IMMEDIATE, PHASE_3M):
            total_column = f"{label} Δ% ({PROTOCOL_STN}, {_phase_label(phase)})"
            delta_column = f"Δ{label} Δ% (+SNr, {_phase_label(phase)})"
            if column == total_column:
                return TotalEffectSpec(column, scale, PROTOCOL_STN, phase)
            if column == delta_column:
                return DeltaEffectSpec(column, scale, phase)
    raise ValueError(f"Cannot map legacy delta-effect column: {column}")


def _scale_label(scale: str) -> str:
    try:
        return SCALE_LABELS[scale]
    except KeyError as exc:
        raise ValueError(f"Unknown scale: {scale}") from exc


def _phase_label(phase: str) -> str:
    if phase == PHASE_IMMEDIATE:
        return "immed"
    if phase == PHASE_3M:
        return "3m"
    raise ValueError(f"Unsupported phase: {phase}")


def _values_equal(old_value: object, new_value: object, *, atol: float) -> bool:
    if pd.isna(old_value) and pd.isna(new_value):
        return True

    old_float = _as_float(old_value)
    new_float = _as_float(new_value)
    if math.isfinite(old_float) and math.isfinite(new_float):
        return math.isclose(old_float, new_float, abs_tol=atol, rel_tol=0.0)

    return str(old_value) == str(new_value)


def _as_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _move_to_trash(path: Path) -> Path:
    trash_dir = Path.home() / ".Trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = trash_dir / f"{path.stem}_before_28_scale_update_{timestamp}{path.suffix}"
    counter = 1
    while target.exists():
        target = trash_dir / f"{path.stem}_before_28_scale_update_{timestamp}_{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), str(target))
    return target


def main() -> None:
    result = refresh_workbooks()
    print(f"Saved {result['effect_path']} with shape {result['effect_shape']}")
    print(f"Saved {result['delta_effect_path']} with shape {result['delta_effect_shape']}")
    for backup in result["backups"]:
        print(f"Moved previous workbook to Trash: {backup}")


if __name__ == "__main__":
    main()
