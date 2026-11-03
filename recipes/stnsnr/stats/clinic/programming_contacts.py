from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PROGRAM_PERIOD_3M = "3m"
PROGRAM_PERIOD_IMMEDIATE = "immediate"


def infer_program_period(label: object, default: str = PROGRAM_PERIOD_3M) -> str:
    """Infer the programming period from a scale or phase label."""
    text = str(label).lower()
    if "immed" in text or "immediate" in text:
        return PROGRAM_PERIOD_IMMEDIATE
    if "3m" in text or "3 m" in text:
        return PROGRAM_PERIOD_3M
    return default


def get_active_contacts(
    program_dict: Mapping[str, Any],
    subject_id: object,
    program_type: str,
    *,
    label: object | None = None,
    period: str | None = None,
) -> set[int]:
    """Return active contacts for both legacy and period-nested programming JSON."""
    subject_key = str(subject_id)
    subject_program = program_dict.get(subject_key)
    if not isinstance(subject_program, Mapping):
        return set()

    if program_type in subject_program:
        return _normalize_contacts(subject_program[program_type])

    selected_period = period or infer_program_period(label)
    period_program = subject_program.get(selected_period)
    if not isinstance(period_program, Mapping):
        return set()

    return _normalize_contacts(period_program.get(program_type, []))


def _normalize_contacts(value: object) -> set[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    contacts: set[int] = set()
    for item in value:
        try:
            contacts.add(int(item))
        except (TypeError, ValueError):
            continue
    return contacts
