"""Montage and re-referencing helpers for MNE Raw objects."""

from __future__ import annotations

from typing import Dict, Sequence

import mne
import numpy as np


ReferenceDict = Dict[str, Sequence[str | None]]


def reset_ref(
    raw: mne.io.BaseRaw,
    reference: ReferenceDict,
    drop_refs: bool = True,
) -> mne.io.BaseRaw:
    """Reset channel references according to an anode/cathode mapping.

    Parameters
    ----------
    raw
        Input raw object. This function does not modify it in place.
    reference
        Dictionary with keys ``"anode"``, ``"cathode"``, and ``"ch_name"``.
        Entries are matched by index:
        - ``(a, b, c)``   -> new channel ``c = a - b``
        - ``(None, b, c)`` -> new channel ``c = -b``
        - ``(a, None, c)`` -> new channel ``c = a``
        ``a`` and ``b`` cannot both be ``None``.
    drop_refs
        If ``True``, drop original reference channels (all channels used as anode/cathode).

    Returns
    -------
    mne.io.BaseRaw
        A new raw object containing the re-referenced channels.
    """
    required_keys = ("anode", "cathode", "ch_name")
    # Keep a defensive copy and restore it at the end so events/descriptions
    # are preserved regardless of internal channel operations.
    annotations_in = raw.annotations.copy()
    missing = [k for k in required_keys if k not in reference]
    if missing:
        raise KeyError(f"Missing required keys in `reference`: {missing}")

    anodes = list(reference["anode"])
    cathodes = list(reference["cathode"])
    new_names = list(reference["ch_name"])

    n = len(anodes)
    if len(cathodes) != n or len(new_names) != n:
        raise ValueError(
            "reference['anode'], reference['cathode'], and reference['ch_name'] "
            f"must have the same length, got {len(anodes)}, {len(cathodes)}, {len(new_names)}."
        )

    if n == 0:
        return raw.copy()

    for i, (a, b, c) in enumerate(zip(anodes, cathodes, new_names, strict=True)):
        if a is None and b is None:
            raise ValueError(f"Invalid reference row {i}: anode and cathode cannot both be None.")
        if c is None or not isinstance(c, str) or c == "":
            raise ValueError(f"Invalid reference row {i}: ch_name must be a non-empty string.")
        if a is not None and not isinstance(a, str):
            raise TypeError(f"Invalid reference row {i}: anode must be str or None, got {type(a)!r}.")
        if b is not None and not isinstance(b, str):
            raise TypeError(f"Invalid reference row {i}: cathode must be str or None, got {type(b)!r}.")

    ch_set = set(raw.ch_names)
    for i, (a, b) in enumerate(zip(anodes, cathodes, strict=True)):
        if a is not None and a not in ch_set:
            raise ValueError(f"Invalid reference row {i}: anode channel {a!r} not found in raw.")
        if b is not None and b not in ch_set:
            raise ValueError(f"Invalid reference row {i}: cathode channel {b!r} not found in raw.")

    if len(set(new_names)) != len(new_names):
        raise ValueError("`reference['ch_name']` contains duplicates. New channel names must be unique.")

    refs_to_drop = {x for x in (*anodes, *cathodes) if x is not None}
    existing = set(raw.ch_names)
    temp_to_target: dict[str, str] = {}
    add_names: list[str] = []
    reserved = existing | set(new_names)

    for i, target in enumerate(new_names):
        if target in existing:
            if not (drop_refs and target in refs_to_drop):
                raise ValueError(
                    f"New channel name {target!r} already exists in raw. "
                    "Rename it in `reference['ch_name']` or set `drop_refs=True` and reference that channel."
                )
            tmp = f"__reset_ref_tmp_{i}"
            while tmp in reserved:
                tmp += "_x"
            temp_to_target[tmp] = target
            add_names.append(tmp)
            reserved.add(tmp)
        else:
            add_names.append(target)

    data = raw.get_data()
    name_to_idx = {name: idx for idx, name in enumerate(raw.ch_names)}
    type_map = {name: ch_t for name, ch_t in zip(raw.ch_names, raw.get_channel_types(), strict=True)}

    new_data = np.empty((n, data.shape[1]), dtype=np.float64)
    new_types: list[str] = []

    for i, (a, b) in enumerate(zip(anodes, cathodes, strict=True)):
        if a is None:
            new_data[i, :] = -data[name_to_idx[b], :]
            source_name = b
        elif b is None:
            new_data[i, :] = data[name_to_idx[a], :]
            source_name = a
        else:
            new_data[i, :] = data[name_to_idx[a], :] - data[name_to_idx[b], :]
            source_name = a
        new_types.append(type_map[source_name])

    raw_out = raw.copy()
    if not raw_out.preload:
        raw_out.load_data()
    info_new = mne.create_info(
        ch_names=add_names,
        sfreq=float(raw.info["sfreq"]),
        ch_types=new_types,
    )
    raw_new = mne.io.RawArray(new_data, info_new, verbose=False)
    raw_out.add_channels([raw_new], force_update_info=True)

    if drop_refs:
        drop_list = [ch for ch in raw.ch_names if ch in refs_to_drop]
        if drop_list:
            raw_out.drop_channels(drop_list)

    if temp_to_target:
        raw_out.rename_channels(temp_to_target)

    # Explicitly restore input annotations/events/descriptions.
    raw_out.set_annotations(annotations_in)

    return raw_out
