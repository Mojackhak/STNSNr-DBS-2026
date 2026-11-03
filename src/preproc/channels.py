"""
Channel mapping and bipolar pair derivation utilities.

Purpose
-------
This module standardizes channel metadata for DBS-LFP/EEG-style pipelines by:
  1) Mapping recording channel names to EEG1005 labels along predefined left/right lines.
  2) Deriving within-hemisphere adjacent bipolar pairs (anode = contact[i], cathode = contact[i+1]).
  3) Optionally dropping non-adjacent numeric pairs (keeps only 'a_(a+1)' patterns).

Why this exists
---------------
Many DBS/LFP recordings encode contacts using device-specific channel names. For reproducible,
research-grade analysis, we often need:
  - a deterministic mapping to canonical EEG/EEG1005 labels (for visualization and reporting),
  - consistent bipolar derivations (common DBS-LFP referencing convention),
  - explicit tracking of which bipolar pairs were removed by adjacency rules.

Assumptions
-----------
- The channel count is even and can be split into two equal halves (left/right).
- By default, the first half is treated as left and the second half as right (configurable via 'left_first').
- Bipolar pairs are created within each side only (no cross-hemisphere pairs).

I/O behavior
------------
This module performs no disk I/O. It operates on an MNE-Python Raw-like object (or channel name arrays)
and returns structured metadata (see DerivedChannels).

Recommended location
--------------------
Place this module under 'preproc/' (e.g., preproc/channels.py or preproc/montage.py). If you have an
I/O standardization step, that step can call into this module but should remain separate.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import re


DEFAULT_LEFT_EEG1005_LINE: list[str] = ["O1", "P3", "C3", "F3"]
DEFAULT_RIGHT_EEG1005_LINE: list[str] = ["O2", "P4", "C4", "F4"]


@dataclass(frozen=True)
class DerivedChannels:
    """Container for derived channel metadata and bipolar pairs."""
    chs: np.ndarray
    ch2eeg: dict[str, str]
    eeg2ch: dict[str, str]
    anodes: np.ndarray
    cathodes: np.ndarray
    pair_names: np.ndarray
    dropped_pair_idx: np.ndarray


def split_channels_into_sides(chs: np.ndarray, *, left_first: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Split channels into left and right halves.

    Args:
        chs: Channel name array in acquisition order.
        left_first: If True, first half is treated as left; else reversed.

    Returns:
        (left_chs, right_chs)

    Raises:
        ValueError: If ch count is odd.
    """
    chs = np.asarray(chs, dtype=str)
    n_ch = int(chs.size)
    if n_ch % 2 != 0:
        raise ValueError(f"Expected an even number of channels, got {n_ch}. Cannot split into left/right halves.")
    n_side = n_ch // 2
    first = chs[:n_side]
    second = chs[n_side:]
    return (first, second) if left_first else (second, first)


def make_ch2eeg_mapping(
    chs: np.ndarray,
    *,
    left_eeg_line: list[str] = DEFAULT_LEFT_EEG1005_LINE,
    right_eeg_line: list[str] = DEFAULT_RIGHT_EEG1005_LINE,
    left_first: bool = True,
) -> dict[str, str]:
    """Create a channel->EEG1005 mapping by assigning channels sequentially along predefined lines.

    The mapping is created by:
      - splitting channels into left and right halves
      - mapping left half to left_eeg_line[:n_side]
      - mapping right half to right_eeg_line[:n_side]

    Args:
        chs: Channel name array.
        left_eeg_line: EEG1005 names along the left line (ordered).
        right_eeg_line: EEG1005 names along the right line (ordered).
        left_first: Whether the first half of chs is left side.

    Returns:
        Dictionary mapping original channel names to EEG names.

    Raises:
        ValueError: If the number of channels per side exceeds the provided EEG name lists.
    """
    left_chs, right_chs = split_channels_into_sides(chs, left_first=left_first)
    n_side = int(left_chs.size)

    if n_side > len(left_eeg_line):
        raise ValueError(
            f"Not enough left_eeg_line names: need {n_side}, got {len(left_eeg_line)}. "
            f"Provide a longer left_eeg_line."
        )
    if n_side > len(right_eeg_line):
        raise ValueError(
            f"Not enough right_eeg_line names: need {n_side}, got {len(right_eeg_line)}. "
            f"Provide a longer right_eeg_line."
        )

    ch2eeg: dict[str, str] = {}
    for ch, eeg_name in zip(left_chs, left_eeg_line[:n_side], strict=True):
        ch2eeg[str(ch)] = str(eeg_name)
    for ch, eeg_name in zip(right_chs, right_eeg_line[:n_side], strict=True):
        ch2eeg[str(ch)] = str(eeg_name)

    return ch2eeg


def default_contact_labeler(ch_name: str) -> str:
    """Extract a contact label from a channel name.

    Strategy:
      1) take the substring before the first underscore
      2) if that substring contains digits, return the leading digits
      3) otherwise return the substring as-is

    Examples:
      '1_L'   -> '1'
      '02_R'  -> '02'
      'A1_L'  -> 'A1' (no leading digits)
    """
    head = ch_name.split("_", 1)[0]
    m = re.match(r"^(\d+)", head)
    return m.group(1) if m else head


def drop_non_adjacent_pair_names(pair_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop elements not matching '{a}_{a+1}' numeric adjacency and return kept + dropped indices.

    Args:
        pair_names: Array of strings like ['1_2', '2_3', '2_4'].

    Returns:
        kept: Elements that match '{a}_{a+1}' with numeric a,b and b == a+1.
        dropped_idx: Indices of dropped elements.
    """
    pair_names = np.asarray(pair_names, dtype=str)
    pattern = re.compile(r"^(\d+)_(\d+)$")

    kept: list[str] = []
    dropped_idx: list[int] = []

    for i, s in enumerate(pair_names.tolist()):
        m = pattern.match(s)
        if m and int(m.group(2)) == int(m.group(1)) + 1:
            kept.append(s)
        else:
            dropped_idx.append(i)

    return np.asarray(kept, dtype=str), np.asarray(dropped_idx, dtype=int)


def build_bipolar_pairs(
    chs: np.ndarray,
    *,
    left_first: bool = True,
    contact_labeler: Callable[[str], str] = default_contact_labeler,
    drop_non_adjacent_pairs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build bipolar adjacent pairs within each side (left and right halves).

    Rules (per side):
      anodes   = side[0 : n_side-1]
      cathodes = side[1 : n_side]

    Pair names are generated as '{label(anode)}_{label(cathode)}'.
    Optionally drop non-adjacent numeric pairs (e.g., '0_2').

    Args:
        chs: Channel names array.
        left_first: Whether first half is left.
        contact_labeler: Function to extract contact label from channel name.
        drop_non_adjacent_pairs: If True, keep only '{a}_{a+1}' numeric adjacency.

    Returns:
        pair_names, anodes, cathodes, dropped_idx
    """
    left_chs, right_chs = split_channels_into_sides(chs, left_first=left_first)
    n_side = int(left_chs.size)
    if n_side < 2:
        raise ValueError(f"Need at least 2 channels per side to form bipolar pairs, got {n_side}.")

    anodes = np.concatenate([left_chs[:-1], right_chs[:-1]]).astype(str)
    cathodes = np.concatenate([left_chs[1:], right_chs[1:]]).astype(str)

    pair_names = np.asarray(
        [f"{contact_labeler(a)}_{contact_labeler(c)}" for a, c in zip(anodes, cathodes, strict=True)],
        dtype=str,
    )

    dropped_idx = np.asarray([], dtype=int)
    if drop_non_adjacent_pairs:
        kept_pair_names, dropped_idx = drop_non_adjacent_pair_names(pair_names)
        if dropped_idx.size > 0:
            anodes = np.delete(anodes, dropped_idx)
            cathodes = np.delete(cathodes, dropped_idx)
        pair_names = kept_pair_names

    return pair_names, anodes, cathodes, dropped_idx


def derive_channels_from_raw(
    raw: Any,
    *,
    left_eeg_line: list[str] = DEFAULT_LEFT_EEG1005_LINE,
    right_eeg_line: list[str] = DEFAULT_RIGHT_EEG1005_LINE,
    left_first: bool = True,
    drop_non_adjacent_pairs: bool = True,
    contact_labeler: Callable[[str], str] = default_contact_labeler,
) -> DerivedChannels:
    """High-level helper to build ch2eeg/eeg2ch plus bipolar pairs from an MNE Raw-like object."""
    chs = np.asarray(raw.info["ch_names"], dtype=str)

    ch2eeg = make_ch2eeg_mapping(
        chs,
        left_eeg_line=left_eeg_line,
        right_eeg_line=right_eeg_line,
        left_first=left_first,
    )
    eeg2ch = {v: k for k, v in ch2eeg.items()}

    pair_names, anodes, cathodes, dropped_idx = build_bipolar_pairs(
        chs,
        left_first=left_first,
        contact_labeler=contact_labeler,
        drop_non_adjacent_pairs=drop_non_adjacent_pairs,
    )

    return DerivedChannels(
        chs=chs,
        ch2eeg=ch2eeg,
        eeg2ch=eeg2ch,
        anodes=anodes,
        cathodes=cathodes,
        pair_names=pair_names,
        dropped_pair_idx=dropped_idx,
    )
