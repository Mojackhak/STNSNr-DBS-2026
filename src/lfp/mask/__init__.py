"""Annotation-based masking utilities.

This subpackage contains helpers that turn MNE Raw annotations into time-axis masks.
The masks are typically applied to time-aligned tensors (TFR, connectivity) **before**
warping them onto a normalized axis.
"""

from .annotations import (  # noqa: F401
    drop_raw_annotations,
    filter_raw_annotations,
    mask_by_annotations,
    time_mask_by_annotations,
)

__all__ = [
    "drop_raw_annotations",
    "filter_raw_annotations",
    "mask_by_annotations",
    "time_mask_by_annotations",
]
