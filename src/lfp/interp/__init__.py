"""Interpolation utilities for LFP tensors."""

from .freq import (
    InterpMethod,
    interpolate_freq_axis,
    interpolate_tensor_with_metadata,
)

__all__ = [
    "InterpMethod",
    "interpolate_freq_axis",
    "interpolate_tensor_with_metadata",
]
