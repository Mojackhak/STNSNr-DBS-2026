"""Warping subpackage.

This subpackage contains utilities to map time-aligned tensors (TFR or
connectivity) onto a normalized axis, typically for averaging across events.

Warpers provided:
  - gait_warper  : event-anchored, piecewise-linear time normalization (0..100%)
  - linear_warper: generic event-anchored, piecewise-linear time normalization
  - pad_warper   : crop+concat around annotations, with optional resampling
  - concat_warper: concatenate multiple allowed annotation intervals, then resample
  - stack_warper : keep each matched annotation as one epoch, then resample+stack
"""

from .event_anchored_linear import GaitEpoch, LinearEpoch, gait_warper, linear_warper
from .pad_concat import PadEpoch, pad_warper
from .concat import ConcatEpoch, concat_warper
from .stack import StackEpoch, stack_warper

from .metadata import build_warped_tensor_metadata

__all__ = [
    "GaitEpoch",
    "LinearEpoch",
    "PadEpoch",
    "ConcatEpoch",
    "StackEpoch",
    "gait_warper",
    "linear_warper",
    "pad_warper",
    "concat_warper",
    "stack_warper",
    "build_warped_tensor_metadata",
]
