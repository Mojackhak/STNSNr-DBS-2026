"""Connectivity subpackage.

This subpackage provides time-frequency connectivity computations that are
aligned to the same time grid as TFR outputs.

Public API:
  - :func:`grid` : compute time-frequency connectivity on a (freq x time) grid
"""

from .grid import grid

__all__ = [
    "grid",
]
