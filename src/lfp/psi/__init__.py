"""Phase Slope Index (PSI) computation.

PSI is a directed connectivity measure based on the slope of the phase
relationship across frequencies.

The main entry point is :func:`grid`, which computes a time-resolved PSI tensor
aligned to a decimated time grid (compatible with the TFR/connectivity pipelines).
"""

from .grid import grid

__all__ = [
    "grid",
]
