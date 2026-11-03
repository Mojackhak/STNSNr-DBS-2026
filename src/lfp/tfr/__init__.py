"""TFR subpackage.

This subpackage contains:
  - :func:`grid`   : compute time-frequency power on a frequency grid
  - :func:`decompose`  : (optional) SpecParam-based decomposition of a TFR tensor
"""

from .grid import grid as tfr_grid
from .decompose import decompose
