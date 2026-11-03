"""
# Necessary parameters for mer/sorting pipeline
1. band_filter (to keep consistent with lead pipeline)
2. lfp_filter
3. notch_filter
4. compute_default_analyzer_extensions_cfg

"""


band_filter = (300, 6000)
lfp_filter = (1, 200)
notch_filter = [50, 100, 150, 200]
import os

wave_clus_path = os.environ.get("MYLFP_WAVECLUS_SCRIPT", "")

compute_default_analyzer_extensions_cfg = {
    "ms_before": 1.0, # milliseconds before threshold crossing
    "ms_after": 2.0, # milliseconds after
    "peak_relative_threshold": 0.2, # legacy alias mapped to current SpikeInterface threshold
}
