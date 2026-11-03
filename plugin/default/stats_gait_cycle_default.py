"""Default configuration for gait-cycle pointwise statistics."""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(os.environ["MYLFP_PROJECT_ROOT"])
RAW_GAIT_ROOT = PROJECT_ROOT / "summary" / "table" / "raw" / "gait"
GAIT_CYCLE_STATS_ROOT = PROJECT_ROOT / "summary" / "stats" / "gait_cycle"

DEFAULT_TASK = "Single"
RAW_CYCLE_POINTS = 101
CYCLE_BIN_CENTERS = np.arange(0.5, 100.0, 1.0, dtype=float)

MEAN_TRACE_SUFFIX = "mean-trace"
REGION_TRACE_FILE_TEMPLATE = "{subparam}-mean-trace_region_normalized.pkl"
TIMEPOINT_INPUT_FILE = "timepoint_input.csv"
TIMEPOINT_INPUT_PKL = "timepoint_input.pkl"
TIMEPOINT_DEVIATION_FILE = "timepoint_deviation.csv"
TIMEPOINT_INTERVALS_FILE = "timepoint_intervals.csv"
TIMEPOINT_INTERVAL_CANDIDATES_FILE = "timepoint_interval_candidates.csv"
FIGURE_FILE = "Region_norm.pdf"

LMER_MODEL_DIR = "lmer_model"
RLMER_MODEL_DIR = "rlmer_model"
JOINT_MODEL_DIR = "joint_model"
POINTWISE_MODEL_DIRS = (LMER_MODEL_DIR, RLMER_MODEL_DIR)

SIGNIFICANCE_SHADOW_ALPHA = 0.15
SIGNIFICANCE_STAR_TOP_MARGIN = 0.04
SIGNIFICANCE_STAR_BOTTOM_MARGIN = 0.02
SIGNIFICANCE_STAR_LANE_STEP = 0.06
MIN_SIGNIFICANT_INTERVAL_SPAN = 10.0

LOCAL_PARAMS = frozenset({"tfr", "params", "burst"})
CONNECTION_PARAMS = frozenset({"conn", "gc", "psi"})
PARAM_BANDS = ("offset", "exponent")
SIDE_RELATED_COLUMNS = ("Side", "Side_cycle", "Hemi", "Lat")
REGION_TRACE_COLUMNS = (
    "ID",
    "Subject",
    "Region",
    "Side",
    "Side_cycle",
    "Hemi",
    "Lat",
    "Task",
    "Epoch",
    "Param",
    "Subparam",
    "Metric",
    "MetricClass",
    "Band",
    "Value",
)
