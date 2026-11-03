"""Export Region-level gait-cycle pointwise model inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[4]))

import plugin.default.stats_gait_cycle_default as cfg  # noqa: E402
from src.stats.gait_cycle import export_gait_cycle_region_tables  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build Region-level gait_cycle pointwise model inputs."
    )
    parser.add_argument(
        "--input-root",
        default=str(cfg.RAW_GAIT_ROOT),
        help="Raw gait table root. Defaults to the STNSNr raw gait summary directory.",
    )
    parser.add_argument(
        "--output-root",
        default=str(cfg.GAIT_CYCLE_STATS_ROOT),
        help="gait_cycle stats output root.",
    )
    parser.add_argument(
        "--task",
        default=cfg.DEFAULT_TASK,
        help="Task level to include. Defaults to Single.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing output files instead of overwriting them.",
    )
    return parser


def main() -> None:
    """Run the export command."""
    args = build_arg_parser().parse_args()
    report = export_gait_cycle_region_tables(
        input_root=args.input_root,
        output_root=args.output_root,
        task=args.task,
        overwrite=not args.no_overwrite,
    )
    print(f"Output root: {report.output_root}")
    print(f"Trace tables: {len(report.trace_paths)}")
    print(f"Band input tables: {len(report.input_paths)}")
    print(f"Region trace rows: {report.n_trace_rows}")
    print(f"Long timepoint rows: {report.n_long_rows}")


if __name__ == "__main__":
    main()
