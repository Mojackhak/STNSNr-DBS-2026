"""
Run Anipose commands from Python (API wrappers).

This module mirrors common `anipose <command>` CLI actions by calling
the corresponding Anipose Python API functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anipose.anipose as anipose


def anipose_load_config(fname: str | Path | None = None) -> dict[str, Any]:
    """Load an Anipose config (equivalent to CLI loading the config)."""
    if isinstance(fname, Path):
        fname = str(fname)
    return anipose.load_config(fname)


def anipose_calibrate(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose calibrate`."""
    from anipose.calibrate import calibrate_all

    print("Calibrating...")
    calibrate_all(config)
    print("Calibration completed.")


def anipose_analyze(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose analyze`."""
    from anipose.pose_videos import pose_videos_all

    print("Analyzing videos...")
    pose_videos_all(config)


def anipose_filter(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose filter`."""
    from anipose.filter_pose import filter_pose_all

    print("Filtering tracked points...")
    filter_pose_all(config)


def anipose_calibration_errors(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose calibration-errors`."""
    from anipose.calibration_errors import get_errors_all

    print("Getting all the calibration errors...")
    get_errors_all(config)


def anipose_tracking_errors(config: dict[str, Any], scorer: str | None = None) -> None:
    """Equivalent to: `anipose tracking-errors` (optionally with scorer)."""
    from anipose.tracking_errors import get_tracking_errors

    print("Comparing tracking to labeled data...")
    get_tracking_errors(config, scorer)


def anipose_filter_3d(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose filter-3d`."""
    from anipose.filter_3d import filter_pose_3d_all

    print("Filtering tracked points (3D)...")
    filter_pose_3d_all(config)


def anipose_triangulate(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose triangulate`."""
    from anipose.triangulate import triangulate_all

    print("Triangulating points...")
    triangulate_all(config)


def anipose_angles(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose angles`."""
    from anipose.compute_angles import compute_angles_all

    print("Computing angles...")
    compute_angles_all(config)


def anipose_summarize_3d(config: dict[str, Any]) -> None:
    """Summarize 3D outputs (angles + pose3d + optionally filtered pose3d)."""
    from anipose.summarize import summarize_angles, summarize_pose3d, summarize_pose3d_filtered

    print("Summarizing angles...")
    summarize_angles(config)

    print("Summarizing 3D pose...")
    summarize_pose3d(config)

    # Keep original logic (do not change behavior)
    if config["filter3d"]["enabled"]:
        print("Summarizing 3D pose filtered...")
        summarize_pose3d_filtered(config)


def anipose_summarize_2d(config: dict[str, Any]) -> None:
    """Summarize 2D outputs (raw + filtered)."""
    from anipose.summarize import summarize_pose2d, summarize_pose2d_filtered

    print("Summarizing pose 2d...")
    summarize_pose2d(config)

    print("Summarizing pose 2d filtered...")
    summarize_pose2d_filtered(config)


def anipose_summarize_2d_filter(config: dict[str, Any]) -> None:
    """Summarize 2D filtered only."""
    from anipose.summarize import summarize_pose2d_filtered

    print("Summarizing pose 2d filtered...")
    summarize_pose2d_filtered(config)


def anipose_summarize_errors(config: dict[str, Any]) -> None:
    """Summarize errors."""
    from anipose.summarize import summarize_errors

    print("Summarizing errors...")
    summarize_errors(config)


def anipose_extract_frames(
    config: dict[str, Any],
    nframes: int = 200,
    mode: str = "bad",
    no_pred: bool = False,
) -> None:
    """Equivalent to extracting frames for labeling."""
    from anipose.extract_frames import extract_frames_picked, extract_frames_random

    print("Extracting frames...")
    if no_pred:
        extract_frames_random(config, nframes)
    else:
        extract_frames_picked(config, mode, nframes)


def anipose_project_2d(config: dict[str, Any]) -> None:
    """Project 3D points back to 2D."""
    from anipose.project_2d import project_2d_all

    print("Projecting 3D points back to 2D...")
    project_2d_all(config)


def anipose_label_2d_proj(config: dict[str, Any]) -> None:
    """Make 2D videos from 3D projections."""
    from anipose.label_videos_proj import label_proj_all

    print("Making 2D videos from 3D projections...")
    label_proj_all(config)


def anipose_label_2d(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose label-2d`."""
    from anipose.label_videos import label_videos_all

    print("Labeling videos in 2D...")
    label_videos_all(config)


def anipose_label_2d_filter(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose label-2d-filter`."""
    from anipose.label_videos import label_videos_filtered_all

    print("Labeling videos in 2D (filtered)...")
    label_videos_filtered_all(config)


def anipose_label_3d(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose label-3d`."""
    from anipose.label_videos_3d import label_videos_3d_all

    print("Labeling videos in 3D...")
    label_videos_3d_all(config)


def anipose_label_3d_filter(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose label-3d-filter`."""
    from anipose.label_videos_3d import label_videos_3d_filtered_all

    print("Labeling videos in 3D (filtered)...")
    label_videos_3d_filtered_all(config)


def anipose_label_combined(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose label-combined`."""
    from anipose.label_combined import label_combined_all

    print("Labeling combined videos...")
    label_combined_all(config)


def anipose_label_filter_compare(config: dict[str, Any]) -> None:
    """Compare filtered vs raw tracking in labeled videos."""
    from anipose.label_filter_compare import label_filter_compare_all

    print("Labeling videos to compare filtered vs raw tracking...")
    label_filter_compare_all(config)


def anipose_draw_calibration(config: dict[str, Any], out_png: str | Path = "calibration.png") -> None:
    """Draw calibration board image to a PNG file."""
    from anipose.common import get_calibration_board_image

    import cv2  # optional dependency

    out_png = str(out_png)
    print("Drawing calibration board...")
    img = get_calibration_board_image(config)
    cv2.imwrite(out_png, img)


def anipose_train_autoencoder(config: dict[str, Any]) -> None:
    """Equivalent to: `anipose train-autoencoder`."""
    from anipose.train_autoencoder import train_autoencoder

    print("Training autoencoder...")
    train_autoencoder(config)


def anipose_visualizer() -> None:
    """Run Anipose visualizer server."""
    from anipose.server import run_server

    run_server()


def anipose_convert_videos(config: dict[str, Any]) -> None:
    """Convert videos according to config."""
    from anipose.convert_videos import convert_all

    convert_all(config)
