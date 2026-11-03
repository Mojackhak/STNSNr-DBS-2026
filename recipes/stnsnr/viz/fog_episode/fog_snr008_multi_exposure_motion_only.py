from __future__ import annotations

import os

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


PROJECT = Path(os.environ["MYLFP_PROJECT_ROOT"])
VIDEO_DIR = PROJECT / "summary" / "stats" / "fog" / "videos"
VIDEO_PATH = VIDEO_DIR / "fog_SNr008.mp4"
ANNOTATION_DIR = VIDEO_DIR / "img"
OUTPUT_PATH = VIDEO_DIR / "fog_SNr008_multi_exposure_manual_human_outline.png"

SAVE_DEBUG_IMAGES = False
DEBUG_BACKGROUND_PATH = VIDEO_DIR / "fog_SNr008_manual_grayscale_background_debug.png"
DEBUG_HUMAN_MASK_UNION_PATH = VIDEO_DIR / "fog_SNr008_manual_human_mask_debug.png"

START_SEC = 0.0
END_SEC: Optional[float] = None
MAX_FRAMES = 10
BG_MAX_FRAMES = 200
RESIZE_TO: Optional[tuple[int, int]] = None

HUMAN_LABEL = "human"
BACKGROUND_BLUR_SIGMA = 10.0
CORE_ERODE_PX = 5
MASK_FEATHER_SIGMA = 3.0
HUMAN_ALPHA = 0.85
OUTLINE_COLOR_BGR = (224, 227, 215)
OUTLINE_WIDTH = 3


def move_existing_file_to_trash(path: Path) -> None:
    """Move an existing output file to the macOS Trash before rewriting it."""
    if not path.exists():
        return

    trash_dir = Path.home() / ".Trash"
    trash_dir.mkdir(exist_ok=True)

    target = trash_dir / path.name
    if target.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        target = trash_dir / f"{path.stem}-{timestamp}{path.suffix}"

    shutil.move(str(path), str(target))
    print(f"Moved existing file to Trash: {target}")


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image after preserving any existing file at the same path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    move_existing_file_to_trash(path)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def open_video(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> tuple[float, int, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is invalid.")
    return fps, total_frames, width, height


def get_frame_bounds(
    fps: float,
    total_frames: int,
    start_sec: float,
    end_sec: Optional[float],
) -> tuple[int, int]:
    start_frame = int(max(0.0, start_sec) * fps)
    if end_sec is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(total_frames - 1, int(max(0.0, end_sec) * fps))

    if end_frame <= start_frame:
        raise ValueError("Invalid time range: end_sec must be greater than start_sec.")
    return start_frame, end_frame


def make_sample_indices(start_frame: int, end_frame: int, max_count: int) -> list[int]:
    window_len = end_frame - start_frame + 1
    count = min(max_count, window_len)
    if count <= 0:
        raise ValueError("max_count must be positive.")
    return np.linspace(start_frame, end_frame, num=count, dtype=int).tolist()


def read_frame_at(
    cap: cv2.VideoCapture,
    frame_index: int,
    resize_to: Optional[tuple[int, int]],
) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index}.")
    if resize_to is not None:
        frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
    return frame


def estimate_median_background(
    video_path: Path,
    frame_indices: list[int],
    resize_to: Optional[tuple[int, int]],
) -> np.ndarray:
    cap = open_video(video_path)
    frames = []
    try:
        for frame_index in frame_indices:
            frames.append(read_frame_at(cap, frame_index, resize_to))
    finally:
        cap.release()

    if len(frames) < 5:
        raise RuntimeError("Not enough frames to estimate a stable median background.")

    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def render_reference_background(background: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.GaussianBlur(
        gray_bgr,
        ksize=(0, 0),
        sigmaX=BACKGROUND_BLUR_SIGMA,
        sigmaY=BACKGROUND_BLUR_SIGMA,
    )


def list_annotation_pairs(annotation_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = sorted(annotation_dir.glob("fog_SNr008_frame_*.png"))
    if len(image_paths) != MAX_FRAMES:
        raise RuntimeError(
            f"Expected {MAX_FRAMES} annotation images, found {len(image_paths)} in {annotation_dir}."
        )

    pairs = []
    for image_path in image_paths:
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation JSON: {annotation_path}")
        pairs.append((image_path, annotation_path))
    return pairs


def load_human_mask(annotation_path: Path, frame_shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame_height, frame_width = frame_shape[:2]
    json_width = int(data.get("imageWidth", frame_width))
    json_height = int(data.get("imageHeight", frame_height))
    if (json_width, json_height) != (frame_width, frame_height):
        raise ValueError(
            f"Annotation size does not match frame for {annotation_path.name}: "
            f"json={json_width}x{json_height}, frame={frame_width}x{frame_height}."
        )

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    shape_count = 0
    for shape in data.get("shapes", []):
        if shape.get("label") != HUMAN_LABEL or shape.get("shape_type") != "polygon":
            continue
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.shape[0] < 3:
            continue
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, frame_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, frame_height - 1)
        cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 255)
        shape_count += 1

    if shape_count == 0:
        raise ValueError(f"No '{HUMAN_LABEL}' polygon found in {annotation_path}")
    return mask, shape_count


def make_core_mask(human_mask: np.ndarray) -> np.ndarray:
    if CORE_ERODE_PX <= 0:
        return human_mask.copy()
    kernel_size = CORE_ERODE_PX * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(human_mask, kernel, iterations=1)


def make_human_alpha(human_mask: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    soft_mask = cv2.GaussianBlur(
        human_mask.astype(np.float32) / 255.0,
        ksize=(0, 0),
        sigmaX=MASK_FEATHER_SIGMA,
        sigmaY=MASK_FEATHER_SIGMA,
    )
    alpha = np.clip(soft_mask * HUMAN_ALPHA, 0.0, HUMAN_ALPHA)
    alpha[core_mask > 0] = HUMAN_ALPHA
    return alpha[..., None]


def draw_human_outline(image: np.ndarray, human_mask: np.ndarray) -> None:
    contours, _ = cv2.findContours(human_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    cv2.drawContours(
        image,
        contours,
        contourIdx=-1,
        color=OUTLINE_COLOR_BGR,
        thickness=OUTLINE_WIDTH,
        lineType=cv2.LINE_AA,
    )


def composite_manual_human_exposures(
    background: np.ndarray,
    annotation_pairs: list[tuple[Path, Path]],
    resize_to: Optional[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    output = render_reference_background(background).astype(np.float32)
    human_mask_union = np.zeros(background.shape[:2], dtype=np.uint8)
    outline_masks = []
    loaded_annotations = 0
    loaded_shapes = 0

    for image_path, annotation_path in annotation_pairs:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read annotation image: {image_path}")

        human_mask, shape_count = load_human_mask(annotation_path, frame.shape)
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            human_mask = cv2.resize(human_mask, resize_to, interpolation=cv2.INTER_NEAREST)

        core_mask = make_core_mask(human_mask)
        alpha = make_human_alpha(human_mask, core_mask)
        output = output * (1.0 - alpha) + frame.astype(np.float32) * alpha

        human_mask_union = np.maximum(human_mask_union, human_mask)
        outline_masks.append(human_mask)
        loaded_annotations += 1
        loaded_shapes += shape_count

    output = np.clip(output, 0, 255).astype(np.uint8)
    for human_mask in outline_masks:
        draw_human_outline(output, human_mask)

    return output, human_mask_union, loaded_annotations, loaded_shapes


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video does not exist: {VIDEO_PATH}")

    cap = open_video(VIDEO_PATH)
    try:
        fps, total_frames, width, height = get_video_info(cap)
    finally:
        cap.release()

    start_frame, end_frame = get_frame_bounds(
        fps=fps,
        total_frames=total_frames,
        start_sec=START_SEC,
        end_sec=END_SEC,
    )
    duration_s = total_frames / fps
    background_indices = make_sample_indices(start_frame, end_frame, BG_MAX_FRAMES)
    annotation_pairs = list_annotation_pairs(ANNOTATION_DIR)

    print(
        "Video: "
        f"{VIDEO_PATH} | {width}x{height}, {fps:.2f} fps, "
        f"{total_frames} frames, {duration_s:.2f} s"
    )
    print(
        "Frame window: "
        f"{start_frame}-{end_frame}; "
        f"background frames={len(background_indices)}, "
        f"manual annotation frames={len(annotation_pairs)}"
    )

    background = estimate_median_background(
        video_path=VIDEO_PATH,
        frame_indices=background_indices,
        resize_to=RESIZE_TO,
    )
    output, human_mask_union, loaded_annotations, loaded_shapes = composite_manual_human_exposures(
        background=background,
        annotation_pairs=annotation_pairs,
        resize_to=RESIZE_TO,
    )

    if SAVE_DEBUG_IMAGES:
        write_image(DEBUG_BACKGROUND_PATH, render_reference_background(background))
        write_image(
            DEBUG_HUMAN_MASK_UNION_PATH,
            human_mask_union,
        )

    write_image(OUTPUT_PATH, output)

    human_pixels = int(np.count_nonzero(human_mask_union))
    print(f"Loaded annotation JSON files: {loaded_annotations}")
    print(f"Loaded '{HUMAN_LABEL}' polygons: {loaded_shapes}")
    print(f"Human mask union pixels: {human_pixels}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
