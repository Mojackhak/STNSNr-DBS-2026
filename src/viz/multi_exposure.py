import cv2
import numpy as np
from typing import Optional, Tuple

def _iter_sampled_frames(
    cap: cv2.VideoCapture,
    start_f: int,
    end_f: int,
    step: int,
    resize_to: Optional[Tuple[int, int]] = None,
):
    """Yield (frame_index, frame[BGR uint8]) for sampled frames in [start_f, end_f]."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    f = start_f
    while f <= end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (f - start_f) % step == 0:
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            yield f, frame
        f += 1

def _median_background(frames: list[np.ndarray]) -> np.ndarray:
    """Compute per-pixel median background from a list of BGR frames."""
    stack = np.stack(frames, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg

def _motion_mask_absdiff(
    frame: np.ndarray,
    background: np.ndarray,
    thresh: Optional[int] = 25,       # None -> Otsu
    blur_ksize: int = 5,              # 0/1 disables blur
    morph_ksize: int = 5,
    dilate_iter: int = 2,
    min_area: int = 800,
) -> np.ndarray:
    """Foreground mask from absdiff(frame, background)."""
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if thresh is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, k, iterations=dilate_iter)

    # Remove tiny blobs
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255
        mask = cleaned

    return mask

def multi_exposure_motion(
    video_path: str,
    out_path: str = "multi_exposure_motion.jpg",
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    step: Optional[int] = None,       # None -> auto to <= max_frames
    max_frames: int = 150,
    resize_to: Optional[Tuple[int, int]] = None,

    # background estimation
    bg_max_frames: int = 30,          # frames used to estimate background (median)

    # motion mask params
    thresh: Optional[int] = 25,       # None -> Otsu
    blur_ksize: int = 5,
    morph_ksize: int = 5,
    dilate_iter: int = 2,
    min_area: int = 800,

    # composition
    fg_blend: str = "darken",         # "copy" | "alpha" | "darken" | "lighten"
    alpha: float = 0.7,               # used when fg_blend="alpha"
    background_gray: bool = False,    # make background grayscale to pop foreground
    background_dim: float = 1.0,      # <1 dims background (e.g. 0.85)
    debug_save_bg: Optional[str] = None,
    debug_save_mask_union: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Fixed-camera multi-exposure that emphasizes motion:
    1) estimate clean background via median
    2) compute motion mask per frame (absdiff vs background)
    3) blend ONLY the moving pixels onto the background
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(max(0.0, start_sec) * fps)
    end_f = total - 1 if end_sec is None else min(total - 1, int(max(0.0, end_sec) * fps))
    if end_f <= start_f:
        cap.release()
        raise ValueError("Invalid time range: end_sec must be greater than start_sec.")

    window_len = end_f - start_f + 1
    if step is None:
        step = max(1, window_len // max_frames)
    if verbose:
        print(f"FPS={fps:.2f}, frames={total}, window=[{start_f},{end_f}], step={step}")

    # -------- Pass 1: collect frames for background median --------
    sampled_count_est = (window_len + step - 1) // step
    bg_stride = max(1, sampled_count_est // max(1, bg_max_frames))
    bg_frames = []
    k = 0

    for _, frame in _iter_sampled_frames(cap, start_f, end_f, step, resize_to):
        k += 1
        if (k - 1) % bg_stride == 0:
            bg_frames.append(frame)
            if len(bg_frames) >= bg_max_frames:
                break

    if len(bg_frames) < 5:
        cap.release()
        raise RuntimeError("Not enough frames to estimate background (try smaller step or wider time range).")

    background = _median_background(bg_frames)

    # optional background styling
    bg = background.copy()
    if background_gray:
        g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if background_dim != 1.0:
        bg = np.clip(bg.astype(np.float32) * float(background_dim), 0, 255).astype(np.uint8)

    if debug_save_bg is not None:
        cv2.imwrite(debug_save_bg, bg)

    # -------- Pass 2: motion-only blending --------
    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video: {video_path}")

    out = bg.copy()
    mask_union = np.zeros(out.shape[:2], dtype=np.uint8)

    processed = 0
    for _, frame in _iter_sampled_frames(cap, start_f, end_f, step, resize_to):
        mask = _motion_mask_absdiff(
            frame, background,
            thresh=thresh,
            blur_ksize=blur_ksize,
            morph_ksize=morph_ksize,
            dilate_iter=dilate_iter,
            min_area=min_area,
        )
        mask_union = np.maximum(mask_union, mask)

        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            continue

        if fg_blend == "copy":
            out[ys, xs] = frame[ys, xs]

        elif fg_blend == "alpha":
            a = float(alpha)
            out_f = out.astype(np.float32)
            fr_f = frame.astype(np.float32)
            out_f[ys, xs] = out_f[ys, xs] * (1.0 - a) + fr_f[ys, xs] * a
            out = np.clip(out_f, 0, 255).astype(np.uint8)

        elif fg_blend == "darken":
            # best when the moving object is darker than background (your case)
            out[ys, xs] = np.minimum(out[ys, xs], frame[ys, xs])

        elif fg_blend == "lighten":
            out[ys, xs] = np.maximum(out[ys, xs], frame[ys, xs])

        else:
            cap.release()
            raise ValueError("fg_blend must be one of: copy | alpha | darken | lighten")

        processed += 1
        if verbose and processed % 20 == 0:
            print(f"Processed motion blending on {processed} sampled frames...")

    cap.release()

    if debug_save_mask_union is not None:
        cv2.imwrite(debug_save_mask_union, mask_union)

    cv2.imwrite(out_path, out)
    if verbose:
        print(f"Saved: {out_path}")
    return out
