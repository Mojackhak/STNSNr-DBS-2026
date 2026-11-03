from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO
from mmpose.apis import MMPoseInferencer


@dataclass(frozen=True)
class PoseTrackerConfig:
    """Configuration for VideoPoseTracker."""
    yolo_model_path: str | Path
    mmpose_model: str = "rtmpose-l"
    scale: float = 1.2
    device: str = "cuda:0"
    conf_threshold: float = 0.3


class VideoPoseTracker:
    """YOLO + MMPose video pose tracker with optional DLC-format export (H5/CSV)."""

    # COCO-17 keypoints
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    # 0-based skeleton connections (converted from your 1-based list)
    SKELETON = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6),
    ]

    def __init__(
        self,
        yolo_model_path: str | Path,
        mmpose_model: str = "rtmpose-l",
        scale: float = 1.2,
        device: str = "cuda:0",
        conf_threshold: float = 0.3,
    ) -> None:
        """
        Initialize YOLO detector and MMPose model.

        Args:
            yolo_model_path: Path to YOLO model weights.
            mmpose_model: MMPose model name (default: 'rtmpose-l').
            scale: Scale factor for bounding box enlargement.
            device: Device for inference ('cuda:0' or 'cpu').
            conf_threshold: Minimum keypoint confidence for visualization.
        """
        self.yolo_model = YOLO(str(Path(yolo_model_path)))
        self.model_name = mmpose_model
        self.scale = float(scale)
        self.conf_threshold = float(conf_threshold)

        # Respect user's device choice, but fall back if CUDA is unavailable.
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.pose_model = MMPoseInferencer(self.model_name, device=self.device)

    def enlarge_bbox(self, bbox: np.ndarray | list[float], image_shape: tuple[int, int] | None = None) -> list[float]:
        """
        Enlarge bounding box by a scale factor.

        Args:
            bbox: [x1, y1, x2, y2].
            image_shape: (height, width) to clip coordinates.

        Returns:
            Enlarged bbox [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = map(float, bbox)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        new_w, new_h = w * self.scale, h * self.scale

        new_x1 = cx - new_w / 2.0
        new_y1 = cy - new_h / 2.0
        new_x2 = cx + new_w / 2.0
        new_y2 = cy + new_h / 2.0

        if image_shape is not None:
            h_img, w_img = image_shape
            new_x1 = max(0.0, new_x1)
            new_y1 = max(0.0, new_y1)
            new_x2 = min(float(w_img), new_x2)
            new_y2 = min(float(h_img), new_y2)

        return [new_x1, new_y1, new_x2, new_y2]

    def detect_person(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Detect the largest person in a frame using YOLO.

        Args:
            frame: Input image (H, W, 3) BGR.

        Returns:
            bbox [x1, y1, x2, y2] or None.
        """
        results = self.yolo_model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.detach().cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return xyxy[int(np.argmax(areas))]

    def get_keypoints_inferencer(self, cropped_img: np.ndarray) -> np.ndarray | None:
        """
        Run MMPoseInferencer on a cropped image.

        Args:
            cropped_img: ROI image as NumPy array.

        Returns:
            (K, 3) array of (x, y, score) or None.
        """
        gen = self.pose_model(cropped_img, return_vis=False)
        try:
            result = next(gen)
        except StopIteration:
            return None

        preds = result.get("predictions", [])
        if not preds or not preds[0]:
            return None

        inst = preds[0][0]
        kpts = inst.get("keypoints")
        scores = inst.get("keypoint_scores")
        if kpts is None or scores is None:
            return None

        kpts = np.asarray(kpts, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        return np.hstack((kpts, scores[:, None]))

    def visualize_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, bbox: np.ndarray | None = None) -> np.ndarray:
        """
        Draw keypoints and skeleton on a frame.

        Args:
            frame: Input image.
            keypoints: (K, 3) array.
            bbox: Optional bbox [x1, y1, x2, y2].

        Returns:
            Image with annotations.
        """
        vis = frame.copy()

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        thr = self.conf_threshold
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > thr:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.putText(
                    vis,
                    str(i),
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        for p1, p2 in self.SKELETON:
            if keypoints[p1, 2] > thr and keypoints[p2, 2] > thr:
                pt1 = (int(keypoints[p1, 0]), int(keypoints[p1, 1]))
                pt2 = (int(keypoints[p2, 0]), int(keypoints[p2, 1]))
                cv2.line(vis, pt1, pt2, (0, 255, 255), 2)

        return vis

    @staticmethod
    def df2dlc(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Convert a flat keypoints DataFrame to DeepLabCut/Anipose MultiIndex format.

        Expected columns:
            - 'frame'
            - '{keypoint}_x', '{keypoint}_y', '{keypoint}_conf'

        Returns:
            DataFrame indexed by 'frame' with MultiIndex columns:
            (scorer=model_name, bodyparts=keypoint, coords=x/y/likelihood)
        """
        if "frame" not in df.columns:
            raise ValueError(f"DataFrame must include 'frame'; got columns: {df.columns.tolist()}")

        frame = df["frame"].copy()
        data_dict: dict[tuple[str, str, str], pd.Series] = {}

        for col in df.columns:
            if col == "frame":
                continue

            parts = col.rsplit("_", 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse column '{col}'; expected format 'keypoint_x/y/conf'.")

            keypoint, suffix = parts
            if suffix == "conf":
                coord = "likelihood"
            elif suffix in ("x", "y"):
                coord = suffix
            else:
                raise ValueError(f"Column '{col}' has unsupported suffix '{suffix}'.")

            data_dict[(model_name, keypoint, coord)] = df[col].copy()

        tuples = list(data_dict.keys())
        dlc_df = pd.DataFrame({t: data_dict[t] for t in tuples})
        dlc_df.columns = pd.MultiIndex.from_tuples(tuples, names=["scorer", "bodyparts", "coords"])
        dlc_df.insert(0, "frame", frame)
        dlc_df.set_index("frame", inplace=True)
        return dlc_df

    def _empty_record(self, frame_idx: int) -> dict[str, Any]:
        """Return a record filled with NaNs for all keypoints."""
        rec: dict[str, Any] = {"frame": frame_idx}
        for name in self.KEYPOINT_NAMES:
            rec[f"{name}_x"] = np.nan
            rec[f"{name}_y"] = np.nan
            rec[f"{name}_conf"] = np.nan
        return rec

    def process_video(
        self,
        video_path: str | Path,
        output_coords: str | Path | None = None,
        visualize: bool = False,
        output_video: str | Path | None = None,
        skip_frames: int = 1,
        dlc_format: bool = True,
    ) -> pd.DataFrame:
        """
        Process video and extract pose keypoints.

        Args:
            video_path: Input video path.
            output_coords: Optional path to save coordinates ('.csv' or '.h5').
            visualize: Whether to visualize in real time (press 'q' to quit).
            output_video: Optional path to save annotated video.
            skip_frames: Process every Nth frame (>= 1).
            dlc_format: Whether to convert output df to DLC MultiIndex format (for saving/return).

        Returns:
            DataFrame of keypoints per processed frame.
            If output_coords is provided and dlc_format=True, returns DLC MultiIndex df.
        """
        if skip_frames < 1:
            raise ValueError(f"skip_frames must be >= 1, got {skip_frames}")

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        out = None
        if output_video is not None:
            output_video = Path(output_video)
            output_video.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps_out = fps / skip_frames if fps > 0 else 0.0
            out = cv2.VideoWriter(str(output_video), fourcc, fps_out, (w, h))

        records: list[dict[str, Any]] = []
        k_default = np.zeros((len(self.KEYPOINT_NAMES), 3), dtype=np.float32)

        # tqdm total: approximate processed frames count if known
        n_steps = (total + skip_frames - 1) // skip_frames if total > 0 else None
        pbar = tqdm(total=n_steps, desc="Processing frames", leave=False)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue

                bbox = self.detect_person(frame)
                keypoints = k_default.copy()

                if bbox is None:
                    records.append(self._empty_record(frame_idx))
                else:
                    x1, y1, x2, y2 = map(int, self.enlarge_bbox(bbox, image_shape=frame.shape[:2]))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    crop = frame[y1:y2, x1:x2]
                    kpts = self.get_keypoints_inferencer(crop)

                    if kpts is None:
                        records.append(self._empty_record(frame_idx))
                    else:
                        # Map ROI keypoints back to full-frame coordinates
                        kpts = kpts.copy()
                        kpts[:, 0] += x1
                        kpts[:, 1] += y1
                        keypoints = kpts

                        rec: dict[str, Any] = {"frame": frame_idx}
                        for i, name in enumerate(self.KEYPOINT_NAMES):
                            rec[f"{name}_x"] = float(keypoints[i, 0])
                            rec[f"{name}_y"] = float(keypoints[i, 1])
                            rec[f"{name}_conf"] = float(keypoints[i, 2])
                        records.append(rec)

                if out is not None:
                    vis_frame = self.visualize_keypoints(frame, keypoints, bbox)
                    out.write(vis_frame)

                if visualize:
                    vis_frame = self.visualize_keypoints(frame, keypoints, bbox)
                    cv2.imshow("Pose Tracking", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                pbar.update(1)
                frame_idx += 1
        finally:
            pbar.close()
            cap.release()
            if out is not None:
                out.release()
            if visualize:
                cv2.destroyAllWindows()

        df = pd.DataFrame(records)

        if output_coords is not None:
            output_coords = Path(output_coords)
            output_coords.parent.mkdir(parents=True, exist_ok=True)

            if dlc_format:
                df = self.df2dlc(df, model_name=self.model_name)

            suffix = output_coords.suffix.lower()
            if suffix == ".h5":
                df.to_hdf(str(output_coords), key="df_with_missing", mode="w")
            elif suffix == ".csv":
                # IMPORTANT:
                # - flat df: 'frame' is a column -> index=False is OK
                # - DLC df: 'frame' is index -> must keep index in csv
                df.to_csv(str(output_coords), index=dlc_format)
            else:
                raise ValueError(f"Unsupported output format: {suffix}. Use '.csv' or '.h5'.")

        return df
