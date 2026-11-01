"""Turn tracked RGB-D detections into stable robot-base targets."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

from .orbbec_camera import OrbbecCamera
from .yolo_tracker import Detection, YoloTracker

if TYPE_CHECKING:
    from robot_arm.manipulation.types import Target


@dataclass(frozen=True)
class TrackedTarget:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    camera_xyz: np.ndarray
    base_xyz: np.ndarray
    timestamp: float


def median_bbox_depth(
    depth_m: np.ndarray,
    bbox_xyxy: Sequence[float],
    center_ratio: float = 0.35,
    min_depth_m: float = 0.02,
    max_depth_m: float = 10.0,
    min_pixels: int = 9,
) -> float | None:
    """Return robust depth from the center of a detection box."""
    if depth_m.ndim != 2 or not 0.0 < center_ratio <= 1.0:
        raise ValueError("depth must be 2D and center_ratio must be in (0, 1]")
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * center_ratio / 2.0, (y2 - y1) * center_ratio / 2.0
    height, width = depth_m.shape
    left, right = max(0, int(cx - half_w)), min(width, int(np.ceil(cx + half_w)))
    top, bottom = max(0, int(cy - half_h)), min(height, int(np.ceil(cy + half_h)))
    if left >= right or top >= bottom:
        return None
    values = depth_m[top:bottom, left:right]
    values = values[np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)]
    return float(np.median(values)) if values.size >= min_pixels else None


def depth_diagnostics(
    depth_m: np.ndarray, bbox_xyxy: Sequence[float], center_ratio: float = 0.35
) -> dict[str, object]:
    """Return same-frame pixel/ROI/depth values for RGB-D troubleshooting."""
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * center_ratio / 2.0, (y2 - y1) * center_ratio / 2.0
    height, width = depth_m.shape
    left, right = max(0, int(u - half_w)), min(width, int(np.ceil(u + half_w)))
    top, bottom = max(0, int(v - half_h)), min(height, int(np.ceil(v + half_h)))
    values = depth_m[top:bottom, left:right]
    values = values[np.isfinite(values) & (values > 0.0)]
    center_u, center_v = int(round(u)), int(round(v))
    center_depth = (
        float(depth_m[center_v, center_u])
        if 0 <= center_v < height and 0 <= center_u < width
        else 0.0
    )
    return {
        "image_shape": (int(height), int(width)),
        "bbox_xyxy": tuple(round(value, 1) for value in bbox_xyxy),
        "bbox_center": (round(u, 1), round(v, 1)),
        "depth_roi_xyxy": (left, top, right, bottom),
        "center_depth_m": center_depth,
        "median_depth_m": float(np.median(values)) if values.size else 0.0,
        "valid_depth_pixels": int(values.size),
    }


def deproject_pixel(
    pixel: Sequence[float],
    depth_m: float,
    camera_matrix: np.ndarray,
    distortion: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert a color pixel and aligned depth to an optical-frame point."""
    matrix = np.asarray(camera_matrix, dtype=float)
    if matrix.shape != (3, 3) or depth_m <= 0.0:
        raise ValueError("camera_matrix must be 3x3 and depth must be positive")
    u, v = map(float, pixel)
    if distortion is not None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("畸变校正需要 OpenCV") from exc
        point = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float64),
            matrix,
            np.asarray(distortion, dtype=float),
            P=matrix,
        )[0, 0]
        u, v = float(point[0]), float(point[1])
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if fx == 0.0 or fy == 0.0:
        raise ValueError("camera focal length must be non-zero")
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def camera_point_to_base(
    camera_xyz: Sequence[float], base_T_eef: np.ndarray, eef_T_camera: np.ndarray
) -> np.ndarray:
    """Apply base^T_eef * eef^T_camera to an optical-frame point."""
    base_T_eef = _transform(base_T_eef, "base_T_eef")
    eef_T_camera = _transform(eef_T_camera, "eef_T_camera")
    point = np.append(np.asarray(camera_xyz, dtype=float), 1.0)
    if point.shape != (4,):
        raise ValueError("camera_xyz must contain three values")
    return (base_T_eef @ eef_T_camera @ point)[:3]


def load_eye_in_hand(path: str | Path) -> np.ndarray:
    """Load the saved camera-to-EEF transform without inverting it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))["camera_to_eef"]
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(data["rotation_matrix"], dtype=float)
    transform[:3, 3] = np.asarray(data["translation"], dtype=float)
    return _transform(transform, "camera_to_eef")


def stable_target(
    frames: Sequence[Sequence[TrackedTarget]],
    min_frames: int = 3,
    max_spread_m: float = 0.015,
) -> TrackedTarget | None:
    """Lock one ID seen in every recent frame with a stable base position."""
    if min_frames < 1 or len(frames) < min_frames:
        return None
    recent = list(frames)[-min_frames:]
    common_ids = set(item.track_id for item in recent[0])
    for frame in recent[1:]:
        common_ids &= {item.track_id for item in frame}
    candidates = []
    for track_id in common_ids:
        items = [next(item for item in frame if item.track_id == track_id) for frame in recent]
        points = np.stack([item.base_xyz for item in items])
        mean = points.mean(axis=0)
        if np.linalg.norm(points - mean, axis=1).max() <= max_spread_m:
            camera_mean = np.stack([item.camera_xyz for item in items]).mean(axis=0)
            confidence = float(np.mean([item.confidence for item in items]))
            candidates.append(replace(items[-1], base_xyz=mean, camera_xyz=camera_mean, confidence=confidence))
    return max(candidates, key=lambda item: item.confidence, default=None)


class TargetProvider:
    """Acquire and lock a stable real-robot target; simulation uses another provider."""

    def __init__(
        self,
        camera: OrbbecCamera,
        tracker: YoloTracker,
        eef_T_camera: np.ndarray,
        base_T_eef: Callable[[], np.ndarray],
        target_class: str | None = None,
        acquire_confidence: float = 0.5,
        stable_frames: int = 3,
        max_spread_m: float = 0.015,
        lock_timeout_s: float = 3.0,
        show: bool = True,
        show_depth: bool = True,
        distortion: Sequence[float] | None = None,
        continuous_tracking: bool = True,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        if not 0.0 <= acquire_confidence <= 1.0:
            raise ValueError("acquire_confidence must be between zero and one")
        if stable_frames < 1 or max_spread_m <= 0.0 or lock_timeout_s <= 0.0:
            raise ValueError("stable_frames, max_spread_m and lock_timeout_s must be positive")
        self.camera = camera
        self.tracker = tracker
        self.eef_T_camera = _transform(eef_T_camera, "eef_T_camera")
        self.base_T_eef = base_T_eef
        self.target_class = target_class
        self.acquire_confidence = acquire_confidence
        self.stable_frames = stable_frames
        self.max_spread_m = max_spread_m
        self.lock_timeout_s = lock_timeout_s
        self.show = show
        self.show_depth = show_depth
        self.distortion = distortion
        self.continuous_tracking = continuous_tracking
        self.frame_callback = frame_callback
        self._history: deque[list[TrackedTarget]] = deque(maxlen=stable_frames)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._io_lock = threading.Lock()

    def observe(self) -> list[TrackedTarget]:
        with self._io_lock:
            frame = self.camera.read()
            detections = self.tracker.track(frame.color_bgr)
            targets = []
            display = []
            for detection in detections:
                if self.target_class is not None and detection.class_name != self.target_class:
                    display.append(detection)
                    continue
                if detection.confidence < self.acquire_confidence:
                    display.append(replace(detection, detail="below lock threshold"))
                    continue
                depth = median_bbox_depth(frame.depth_m, detection.bbox_xyxy)
                if depth is None:
                    display.append(replace(detection, detail="depth=invalid"))
                    continue
                # Intrinsics and distortion must describe this exact RGB-D
                # profile.  The hand-eye file is a fallback for older camera
                # implementations that do not expose per-frame distortion.
                distortion = (
                    frame.distortion_coefficients
                    if frame.distortion_coefficients is not None
                    else self.distortion
                )
                camera_xyz = deproject_pixel(
                    detection.center,
                    depth,
                    frame.camera_matrix,
                    distortion,
                )
                # 锁定阶段机械臂停在搜索位；用该帧处理时的反馈转到基座系。
                base_T_eef = self.base_T_eef()
                base_xyz = camera_point_to_base(camera_xyz, base_T_eef, self.eef_T_camera)
                diagnostic = depth_diagnostics(frame.depth_m, detection.bbox_xyxy)
                targets.append(_tracked_target(detection, camera_xyz, base_xyz, frame.timestamp))
                display.append(
                    replace(
                        detection,
                        detail=(
                            f"z={depth:.3f}m "
                            f"px={diagnostic['bbox_center']} "
                            f"roi={diagnostic['depth_roi_xyxy']} "
                            f"dc={diagnostic['center_depth_m']:.3f} "
                            f"C=({camera_xyz[0]:.3f},{camera_xyz[1]:.3f},{camera_xyz[2]:.3f}) "
                            f"B=({base_xyz[0]:.3f},{base_xyz[1]:.3f},{base_xyz[2]:.3f})"
                        ),
                    )
                )
            self._history.append(targets)
            if self.show:
                self.tracker.show(
                    frame.color_bgr,
                    display,
                    frame.depth_m if self.show_depth else None,
                )
            if self.frame_callback is not None:
                self.frame_callback(self.tracker.draw(frame.color_bgr, display))
            return targets

    def lock_observation(self) -> TrackedTarget:
        self._history.clear()
        deadline = time.monotonic() + self.lock_timeout_s
        while time.monotonic() < deadline:
            self.observe()
            target = stable_target(self._history, self.stable_frames, self.max_spread_m)
            if target is not None:
                return target
        raise TimeoutError("在规定时间内没有找到稳定的 YOLO RGB-D 目标")

    def lock(self) -> "Target":
        """Return the shared manipulation Target contract."""
        from robot_arm.manipulation.types import Pose, Target

        item = self.lock_observation()
        target = Target(
            target_id=f"{item.class_name}#{item.track_id}",
            pose=Pose.from_xyz(item.base_xyz),
            confidence=item.confidence,
            stamp_s=item.timestamp,
        )
        # OpenCV GUI 在 Linux 主线程外通常可用；macOS 要求窗口留在主线程。
        if self.show and self.continuous_tracking and sys.platform != "darwin":
            self._stop.clear()
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()
        return target

    def _monitor(self) -> None:
        """Keep visualization live without changing the locked grasp target."""
        while not self._stop.is_set():
            try:
                self._display_once()
            except (ImportError, RuntimeError, TimeoutError, ValueError):
                return

    def _display_once(self) -> None:
        """Update RGB-D tracking without reading robot state after target lock."""
        with self._io_lock:
            frame = self.camera.read()
            detections = self.tracker.track(frame.color_bgr)
            display = []
            for detection in detections:
                depth = median_bbox_depth(frame.depth_m, detection.bbox_xyxy)
                if depth is None:
                    display.append(replace(detection, detail="depth=invalid"))
                    continue
                distortion = (
                    frame.distortion_coefficients
                    if frame.distortion_coefficients is not None
                    else self.distortion
                )
                camera_xyz = deproject_pixel(
                    detection.center,
                    depth,
                    frame.camera_matrix,
                    distortion,
                )
                diagnostic = depth_diagnostics(frame.depth_m, detection.bbox_xyxy)
                display.append(
                    replace(
                        detection,
                        detail=(
                            f"z={depth:.3f}m "
                            f"px={diagnostic['bbox_center']} "
                            f"roi={diagnostic['depth_roi_xyxy']} "
                            f"dc={diagnostic['center_depth_m']:.3f} "
                            f"C=({camera_xyz[0]:.3f},{camera_xyz[1]:.3f},{camera_xyz[2]:.3f})"
                        ),
                    )
                )
            self.tracker.show(
                frame.color_bgr,
                display,
                frame.depth_m if self.show_depth else None,
            )

    def prepare_retry(self) -> None:
        """Stop only the monitor thread; keep camera and model open for reacquisition."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._history.clear()

    def close(self) -> None:
        self.prepare_retry()
        self.camera.close()
        self.tracker.close()


def _tracked_target(
    detection: Detection, camera_xyz: np.ndarray, base_xyz: np.ndarray, timestamp: float
) -> TrackedTarget:
    return TrackedTarget(
        detection.track_id,
        detection.class_name,
        detection.confidence,
        detection.bbox_xyxy,
        camera_xyz,
        base_xyz,
        timestamp,
    )


def _transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    ):
        raise ValueError(f"{name} must be a homogeneous 4x4 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.0:
        raise ValueError(f"{name} rotation must be right-handed and orthonormal")
    return matrix.copy()
