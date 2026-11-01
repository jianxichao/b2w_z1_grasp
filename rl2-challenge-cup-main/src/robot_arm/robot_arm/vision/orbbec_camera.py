"""Minimal Orbbec Gemini 2 RGB-D capture."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RgbdFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    camera_matrix: np.ndarray
    timestamp: float
    distortion_coefficients: np.ndarray | None = None
    # 原始深度整数转换为米时使用的比例；Gemini 2 常见值为 0.001 m/单位。
    depth_scale_m: float = 0.001
    depth_raw: np.ndarray | None = None


class OrbbecCamera:
    """Return synchronized 1280x720 color and color-aligned depth in meters."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        depth_width: int = 1280,
        depth_height: int = 800,
        min_depth_m: float = 0.02,
        max_depth_m: float = 10.0,
        timeout_ms: int = 1000,
        calibrated_camera_matrix: np.ndarray | None = None,
        calibration_width: int | None = None,
        calibration_height: int | None = None,
        color_only: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.timeout_ms = timeout_ms
        self.color_only = color_only
        self.calibrated_camera_matrix = (
            None
            if calibrated_camera_matrix is None
            else np.asarray(calibrated_camera_matrix, dtype=float)
        )
        self.calibration_width = int(calibration_width or width)
        self.calibration_height = int(calibration_height or height)
        if self.calibrated_camera_matrix is not None and self.calibrated_camera_matrix.shape != (3, 3):
            raise ValueError("calibrated_camera_matrix must be 3x3")
        self._sdk = self._pipeline = self._align = self._camera_matrix = None
        self._distortion_coefficients = None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import pyorbbecsdk as sdk
        except ImportError as exc:
            raise ImportError("Orbbec capture requires pyorbbecsdk.") from exc

        pipeline = None
        for _ in range(3):
            try:
                pipeline = sdk.Pipeline()
                break
            except Exception:
                time.sleep(0.5)
        if pipeline is None:
            raise RuntimeError("Gemini 2 无法打开：若上次程序被系统杀死，请拔插相机后重试")
        config = sdk.Config()
        color = self._color_profile(sdk, pipeline)
        config.enable_stream(color)
        if not self.color_only:
            depth_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
            try:
                # AlignFilter converts Gemini 2's 1280x800 depth to color coordinates.
                depth = depth_profiles.get_video_stream_profile(
                    self.depth_width, self.depth_height, sdk.OBFormat.Y16, self.fps
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini 2 无 {self.depth_width}x{self.depth_height}@{self.fps} Y16 深度流，无法与彩色图对齐"
                ) from exc
            config.enable_stream(depth)
            pipeline.enable_frame_sync()
        pipeline.start(config)
        try:
            intrinsic = color.get_intrinsic()
            self._validate_calibrated_intrinsics(intrinsic)
        except Exception:
            pipeline.stop()
            raise
        self._camera_matrix = np.array(
            [[intrinsic.fx, 0.0, intrinsic.cx], [0.0, intrinsic.fy, intrinsic.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = color.get_distortion() if hasattr(color, "get_distortion") else None
        if distortion is not None:
            self._distortion_coefficients = np.array(
                [
                    distortion.k1,
                    distortion.k2,
                    distortion.p1,
                    distortion.p2,
                    distortion.k3,
                    distortion.k4,
                    distortion.k5,
                    distortion.k6,
                ],
                dtype=np.float64,
            )
        self._sdk = sdk
        self._pipeline = pipeline
        self._align = (
            None
            if self.color_only
            else sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
        )

    def _validate_calibrated_intrinsics(self, intrinsic: object) -> None:
        """Reject a stream profile that does not match the saved 1280x720 calibration."""
        if self.calibrated_camera_matrix is None:
            return
        matrix = self.calibrated_camera_matrix.copy()
        matrix[0, :] *= float(self.width) / float(self.calibration_width)
        matrix[1, :] *= float(self.height) / float(self.calibration_height)
        focal_ok = np.allclose(
            [intrinsic.fx, intrinsic.fy], [matrix[0, 0], matrix[1, 1]], rtol=0.03
        )
        center_ok = np.allclose(
            [intrinsic.cx, intrinsic.cy], [matrix[0, 2], matrix[1, 2]], atol=8.0, rtol=0.0
        )
        if not focal_ok or not center_ok:
            raise RuntimeError(
                "Gemini 2 intrinsics do not match the scaled eye-in-hand calibration."
            )

    def read(self) -> RgbdFrame:
        self.start()
        assert self._pipeline is not None and self._sdk is not None
        if self.color_only:
            for _ in range(8):
                frames = self._pipeline.wait_for_frames(self.timeout_ms)
                color_frame = None if frames is None else frames.get_color_frame()
                if color_frame is not None:
                    break
            if color_frame is None:
                raise RuntimeError("Gemini 2 连续 8 帧 RGB 采集失败")
            color = _frame_to_bgr(color_frame, self._sdk)
            return RgbdFrame(
                color,
                np.empty((0, 0), dtype=np.float32),
                self._camera_matrix.copy(),
                time.time(),
                self._distortion_coefficients.copy()
                if self._distortion_coefficients is not None
                else None,
            )
        assert self._align is not None
        frames = color_frame = depth_frame = None
        for _ in range(8):
            raw = self._pipeline.wait_for_frames(self.timeout_ms)
            if raw is None:
                continue
            aligned = self._align.process(raw)
            frames = None if aligned is None else aligned.as_frame_set()
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is not None and depth_frame is not None:
                break
        if color_frame is None or depth_frame is None:
            raise RuntimeError("Gemini 2 连续 8 帧 RGB-D 对齐失败")
        timestamp = time.time()
        color = _frame_to_bgr(color_frame, self._sdk)
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )
        depth_scale_m = float(depth_frame.get_depth_scale()) / 1000.0
        if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
            raise RuntimeError(f"Gemini 2 深度比例无效：{depth_scale_m}")
        depth_m = depth.astype(np.float32) * depth_scale_m
        valid = (depth_m >= self.min_depth_m) & (depth_m <= self.max_depth_m)
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
        if color.shape[:2] != (self.height, self.width) or depth_m.shape != color.shape[:2]:
            raise RuntimeError(
                f"Expected aligned {self.width}x{self.height} RGB-D, got "
                f"color={color.shape[1]}x{color.shape[0]}, depth={depth_m.shape[::-1]}."
            )
        distortion = (
            None
            if self._distortion_coefficients is None
            else self._distortion_coefficients.copy()
        )
        return RgbdFrame(
            color,
            depth_m,
            self._camera_matrix.copy(),
            timestamp,
            distortion,
            depth_scale_m,
            depth.copy(),
        )

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._sdk = self._pipeline = self._align = self._camera_matrix = None
        self._distortion_coefficients = None

    def __enter__(self) -> "OrbbecCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _color_profile(self, sdk: object, pipeline: object) -> object:
        profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        errors = []
        for name in ("RGB", "MJPG"):
            image_format = getattr(sdk.OBFormat, name, None)
            if image_format is None:
                continue
            try:
                return profiles.get_video_stream_profile(
                    self.width, self.height, image_format, self.fps
                )
            except Exception as exc:  # SDK exception types vary by release.
                errors.append(str(exc))
        detail = "; ".join(errors)
        raise RuntimeError(
            f"Gemini 2 has no {self.width}x{self.height}@{self.fps} RGB/MJPG profile. {detail}"
        )


def _frame_to_bgr(frame: object, sdk: object) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Orbbec color conversion requires opencv-python.") from exc

    width, height = frame.get_width(), frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    image_format = frame.get_format()
    if image_format == sdk.OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if image_format == getattr(sdk.OBFormat, "BGR", None):
        return data.reshape(height, width, 3).copy()
    if image_format == sdk.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    if image_format == getattr(sdk.OBFormat, "YUYV", None):
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    raise RuntimeError(f"Unsupported Orbbec color format: {image_format}")
