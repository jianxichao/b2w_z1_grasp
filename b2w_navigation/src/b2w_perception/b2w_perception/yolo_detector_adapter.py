"""Ultralytics YOLO adapter for offline batch image/video detection.

This adapter is used only by offline tools on the development PC. It wraps
the Ultralytics API (model.predict/track) and converts results into project
``Detection2D`` objects. The online robot node does not use this class; it
runs ONNX Runtime directly in ``yolo_target_perception_node.py``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from .schema import BBoxXYXY, Detection2D, Header


class YoloDetectorAdapter:
    """YOLO image detection and persistent ByteTrack video tracking."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.5,
        class_map: dict[str, str] | None = None,
        project_class_ids: dict[str, int] | None = None,
        allowed_classes: set[str] | None = None,
        frame_id: str = "camera_color_optical_frame",
        device: str = "cpu",
        post_mapping_nms_enabled: bool = True,
        post_mapping_nms_iou: float = 0.70,
        agnostic_nms: bool = False,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model was not found: {model_path}. "
                "Pass an existing local --model path; automatic model download is disabled."
            )
        if not 0.0 <= post_mapping_nms_iou <= 1.0:
            raise ValueError("post_mapping_nms_iou must be between 0 and 1")

        cache_root = Path(tempfile.gettempdir()) / "b2w_perception"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Real YOLO detection requires optional dependencies. "
                "Install them with: python -m pip install -r requirements-detection.txt"
            ) from exc

        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.confidence = confidence
        self.class_map = class_map or {}
        self.project_class_ids = project_class_ids or {}
        self.allowed_classes = allowed_classes
        self.frame_id = frame_id
        self.post_mapping_nms_enabled = post_mapping_nms_enabled
        self.post_mapping_nms_iou = post_mapping_nms_iou
        self.agnostic_nms = agnostic_nms
        self.tracking_dependencies_checked = False
        self.last_raw_detection_count = 0
        self.last_project_detection_count = 0
        self.last_untracked_count = 0
        self.last_nms_removed = 0

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def reset_tracker(self) -> None:
        """Drop Ultralytics predictor state before processing a new video."""
        self.model.predictor = None

    def detect_image(self, image: Any, timestamp: float = 0.0) -> list[Detection2D]:
        """Detect objects in one image frame."""
        results = self.model(
            image,
            verbose=False,
            conf=self.confidence,
            agnostic_nms=getattr(self, "agnostic_nms", False),
        )
        detections = self._convert_result(results[0], image, timestamp, tracking=False)
        return _assign_image_ids(detections)

    def track_frame(
        self,
        image: Any,
        timestamp: float = 0.0,
        tracker: str = "bytetrack.yaml",
    ) -> list[Detection2D]:
        """Detect and track objects across video frames with ByteTrack."""
        if not self.tracking_dependencies_checked:
            try:
                import lap  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "ByteTrack requires lap>=0.5.12. Install requirements-detection.txt "
                    "outside the application before running video tracking."
                ) from exc
            self.tracking_dependencies_checked = True
        results = self.model.track(
            image,
            persist=True,
            tracker=tracker,
            verbose=False,
            conf=self.confidence,
            agnostic_nms=self.agnostic_nms,
        )
        return self._convert_result(results[0], image, timestamp, tracking=True)

    def _convert_result(
        self,
        result: Any,
        image: Any,
        timestamp: float,
        tracking: bool,
    ) -> list[Detection2D]:
        boxes = result.boxes
        self.last_raw_detection_count = 0 if boxes is None else len(boxes)
        self.last_project_detection_count = 0
        self.last_untracked_count = 0
        self.last_nms_removed = 0
        if boxes is None or len(boxes) == 0:
            return []

        height, width = image.shape[:2]
        detections: list[Detection2D] = []
        for box in boxes:
            raw_class_id = int(_scalar(box.cls))
            raw_class_name = str(self.model.names[raw_class_id])
            class_name = self.class_map.get(raw_class_name, raw_class_name)
            if self.allowed_classes is not None and class_name not in self.allowed_classes:
                continue
            if class_name not in self.project_class_ids:
                continue

            self.last_project_detection_count += 1
            object_id = ""
            if tracking:
                track_id = _optional_track_id(box)
                if track_id is None:
                    self.last_untracked_count += 1
                    continue
                object_id = format_track_object_id(track_id, class_name)

            x1, y1, x2, y2 = [_scalar(value) for value in box.xyxy[0]]
            x1 = min(max(float(x1), 0.0), float(width))
            y1 = min(max(float(y1), 0.0), float(height))
            x2 = min(max(float(x2), 0.0), float(width))
            y2 = min(max(float(y2), 0.0), float(height))
            if x1 >= x2 or y1 >= y2:
                continue
            detections.append(
                Detection2D(
                    header=Header(self.frame_id, timestamp),
                    object_id=object_id,
                    class_name=class_name,
                    class_id=self.project_class_ids[class_name],
                    confidence=float(_scalar(box.conf)),
                    bbox_xyxy=BBoxXYXY(x1, y1, x2, y2),
                    image_width=int(width),
                    image_height=int(height),
                )
            )

        if getattr(self, "post_mapping_nms_enabled", True):
            before = len(detections)
            detections = post_mapping_nms(
                detections, getattr(self, "post_mapping_nms_iou", 0.70)
            )
            self.last_nms_removed = before - len(detections)
        return detections


def preprocess_image(
    image: Any,
    output_width: int = 640,
    output_height: int = 480,
    mode: str = "auto",
    letterbox_value: int = 114,
) -> tuple[Any, dict[str, Any]]:
    """Resize 4:3 input directly and letterbox other aspect ratios."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError("Image preprocessing requires numpy and opencv-python.") from exc
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output dimensions must be positive")
    if mode not in {"auto", "resize", "letterbox"}:
        raise ValueError("preprocess mode must be auto, resize, or letterbox")

    source_height, source_width = image.shape[:2]
    source_ratio = source_width / source_height
    output_ratio = output_width / output_height
    selected_mode = mode
    if mode == "auto":
        selected_mode = "resize" if abs(source_ratio - output_ratio) <= 0.01 else "letterbox"

    if selected_mode == "resize":
        output = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        scale_x = output_width / source_width
        scale_y = output_height / source_height
        scale = min(scale_x, scale_y)
        pad_x = 0
        pad_y = 0
    else:
        scale = min(output_width / source_width, output_height / source_height)
        resized_width = max(1, min(output_width, int(round(source_width * scale))))
        resized_height = max(1, min(output_height, int(round(source_height * scale))))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (output_width - resized_width) // 2
        pad_y = (output_height - resized_height) // 2
        output = np.full(
            (output_height, output_width, image.shape[2]),
            letterbox_value,
            dtype=image.dtype,
        )
        output[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        scale_x = resized_width / source_width
        scale_y = resized_height / source_height

    metadata = {
        "source_width": int(source_width),
        "source_height": int(source_height),
        "output_width": int(output_width),
        "output_height": int(output_height),
        "mode": selected_mode,
        "scale": float(scale),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
    }
    return output, metadata


def format_track_object_id(track_id: int, class_name: str) -> str:
    if track_id < 0:
        raise ValueError("track_id must not be negative")
    return f"track_{track_id:06d}_{class_name}"


def post_mapping_nms(
    detections: list[Detection2D], iou_threshold: float = 0.70
) -> list[Detection2D]:
    """Suppress overlapping boxes only after they map to the same project class."""
    kept: list[Detection2D] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            detection.class_name == other.class_name
            and _bbox_iou(detection.bbox_xyxy, other.bbox_xyxy) >= iou_threshold
            for other in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def _assign_image_ids(detections: list[Detection2D]) -> list[Detection2D]:
    for index, detection in enumerate(detections):
        detection.object_id = f"det_{index:03d}_{detection.class_name}"
    return detections


def _bbox_iou(first: BBoxXYXY, second: BBoxXYXY) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _optional_track_id(box: Any) -> int | None:
    value = getattr(box, "id", None)
    if value is None:
        return None
    try:
        return int(_scalar(value))
    except (TypeError, ValueError):
        return None


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
