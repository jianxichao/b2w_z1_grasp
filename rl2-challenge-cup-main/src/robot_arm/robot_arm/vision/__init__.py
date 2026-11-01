"""Real-robot RGB-D perception."""

from .orbbec_camera import OrbbecCamera, RgbdFrame
from .target_provider import (
    TargetProvider,
    TrackedTarget,
    camera_point_to_base,
    deproject_pixel,
    load_eye_in_hand,
    median_bbox_depth,
    stable_target,
)
from .yolo_tracker import Detection, YoloTracker, draw_depth, draw_detections

__all__ = [
    "Detection",
    "OrbbecCamera",
    "RgbdFrame",
    "TargetProvider",
    "TrackedTarget",
    "YoloTracker",
    "camera_point_to_base",
    "deproject_pixel",
    "draw_detections",
    "draw_depth",
    "load_eye_in_hand",
    "median_bbox_depth",
    "stable_target",
]
