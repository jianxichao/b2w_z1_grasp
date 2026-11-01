"""Configuration loading for target classes, runtime, and event policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_TARGET_CLASSES = ["science_target", "rock_obstacle", "base_marker", "drop_zone", "unknown"]
DEFAULT_CLASS_IDS = {
    "science_target": 0,
    "rock_obstacle": 1,
    "base_marker": 2,
    "drop_zone": 3,
    "unknown": 4,
}
DEFAULT_RUNTIME = {
    "detector_mode": "fake",
    "publish_rate_hz": 1.0,
    "frame_id": "camera_color_optical_frame",
    "target_candidates_topic": "/target_candidates",
    "task_event_topic": "/task_event",
    "perception_event_detail_topic": "/perception_event_detail",
    "task_state_topic": "/task_state",
    "fake_no_detection": False,
}
DEFAULT_EVENT_POLICY = {
    "target_class": "science_target",
    "acquire_confidence": 0.5,
    "maintain_confidence": 0.25,
    "seen_confirm_frames": 2,
    "seen_confirm_s": 0.10,
    "lost_confirm_frames": 3,
    "lost_confirm_s": 0.50,
    "lost_timeout_s": 3.0,
}
DEFAULT_PREPROCESS = {
    "enabled": True,
    "output_width": 640,
    "output_height": 480,
    "mode": "auto",
    "letterbox_value": 114,
}
DEFAULT_DETECTION = {
    "post_mapping_nms_enabled": True,
    "post_mapping_nms_iou": 0.70,
}


def load_target_config(path: str | Path | None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if path is not None:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Loading target_classes.yaml requires pyyaml.") from exc
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    target_classes = data.get("target_classes")
    if target_classes is None and "classes" in data:
        target_classes = [item["name"] for item in data["classes"] if "name" in item]
    target_classes = target_classes or DEFAULT_TARGET_CLASSES

    class_ids = dict(DEFAULT_CLASS_IDS)
    for item in data.get("classes", []):
        if "name" in item and "id" in item:
            class_ids[str(item["name"])] = int(item["id"])

    demo_mapping = data.get("demo_mapping", {})
    # Accept the Week 1 class_map shape while keeping it disabled by default.
    demo_class_map = demo_mapping.get("class_map", data.get("class_map", {}))
    demo_mapping_enabled = bool(demo_mapping.get("enabled", False))

    runtime = {**DEFAULT_RUNTIME, **data.get("runtime", {})}
    event_values = data.get("event_policy", {})
    # Accept the previous name while preferring the explicit acquire threshold.
    if "minimum_confidence" in event_values and "acquire_confidence" not in event_values:
        event_values = {**event_values, "acquire_confidence": event_values["minimum_confidence"]}
    event_values = {key: value for key, value in event_values.items() if key != "minimum_confidence"}
    event_policy = {**DEFAULT_EVENT_POLICY, **event_values}
    preprocess = {**DEFAULT_PREPROCESS, **data.get("preprocess", {})}
    detection = {**DEFAULT_DETECTION, **data.get("detection", {})}
    return {
        "target_classes": [str(item) for item in target_classes],
        "class_name_to_id": class_ids,
        "demo_mapping_enabled": demo_mapping_enabled,
        "demo_class_map": {str(key): str(value) for key, value in demo_class_map.items()},
        "class_map": (
            {str(key): str(value) for key, value in demo_class_map.items()}
            if demo_mapping_enabled
            else {}
        ),
        "runtime": runtime,
        "event_policy": event_policy,
        "preprocess": preprocess,
        "detection": detection,
        "raw": data,
    }
