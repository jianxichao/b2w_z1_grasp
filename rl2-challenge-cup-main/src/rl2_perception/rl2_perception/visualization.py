"""Drawing helpers for target detection results."""

from __future__ import annotations

from typing import Any

from .schema import Detection2D


def draw_detections(image: Any, detections: list[Detection2D]) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Drawing detections requires opencv-python.") from exc

    output = image.copy()
    for det in detections:
        x1 = int(det.bbox_xyxy.x1)
        y1 = int(det.bbox_xyxy.y1)
        x2 = int(det.bbox_xyxy.x2)
        y2 = int(det.bbox_xyxy.y2)
        label = f"{det.object_id} {det.confidence:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 180, 40), 2)
        cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 180, 40), 2)
    return output
