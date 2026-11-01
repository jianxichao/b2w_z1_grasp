"""Offline 2D detection JSON schema.

These dataclasses are for week-1 detector testing outside ROS2. They mirror the
fields that later need to be mapped into ROS2 messages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Header:
    frame_id: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Header":
        return cls(frame_id=str(data["frame_id"]), timestamp=float(data["timestamp"]))


@dataclass
class BBoxXYXY:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_u(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def center_v(self) -> float:
        return (self.y1 + self.y2) * 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBoxXYXY":
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
        )


@dataclass
class Detection2D:
    header: Header
    object_id: str
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: BBoxXYXY
    image_width: int
    image_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "object_id": self.object_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy.to_dict(),
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection2D":
        return cls(
            header=Header.from_dict(data["header"]),
            object_id=str(data["object_id"]),
            class_name=str(data["class_name"]),
            class_id=int(data["class_id"]),
            confidence=float(data["confidence"]),
            bbox_xyxy=BBoxXYXY.from_dict(data["bbox_xyxy"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
        )


def detections_to_jsonable(detections: list[Detection2D]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in detections]
