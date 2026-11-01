"""ONNX Runtime helpers for YOLO detection without PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def load_class_names(path: str | Path) -> list[str]:
    """Load a YOLO class-name list from a JSON array or {'names': [...]} dict."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("names")
        if data is None:
            raise ValueError("class names JSON must contain a 'names' list")
    return [str(name) for name in data]


def letterbox_square(
    image: np.ndarray,
    size: int = 640,
    color: int | tuple[int, int, int] = 114,
) -> tuple[np.ndarray, float, int, int]:
    """Resize an image into a square canvas keeping aspect ratio."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((size, size, image.shape[2]), color, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def preprocess_rgb(
    rgb: np.ndarray,
    size: int = 640,
) -> tuple[np.ndarray, float, int, int]:
    """Build the normalized CHW blob expected by the exported YOLO ONNX model."""
    boxed, scale, pad_x, pad_y = letterbox_square(rgb, size)
    blob = boxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), scale, pad_x, pad_y


def decode_yolo_output(
    output: np.ndarray,
    class_names: Sequence[str],
    confidence_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    image_size: tuple[int, int],
    iou_threshold: float = 0.7,
    max_detections: int = 300,
) -> list[tuple[str, float, float, float, float, float]]:
    """Decode YOLO ONNX output into (class, confidence, u, v, box_w, box_h)."""
    num_classes = len(class_names)
    if output.ndim == 3:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(
            "unexpected ONNX output shape: %r" % (output.shape,)
        )
    feature_count = 4 + num_classes
    if output.shape[0] in (feature_count, feature_count + 1) and output.shape[
        1
    ] not in (feature_count, feature_count + 1):
        output = output.T
    if output.shape[1] < 4 + num_classes:
        raise ValueError(
            "ONNX output width %d is smaller than 4 + %d classes"
            % (output.shape[1], num_classes)
        )

    has_objectness = output.shape[1] >= 5 + num_classes
    scores = (
        output[:, 5 : 5 + num_classes]
        if has_objectness
        else output[:, 4 : 4 + num_classes]
    )
    if has_objectness:
        scores = scores * output[:, 4:5]

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    mask = confidences >= confidence_threshold
    if not np.any(mask):
        return []

    boxes = output[mask, :4]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    half_width = boxes[:, 2] / 2.0
    half_height = boxes[:, 3] / 2.0
    x1 = (x_center - half_width - pad_x) / scale
    y1 = (y_center - half_height - pad_y) / scale
    x2 = (x_center + half_width - pad_x) / scale
    y2 = (y_center + half_height - pad_y) / scale

    width, height = image_size
    x1 = np.clip(x1, 0.0, float(width - 1))
    y1 = np.clip(y1, 0.0, float(height - 1))
    x2 = np.clip(x2, 0.0, float(width - 1))
    y2 = np.clip(y2, 0.0, float(height - 1))

    kept_indices = _class_aware_nms(
        x1, y1, x2, y2, class_ids, confidences, iou_threshold, max_detections
    )

    results: list[tuple[str, float, float, float, float, float]] = []
    for index in kept_indices:
        center_u = float((x1[index] + x2[index]) / 2.0)
        center_v = float((y1[index] + y2[index]) / 2.0)
        results.append(
            (
                str(class_names[class_ids[index]]),
                float(confidences[index]),
                center_u,
                center_v,
                float(x2[index] - x1[index]),
                float(y2[index] - y1[index]),
            )
        )
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def _class_aware_nms(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """Run per-class NMS and return indices kept after suppression."""
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        boxes = np.column_stack((x1[indices], y1[indices], x2[indices], y2[indices]))
        scores = confidences[indices]
        order = np.argsort(-scores)
        keep = []
        while len(order) > 0:
            current = order[0]
            keep.append(int(indices[current]))
            if len(order) == 1:
                break
            rest = order[1:]
            overlaps = _iou_boxes(boxes[current], boxes[rest])
            order = rest[overlaps <= iou_threshold]
        kept.extend(keep)

    kept.sort(key=lambda index: -float(confidences[index]))
    return kept[:max_detections]


def _iou_boxes(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return per-box IoU between one box and an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return np.where(union > 0.0, intersection / union, 0.0)
