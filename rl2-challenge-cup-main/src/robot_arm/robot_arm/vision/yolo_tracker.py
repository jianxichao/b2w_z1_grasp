"""Small Ultralytics/ONNX YOLO adapter with persistent IDs."""

from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    detail: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class YoloTracker:
    """Track official YOLO classes with persistent ByteTrack IDs."""

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        confidence: float = 0.35,
        device: str = "auto",
        tracker: str = "bytetrack.yaml",
        image_size: int = 416,
        class_map: Optional[Dict[str, str]] = None,
        target_classes: Optional[Set[str]] = None,
    ) -> None:
        self._onnx = str(weights).lower().endswith(".onnx")
        self.model = None
        self.session = None
        self.names = {}
        if self._onnx:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError("ONNX tracking requires onnxruntime.") from exc
            self.session = ort.InferenceSession(
                str(weights), providers=["CPUExecutionProvider"]
            )
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.image_size = int(input_info.shape[-1])
            metadata = self.session.get_modelmeta().custom_metadata_map
            raw_names = metadata.get("names", "{}")
            try:
                self.names = {int(key): str(value) for key, value in ast.literal_eval(raw_names).items()}
            except (SyntaxError, ValueError, AttributeError):
                self.names = {0: "object"}
        else:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError("YOLO tracking requires ultralytics or .onnx weights.") from exc
            self.model = YOLO(weights)
            if device != "auto":
                self.model.to(device)
            self.image_size = int(image_size)
        self.confidence = confidence
        self.tracker = tracker
        self.device = device
        self.class_map = class_map or {}
        self.target_classes = target_classes
        self._tracks: Dict[int, Tuple[float, float, float, float]] = {}
        self._next_track_id = 1

    def track(self, image: Any) -> list[Detection]:
        if self._onnx:
            return self._track_onnx(image)
        assert self.model is not None
        result = self.model.track(
            image,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            imgsz=self.image_size,
            device="cpu" if self.device == "auto" else self.device,
            verbose=False,
        )[0]
        detections = []
        boxes = result.boxes
        if boxes is None:
            return detections
        for index, box in enumerate(boxes):
            raw_name = str(self.model.names[int(_number(box.cls))])
            name = self.class_map.get(raw_name, raw_name)
            if self.class_map and raw_name not in self.class_map:
                continue
            if self.target_classes is not None and name not in self.target_classes:
                continue
            xyxy = tuple(float(value) for value in _values(box.xyxy[0]))
            # ByteTrack can emit a valid first-frame detection before assigning
            # it a persistent ID.  Keep that detection for immediate grasp
            # evaluation instead of silently dropping an obvious target.
            track_id = -(index + 1) if box.id is None else int(_number(box.id))
            detections.append(
                Detection(track_id, name, float(_number(box.conf)), xyxy)
            )
        return detections

    def _track_onnx(self, image: Any) -> List[Detection]:
        """Run an Ultralytics detect ONNX export and assign stable IoU IDs."""
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("ONNX preprocessing requires OpenCV.") from exc
        assert self.session is not None
        height, width = image.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.image_size - resized_w) // 2
        pad_y = (self.image_size - resized_h) // 2
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = np.asarray(self.session.run(None, {self.input_name: blob})[0])
        predictions = output[0].T if output.ndim == 3 else output
        if predictions.shape[1] < 5:
            raise RuntimeError("Unsupported YOLO ONNX output shape: %s" % (output.shape,))
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]
        keep = scores >= self.confidence
        boxes_xywh, class_ids, scores = boxes_xywh[keep], class_ids[keep], scores[keep]
        boxes = []
        for cx, cy, box_w, box_h in boxes_xywh:
            x1 = max(0.0, (float(cx - box_w / 2.0) - pad_x) / scale)
            y1 = max(0.0, (float(cy - box_h / 2.0) - pad_y) / scale)
            x2 = min(float(width), (float(cx + box_w / 2.0) - pad_x) / scale)
            y2 = min(float(height), (float(cy + box_h / 2.0) - pad_y) / scale)
            boxes.append((x1, y1, x2, y2))
        nms_boxes = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self.confidence, 0.45)
        selected = [int(index) for index in np.asarray(indices).reshape(-1)] if len(indices) else []
        raw = []
        for index in selected:
            raw_name = self.names.get(int(class_ids[index]), str(int(class_ids[index])))
            name = self.class_map.get(raw_name, raw_name)
            if self.class_map and raw_name not in self.class_map:
                continue
            if self.target_classes is not None and name not in self.target_classes:
                continue
            raw.append((name, float(scores[index]), boxes[index]))
        return self._assign_track_ids(raw)

    def _assign_track_ids(
        self, raw: List[Tuple[str, float, Tuple[float, float, float, float]]]
    ) -> List[Detection]:
        available = set(self._tracks)
        updated = {}
        detections = []
        for name, confidence, box in sorted(raw, key=lambda item: item[1], reverse=True):
            best_id, best_iou = None, 0.0
            for track_id in available:
                overlap = _iou(box, self._tracks[track_id])
                if overlap > best_iou:
                    best_id, best_iou = track_id, overlap
            if best_id is None or best_iou < 0.30:
                best_id = self._next_track_id
                self._next_track_id += 1
            else:
                available.remove(best_id)
            updated[best_id] = box
            detections.append(Detection(best_id, name, confidence, box))
        self._tracks = updated
        return detections

    def draw(self, image: Any, detections: list[Detection]) -> Any:
        return draw_detections(image, detections)

    def show(
        self,
        image: Any,
        detections: list[Detection],
        depth_m: Optional[np.ndarray] = None,
    ) -> bool:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("YOLO display requires opencv-python.") from exc
        cv2.imshow("Gemini 2 RGB + YOLO", self.draw(image, detections))
        if depth_m is not None:
            cv2.imshow("Gemini 2 Depth", draw_depth(depth_m, detections))
        return cv2.waitKey(1) & 0xFF == ord("q")

    @staticmethod
    def close() -> None:
        try:
            import cv2
        except ImportError:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def draw_detections(image: Any, detections: list[Detection]) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Drawing detections requires opencv-python.") from exc
    output = image.copy()
    for item in detections:
        x1, y1, x2, y2 = (int(value) for value in item.bbox_xyxy)
        label = f"{item.class_name} #{item.track_id} {item.confidence:.2f}"
        if item.detail:
            label = f"{label} {item.detail}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 200, 40), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 200, 40),
            2,
        )
    return output


def draw_depth(depth_m: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Create an aligned metric-depth view with the same tracking boxes."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Depth display requires opencv-python.") from exc
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    scale = float(np.percentile(valid, 95)) if valid.size else 1.0
    image = cv2.applyColorMap(
        np.clip(depth / max(scale, 1e-6) * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return draw_detections(image, detections)


def _number(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _iou(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0
