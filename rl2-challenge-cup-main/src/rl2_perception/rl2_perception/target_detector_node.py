"""Offline target detection CLI and ROS2 fake perception publisher."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config_io import load_target_config
from .schema import Detection2D, detections_to_jsonable


@dataclass
class TargetEventPolicy:
    """Track a selected target and emit task-compatible and detailed events."""

    target_class: str = "science_target"
    acquire_confidence: float = 0.5
    maintain_confidence: float = 0.25
    seen_confirm_frames: int = 2
    seen_confirm_s: float = 0.10
    lost_confirm_frames: int = 3
    lost_confirm_s: float = 0.50
    lost_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.maintain_confidence <= self.acquire_confidence <= 1.0:
            raise ValueError(
                "confidence thresholds must satisfy 0 <= maintain <= acquire <= 1"
            )
        if self.seen_confirm_frames <= 0 or self.lost_confirm_frames <= 0:
            raise ValueError("frame confirmation counts must be greater than zero")
        if self.seen_confirm_s < 0.0 or self.lost_confirm_s < 0.0:
            raise ValueError("confirmation durations must not be negative")
        if self.lost_timeout_s <= 0.0 or self.lost_timeout_s < self.lost_confirm_s:
            raise ValueError("lost_timeout_s must be positive and >= lost_confirm_s")
        self._task_state: str | None = None
        self._candidate_id: str | None = None
        self._candidate_first_seen_time: float | None = None
        self._seen_frames = 0
        self._locked_id: str | None = None
        self._locked_class_id: int | None = None
        self._lost_frames = 0
        self._missing_since: float | None = None
        self._last_valid_time: float | None = None
        self._last_confidence: float | None = None
        self._last_bbox: dict[str, float] | None = None
        self._lost_emitted = False

    @property
    def locked_id(self) -> str | None:
        return self._locked_id

    @property
    def locked_object_id(self) -> str | None:
        return self._locked_id

    @property
    def missed_frames(self) -> int:
        return self._lost_frames

    @property
    def missing_since(self) -> float | None:
        return self._missing_since

    @property
    def lost_emitted(self) -> bool:
        return self._lost_emitted

    @property
    def last_confidence(self) -> float | None:
        return self._last_confidence

    def update(
        self,
        detections: Sequence[Detection2D],
        timestamp: float,
        task_state: str | None = None,
        absence_reason: str | None = None,
    ) -> str | None:
        event, _ = self.update_with_detail(
            detections, timestamp, task_state=task_state, absence_reason=absence_reason
        )
        return event

    def update_with_detail(
        self,
        detections: Sequence[Detection2D],
        timestamp: float,
        task_state: str | None,
        absence_reason: str | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        self._handle_state_transition(task_state)
        if task_state == "SEARCH_TARGET":
            return self._update_search(detections, timestamp)
        if task_state == "APPROACH_TARGET":
            return self._update_approach(detections, timestamp, absence_reason)
        return None, None

    def _handle_state_transition(self, task_state: str | None) -> None:
        if task_state == self._task_state:
            return
        previous = self._task_state
        self._task_state = task_state
        if task_state == "SEARCH_TARGET":
            self._candidate_id = None
            self._candidate_first_seen_time = None
            self._seen_frames = 0
            self._locked_id = None
            self._locked_class_id = None
            self._lost_frames = 0
            self._missing_since = None
            self._lost_emitted = False
        elif previous == "APPROACH_TARGET":
            self._lost_frames = 0
            self._missing_since = None
            self._lost_emitted = False

    def _update_search(
        self, detections: Sequence[Detection2D], timestamp: float
    ) -> tuple[str | None, dict[str, Any] | None]:
        if self._locked_id is not None:
            return None, None
        valid = [
            item
            for item in detections
            if item.class_name == self.target_class
            and item.confidence >= self.acquire_confidence
            and _is_trackable_id(item.object_id)
        ]
        selected = max(valid, key=lambda item: item.confidence, default=None)
        if selected is None:
            self._candidate_id = None
            self._candidate_first_seen_time = None
            self._seen_frames = 0
            return None, None
        if selected.object_id == self._candidate_id:
            self._seen_frames += 1
        else:
            self._candidate_id = selected.object_id
            self._candidate_first_seen_time = timestamp
            self._seen_frames = 1
        seen_duration = (
            0.0
            if self._candidate_first_seen_time is None
            else max(0.0, timestamp - self._candidate_first_seen_time)
        )
        if (
            self._seen_frames < self.seen_confirm_frames
            or seen_duration < self.seen_confirm_s
        ):
            return None, None

        self._locked_id = selected.object_id
        self._locked_class_id = selected.class_id
        self._last_valid_time = timestamp
        self._last_confidence = selected.confidence
        self._last_bbox = selected.bbox_xyxy.to_dict()
        self._seen_frames = 0
        self._candidate_first_seen_time = None
        detail = {
            "event": "target_seen",
            "target_id": selected.object_id,
            "class_name": selected.class_name,
            "class_id": selected.class_id,
            "confidence": selected.confidence,
            "bbox_xyxy": selected.bbox_xyxy.to_dict(),
            "timestamp": timestamp,
        }
        return f"target_seen:{selected.object_id}", detail

    def _update_approach(
        self,
        detections: Sequence[Detection2D],
        timestamp: float,
        absence_reason: str | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if self._locked_id is None or self._lost_emitted:
            return None, None

        locked = next((item for item in detections if item.object_id == self._locked_id), None)
        if (
            locked is not None
            and locked.class_name == self.target_class
            and locked.confidence >= self.maintain_confidence
        ):
            self._last_valid_time = timestamp
            self._last_confidence = locked.confidence
            self._last_bbox = locked.bbox_xyxy.to_dict()
            self._lost_frames = 0
            self._missing_since = None
            return None, None

        self._lost_frames += 1
        if self._missing_since is None:
            self._missing_since = timestamp
        reason = absence_reason or _classify_absence(
            detections, self.target_class, self.maintain_confidence, self._locked_id
        )
        missing_duration = max(0.0, timestamp - self._missing_since)
        elapsed_since_valid = (
            0.0
            if self._last_valid_time is None
            else max(0.0, timestamp - self._last_valid_time)
        )
        soft_lost = (
            self._lost_frames >= self.lost_confirm_frames
            and missing_duration >= self.lost_confirm_s
        )
        timed_out = elapsed_since_valid >= self.lost_timeout_s
        if not soft_lost and not timed_out:
            return None, None

        trigger = "timeout" if timed_out else "missed_frames"
        self._lost_emitted = True
        detail = {
            "event": "target_lost",
            "target_id": self._locked_id,
            "class_name": self.target_class,
            "class_id": self._locked_class_id,
            "last_confidence": self._last_confidence,
            "reason": reason,
            "trigger": trigger,
            "missed_frames": self._lost_frames,
            "elapsed_s": elapsed_since_valid,
            "timestamp": timestamp,
        }
        return "target_lost", detail


def _is_trackable_id(object_id: str) -> bool:
    return object_id.startswith("track_") or object_id.startswith("sim_")


def _classify_absence(
    detections: Sequence[Detection2D],
    target_class: str,
    maintain_confidence: float,
    locked_id: str | None,
) -> str:
    locked = next((item for item in detections if item.object_id == locked_id), None)
    if (
        locked is not None
        and locked.class_name == target_class
        and locked.confidence < maintain_confidence
    ):
        return "low_confidence"
    if detections:
        return "locked_track_missing"
    return "no_detections"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    detector_mode = _resolve_detector_mode(args, parser)
    if detector_mode == "fake":
        if args.image or args.video:
            parser.error("fake mode does not accept --image or --video")
        return run_simulation(args)
    if args.image:
        return run_image(args)
    if args.video:
        return run_video(args)
    parser.error("yolo mode requires --image or --video")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=False)
    inputs.add_argument("--simulate", action="store_true", help="Generate fake Detection2D JSON.")
    inputs.add_argument("--image", help="Input RGB image path.")
    inputs.add_argument("--video", help="Input RGB video path.")
    parser.add_argument("--detector", choices=("fake", "yolo"), default=None)
    parser.add_argument("--model", default="assets/models/yolo11n.pt")
    parser.add_argument("--target-config", default="src/rl2_perception/config/target_classes.yaml")
    parser.add_argument("--enable-demo-mapping", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-id", default="camera_color_optical_frame")
    parser.add_argument("--timestamp", type=float, default=None)
    parser.add_argument("--sim-width", type=int, default=640)
    parser.add_argument("--sim-height", type=int, default=480)
    parser.add_argument("--sim-no-detection", action="store_true")
    parser.add_argument("--out-json", default="outputs/rl2_perception/detections.json")
    parser.add_argument("--out-image", default="outputs/rl2_perception/detection_result.jpg")
    parser.add_argument("--out-video", default="outputs/rl2_perception/detection_result.mp4")
    parser.add_argument("--out-events-json", default="")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--disable-tracking", action="store_true")
    parser.add_argument("--preprocess-mode", choices=("auto", "resize", "letterbox"), default=None)
    parser.add_argument("--output-width", type=int, default=None)
    parser.add_argument("--output-height", type=int, default=None)
    parser.add_argument("--post-mapping-nms-iou", type=float, default=None)
    return parser


def run_simulation(args: argparse.Namespace) -> int:
    # Keep the optional fake detector isolated from real YOLO/offline imports.
    # The repository does not ship the fake detector in production snapshots;
    # importing it at module load time previously broke every YOLO invocation.
    from .fake_target_detector import FakeTargetDetector

    detector = FakeTargetDetector(
        image_width=args.sim_width,
        image_height=args.sim_height,
        force_no_detection=args.sim_no_detection,
        frame_id=args.frame_id,
    )
    detections = detector.detect(timestamp=args.timestamp)
    _write_json(Path(args.out_json), detections_to_jsonable(detections))
    print(f"saved {len(detections)} simulated detections to {args.out_json}")
    return 0


def run_image(args: argparse.Namespace) -> int:
    cv2 = _import_cv2()
    from .visualization import draw_detections
    from .yolo_detector_adapter import preprocess_image

    source = cv2.imread(args.image)
    if source is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    config = load_target_config(args.target_config)
    image, metadata = _preprocess(source, args, config)
    detector = _build_yolo_detector(args, config)
    timestamp = _timestamp_or_now(args.timestamp)
    detections = detector.detect_image(image, timestamp=timestamp)

    out_image = Path(args.out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), draw_detections(image, detections))
    _write_json(Path(args.out_json), detections_to_jsonable(detections))
    print(f"preprocessed {metadata['source_width']}x{metadata['source_height']} -> 640x480 ({metadata['mode']})")
    print(f"saved {len(detections)} detections; post-mapping NMS removed {detector.last_nms_removed}")
    return 0


def run_video(args: argparse.Namespace) -> int:
    cv2 = _import_cv2()
    from .visualization import draw_detections

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    config = load_target_config(args.target_config)
    preprocess_config = config["preprocess"]
    width = args.output_width or int(preprocess_config["output_width"])
    height = args.output_height or int(preprocess_config["output_height"])
    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    detector = _build_yolo_detector(args, config)
    detector.reset_tracker()
    policy = TargetEventPolicy(**config["event_policy"])
    offline_state = "SEARCH_TARGET"
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    frame_index = 0
    tracker_frames = 0
    untracked_count = 0
    nms_removed = 0
    stride = max(args.frame_stride, 1)
    try:
        while frame_index < args.max_frames:
            ok, source_frame = cap.read()
            if not ok:
                break
            frame, metadata = _preprocess(source_frame, args, config)
            timestamp = frame_index / fps
            if args.disable_tracking:
                detections = detector.detect_image(frame, timestamp=timestamp)
            else:
                detections = detector.track_frame(frame, timestamp=timestamp, tracker=args.tracker)
            tracker_frames += 1
            untracked_count += detector.last_untracked_count
            nms_removed += detector.last_nms_removed
            writer.write(draw_detections(frame, detections))

            absence_reason = _offline_absence_reason(detector, detections, config["event_policy"])
            event, detail = policy.update_with_detail(
                detections, timestamp, task_state=offline_state, absence_reason=absence_reason
            )
            if detail is not None:
                detail.update({"frame_index": frame_index, "source": args.video})
                events.append(detail)
                offline_state = "APPROACH_TARGET" if event and event.startswith("target_seen:") else "SEARCH_TARGET"

            if frame_index % stride == 0:
                records.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "preprocess": metadata,
                        "detections": detections_to_jsonable(detections),
                    }
                )
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    _write_json(Path(args.out_json), records)
    events_path = Path(args.out_events_json) if args.out_events_json else Path(args.out_json).with_name(f"{Path(args.out_json).stem}_events.json")
    _write_json(events_path, events)
    detection_count = sum(len(item["detections"]) for item in records)
    print(f"processed {frame_index} frames; tracker processed {tracker_frames}; saved {len(records)} JSON frames")
    print(f"saved {detection_count} tracked detections; untracked={untracked_count}; NMS removed={nms_removed}")
    print(f"saved {len(events)} events to {events_path}")
    return 0


def _offline_absence_reason(
    detector: Any, detections: Sequence[Detection2D], event_config: dict[str, Any]
) -> str | None:
    if detector.last_raw_detection_count == 0:
        return "no_detections"
    if detector.last_project_detection_count == 0:
        return "target_class_missing"
    target = [item for item in detections if item.class_name == event_config["target_class"]]
    if target and all(
        item.confidence < float(event_config["maintain_confidence"]) for item in target
    ):
        return "low_confidence"
    return None


def _preprocess(image: Any, args: argparse.Namespace, config: dict[str, Any]):
    from .yolo_detector_adapter import preprocess_image

    values = config["preprocess"]
    return preprocess_image(
        image,
        output_width=args.output_width or int(values["output_width"]),
        output_height=args.output_height or int(values["output_height"]),
        mode=args.preprocess_mode or str(values["mode"]),
        letterbox_value=int(values["letterbox_value"]),
    )


def _build_yolo_detector(args: argparse.Namespace, config: dict[str, Any] | None = None):
    from .yolo_detector_adapter import YoloDetectorAdapter

    config = config or load_target_config(args.target_config)
    allowed = set(args.classes) if args.classes else set(config["target_classes"])
    class_map = config["demo_class_map"] if args.enable_demo_mapping else {}
    detection_config = config["detection"]
    nms_iou = args.post_mapping_nms_iou
    if nms_iou is None:
        nms_iou = float(detection_config["post_mapping_nms_iou"])
    return YoloDetectorAdapter(
        model_path=args.model,
        confidence=args.confidence,
        class_map=class_map,
        project_class_ids=config["class_name_to_id"],
        allowed_classes=allowed,
        frame_id=args.frame_id,
        device=args.device,
        post_mapping_nms_enabled=bool(detection_config["post_mapping_nms_enabled"]),
        post_mapping_nms_iou=nms_iou,
        agnostic_nms=args.enable_demo_mapping,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _resolve_detector_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.simulate:
        if args.detector == "yolo":
            parser.error("--simulate conflicts with --detector yolo")
        return "fake"
    if args.detector is not None:
        return args.detector
    return "yolo" if args.image or args.video else "fake"


def _timestamp_or_now(timestamp: float | None) -> float:
    return time.time() if timestamp is None else timestamp


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Image/video detection requires opencv-python.") from exc
    return cv2


def ros_main(args: list[str] | None = None) -> int:
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as exc:
        raise ImportError("The ROS node requires ROS2 Humble rclpy and std_msgs.") from exc
    node_class = _make_ros_node_class(Node, String)
    rclpy.init(args=args)
    node = None
    try:
        node = node_class()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def _make_ros_node_class(node_base: Any, string_message: Any):
    class TargetDetectorRosNode(node_base):
        def __init__(self) -> None:
            from .fake_target_detector import FakeTargetDetector

            super().__init__("target_detector_node")
            self.declare_parameter("target_config", str(_default_ros_config_path()))
            config = load_target_config(self.get_parameter("target_config").value)
            runtime = config["runtime"]
            event_config = config["event_policy"]
            self._declare_runtime_parameters(runtime, event_config)
            if str(self.get_parameter("detector_mode").value) != "fake":
                raise ValueError("The Week 2 ROS publisher supports detector_mode=fake only")
            publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
            if publish_rate_hz <= 0.0:
                raise ValueError("publish_rate_hz must be greater than zero")

            self._task_state: str | None = None
            self._detector = FakeTargetDetector(frame_id=str(self.get_parameter("frame_id").value))
            self._event_policy = TargetEventPolicy(
                target_class=str(self.get_parameter("target_class").value),
                acquire_confidence=float(self.get_parameter("acquire_confidence").value),
                maintain_confidence=float(self.get_parameter("maintain_confidence").value),
                seen_confirm_frames=int(self.get_parameter("seen_confirm_frames").value),
                seen_confirm_s=float(self.get_parameter("seen_confirm_s").value),
                lost_confirm_frames=int(self.get_parameter("lost_confirm_frames").value),
                lost_confirm_s=float(self.get_parameter("lost_confirm_s").value),
                lost_timeout_s=float(self.get_parameter("lost_timeout_s").value),
            )
            self._candidate_publisher = self.create_publisher(
                string_message, str(self.get_parameter("target_candidates_topic").value), 10
            )
            self._event_publisher = self.create_publisher(
                string_message, str(self.get_parameter("task_event_topic").value), 10
            )
            self._detail_publisher = self.create_publisher(
                string_message, str(self.get_parameter("perception_event_detail_topic").value), 10
            )
            self._state_subscription = self.create_subscription(
                string_message,
                str(self.get_parameter("task_state_topic").value),
                self._on_task_state,
                10,
            )
            self._timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        def _declare_runtime_parameters(self, runtime: dict[str, Any], event: dict[str, Any]) -> None:
            for name in (
                "detector_mode", "publish_rate_hz", "frame_id", "target_candidates_topic",
                "task_event_topic", "perception_event_detail_topic", "task_state_topic",
                "fake_no_detection",
            ):
                self.declare_parameter(name, runtime[name])
            for name in (
                "target_class", "acquire_confidence", "maintain_confidence",
                "seen_confirm_frames", "seen_confirm_s", "lost_confirm_frames",
                "lost_confirm_s", "lost_timeout_s",
            ):
                self.declare_parameter(name, event[name])

        def _on_task_state(self, message: Any) -> None:
            self._task_state = str(message.data)
            self.get_logger().info(f"task state changed to {self._task_state}")

        def _on_timer(self) -> None:
            now = self.get_clock().now()
            sec, nanosec = now.seconds_nanoseconds()
            timestamp = sec + nanosec / 1_000_000_000.0
            no_detection = bool(self.get_parameter("fake_no_detection").value)
            self._detector.force_no_detection = no_detection
            detections = self._detector.detect(timestamp=timestamp)
            message = string_message()
            message.data = json.dumps(detections_to_jsonable(detections), separators=(",", ":"), allow_nan=False)
            self._candidate_publisher.publish(message)

            event, detail = self._event_policy.update_with_detail(
                detections,
                timestamp,
                task_state=self._task_state,
                absence_reason="simulated_no_detection" if no_detection else None,
            )
            if event is None or detail is None:
                return
            event_message = string_message()
            event_message.data = event
            self._event_publisher.publish(event_message)
            detail_message = string_message()
            detail_message.data = json.dumps(detail, separators=(",", ":"), allow_nan=False)
            self._detail_publisher.publish(detail_message)
            self.get_logger().info(f"perception event {detail_message.data}")

    return TargetDetectorRosNode


def _default_ros_config_path() -> Path:
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    except ImportError:
        return _source_config_path()
    try:
        return Path(get_package_share_directory("rl2_perception")) / "config" / "target_classes.yaml"
    except PackageNotFoundError:
        return _source_config_path()


def _source_config_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "config" / "target_classes.yaml"
    if path.is_file():
        return path
    raise FileNotFoundError("Could not locate rl2_perception/config/target_classes.yaml")


if __name__ == "__main__":
    raise SystemExit(main())
