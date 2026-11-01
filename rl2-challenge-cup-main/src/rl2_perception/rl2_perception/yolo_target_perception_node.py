"""Online YOLO target perception node using ONNX Runtime inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from rl2_perception.onnx_yolo import decode_yolo_output, load_class_names, preprocess_rgb


_CAMERA_OPTICAL_TO_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _depth_roi_bounds(
    u: float,
    v: float,
    box_w: float,
    box_h: float,
    roi_scale: float,
    color_size: tuple[int, int],
    depth_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Map a color-image detection ROI into an aligned depth image.

    RealSense may publish aligned depth at the depth profile resolution rather
    than the color profile resolution. The streams still cover the same
    aligned image plane, so scale color pixel coordinates before sampling.
    """
    color_width, color_height = color_size
    depth_width, depth_height = depth_size
    if min(color_width, color_height, depth_width, depth_height) <= 0:
        return None

    scale_x = depth_width / float(color_width)
    scale_y = depth_height / float(color_height)
    depth_u = u * scale_x
    depth_v = v * scale_y
    half_w = max(1.0, box_w * max(0.05, roi_scale) * scale_x / 2.0)
    half_h = max(1.0, box_h * max(0.05, roi_scale) * scale_y / 2.0)
    x1 = max(0, int(math.floor(depth_u - half_w)))
    x2 = min(depth_width - 1, int(math.ceil(depth_u + half_w)))
    y1 = max(0, int(math.floor(depth_v - half_h)))
    y2 = min(depth_height - 1, int(math.ceil(depth_v + half_h)))
    if x1 > x2 or y1 > y2:
        return None
    return x1, x2, y1, y2


class YoloTargetPerceptionNode(Node):
    """Detect targets with YOLO and estimate pose from RealSense depth data."""

    def __init__(self) -> None:
        super().__init__("yolo_target_perception_node")

        # Parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("candidate_confidence_threshold", 0.30)
        self.declare_parameter("max_confirm_distance_m", 3.0)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("class_names_path", "")
        self.declare_parameter("class_mapping", "")
        self.declare_parameter("target_class", "science_target")
        self.declare_parameter("seen_confirm_frames", 3)
        self.declare_parameter("min_distance_m", 0.15)
        self.declare_parameter("max_distance_m", 6.0)
        self.declare_parameter("pose_smoothing_alpha", 0.35)
        # One smoothing track is shared by detections of the configured
        # class.  When the camera switches directly from one physical target
        # to another there may be no seven-frame detection gap to clear that
        # track.  Reset on a large map-space jump so exponential smoothing
        # cannot create a high-confidence "ghost" coordinate between two
        # targets.
        self.declare_parameter("pose_smoothing_reset_distance_m", 0.80)
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("color_topic", "/d435i/color/image_raw")
        self.declare_parameter("camera_info_topic", "/d435i/color/camera_info")
        self.declare_parameter("odom_topic", "/slam/odom")
        self.declare_parameter("pose_topic", "/perception/target_pose")
        self.declare_parameter("candidate_pose_topic", "/perception/target_candidate_pose")
        self.declare_parameter("status_topic", "/target_detection_status")
        self.declare_parameter("annotated_image_topic", "/perception/annotated_image")
        self.declare_parameter("task_event_topic", "/task_event")
        self.declare_parameter("distance_source", "depth")
        self.declare_parameter("depth_topic", "/d435i/aligned_depth_to_color/image_raw")
        self.declare_parameter("depth_units", 0.001)
        self.declare_parameter("depth_roi_scale", 0.5)
        self.declare_parameter("inference_rate_hz", 5.0)
        self.declare_parameter("input_timeout_sec", 1.0)
        self.declare_parameter("rgb_depth_sync_tolerance_sec", 0.20)
        self.declare_parameter("rgb_depth_sync_use_receive_time", False)
        self.declare_parameter("camera_to_base_x", 0.15)
        self.declare_parameter("camera_to_base_y", 0.0)
        self.declare_parameter("camera_to_base_z", 0.42)
        self.declare_parameter("camera_roll_offset_deg", 0.0)
        self.declare_parameter("camera_pitch_offset_deg", 0.0)
        self.declare_parameter("camera_yaw_offset_deg", 0.0)

        # Load YOLO model
        model_path_str = str(self.get_parameter("model_path").value)
        if not model_path_str:
            raise ValueError("model_path parameter is required")

        model_path = Path(model_path_str)
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        # Parse class mapping (e.g., "rock_obstacle:science_target")
        mapping_str = str(self.get_parameter("class_mapping").value)
        self.class_map = {}
        if mapping_str:
            for pair in mapping_str.split(","):
                if ":" in pair:
                    src, dst = pair.split(":", 1)
                    self.class_map[src.strip()] = dst.strip()

        self.target_class = str(self.get_parameter("target_class").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.candidate_confidence_threshold = float(
            self.get_parameter("candidate_confidence_threshold").value
        )
        self.max_confirm_distance_m = float(
            self.get_parameter("max_confirm_distance_m").value
        )
        self.seen_confirm_frames = max(1, int(self.get_parameter("seen_confirm_frames").value))
        self.min_distance_m = float(self.get_parameter("min_distance_m").value)
        self.max_distance_m = float(self.get_parameter("max_distance_m").value)
        self.pose_smoothing_alpha = min(
            1.0,
            max(0.0, float(self.get_parameter("pose_smoothing_alpha").value)),
        )
        self.pose_smoothing_reset_distance_m = max(
            0.0,
            float(
                self.get_parameter("pose_smoothing_reset_distance_m").value
            ),
        )
        self.target_frame = str(self.get_parameter("target_frame").value)

        # Initialize the ONNX Runtime session; no PyTorch is needed at inference.
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                "yolo_target_perception_node now requires an ONNX model "
                f"(.onnx), got: {model_path}"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX inference requires onnxruntime. "
                "Install with: pip install onnxruntime"
            ) from exc

        device = str(self.get_parameter("device").value)
        if device != "cpu":
            self.get_logger().warning(
                "ONNX Runtime uses CPUExecutionProvider; device=%s is ignored.", device
            )
        self.ort_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.ort_input_name = self.ort_session.get_inputs()[0].name
        self.imgsz = int(self.get_parameter("imgsz").value)

        names_path = str(self.get_parameter("class_names_path").value)
        if not names_path:
            names_path = str(model_path.with_suffix(".names.json"))
        self.class_names = load_class_names(names_path)
        if not self.class_names:
            raise ValueError("class names file is empty: %s" % names_path)

        # State variables
        self.rgb_msg: Image | None = None
        self.camera_info: CameraInfo | None = None
        self.odom: Odometry | None = None
        self.depth_msg: Image | None = None
        self.rgb_received_ns = 0
        self.camera_info_received_ns = 0
        self.odom_received_ns = 0
        self.depth_received_ns = 0
        self.last_processed_rgb_stamp: tuple[int, int] | None = None
        self.seen_streak = 0
        self.lost_streak = 0
        self.target_seen_emitted = False
        self.smoothed_target_xy: tuple[float, float] | None = None

        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("pose_topic").value),
            10,
        )
        self.candidate_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("candidate_pose_topic").value),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.event_pub = self.create_publisher(
            String,
            str(self.get_parameter("task_event_topic").value),
            10,
        )
        self.annotated_pub = self.create_publisher(
            Image,
            str(self.get_parameter("annotated_image_topic").value),
            2,
        )

        # Subscribers
        self.create_subscription(
            Image,
            str(self.get_parameter("color_topic").value),
            self._on_rgb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            qos_profile_sensor_data,
        )

        # Timer
        rate = max(0.2, float(self.get_parameter("inference_rate_hz").value))
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"YOLO target perception ready: model={model_path}, "
            f"class_map={self.class_map}, target={self.target_class}"
        )

    def _on_rgb(self, msg: Image) -> None:
        self.rgb_msg = msg
        self.rgb_received_ns = self.get_clock().now().nanoseconds

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.camera_info_received_ns = self.get_clock().now().nanoseconds

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = msg
        self.odom_received_ns = self.get_clock().now().nanoseconds

    def _on_depth(self, msg: Image) -> None:
        self.depth_msg = msg
        self.depth_received_ns = self.get_clock().now().nanoseconds

    def _tick(self) -> None:
        distance_source = str(self.get_parameter("distance_source").value).lower()
        now_ns = self.get_clock().now().nanoseconds
        timeout_ns = int(float(self.get_parameter("input_timeout_sec").value) * 1e9)
        inputs = {
            "rgb": self._fresh(now_ns, self.rgb_received_ns, timeout_ns),
            "camera_info": self._fresh(now_ns, self.camera_info_received_ns, timeout_ns),
            "odom": self._fresh(now_ns, self.odom_received_ns, timeout_ns),
            "depth": self._fresh(now_ns, self.depth_received_ns, timeout_ns),
        }
        if distance_source != "depth":
            self._publish_status(False, "invalid_distance_source", [], inputs)
            return

        distance_available = inputs["depth"]

        # Detection and the annotated image only require camera inputs.  Odom
        # is required later for the map-frame PoseStamped, but a stopped SLAM
        # pipeline must not make the operator's camera view go blank.
        if not (inputs["rgb"] and inputs["camera_info"] and distance_available):
            self._publish_status(False, "waiting_for_inputs", [], inputs)
            return

        assert self.rgb_msg is not None and self.depth_msg is not None
        rgb_stamp = (int(self.rgb_msg.header.stamp.sec), int(self.rgb_msg.header.stamp.nanosec))
        if rgb_stamp == self.last_processed_rgb_stamp:
            return
        self.last_processed_rgb_stamp = rgb_stamp
        sync_tolerance_sec = float(
            self.get_parameter("rgb_depth_sync_tolerance_sec").value
        )
        if bool(self.get_parameter("rgb_depth_sync_use_receive_time").value):
            # Some RealSense drivers expose color and aligned-depth images
            # from independent device clocks.  Their ROS header stamps can
            # differ by seconds even though the messages arrive together and
            # the depth image is already aligned to color.  Pair those frames
            # by local receipt time, while retaining the same bounded window.
            rgb_depth_synchronized = abs(
                self.rgb_received_ns - self.depth_received_ns
            ) <= int(sync_tolerance_sec * 1e9)
        else:
            rgb_depth_synchronized = _stamps_within(
                self.rgb_msg,
                self.depth_msg,
                sync_tolerance_sec,
            )
        if not rgb_depth_synchronized:
            self._publish_status(False, "rgb_depth_not_synchronized", [], inputs)
            return

        result = self._detect_yolo_target()
        if result is None:
            self.lost_streak += 1
            if self.lost_streak > 7:
                self.seen_streak = 0
                self.smoothed_target_xy = None
            self._publish_status(False, "no_target_detection", [], inputs)
            self._publish_annotated(None, None)
            return

        u, v, confidence, box_w, box_h = result
        self._publish_annotated(result, None)
        depth = self._distance_from_depth(u, v, box_w, box_h)
        if depth is None:
            self.lost_streak += 1
            if self.lost_streak > 7:
                self.smoothed_target_xy = None
            self._publish_status(
                False,
                "depth_distance_unavailable",
                [f"yolo_conf={confidence:.2f}"],
                inputs,
                candidate_conf=confidence,
            )
            return

        if not math.isfinite(depth) or depth < self.min_distance_m or depth > self.max_distance_m:
            self.lost_streak += 1
            if self.lost_streak > 7:
                self.smoothed_target_xy = None
            self._publish_status(
                False,
                "pose_estimation_invalid_distance",
                [],
                inputs,
                candidate_conf=confidence,
            )
            return

        self.lost_streak = 0

        candidate_pose = None
        if inputs["odom"]:
            candidate_pose = self._estimate_pose_from_depth(u, v, depth)

        confirmed_candidate = (
            confidence >= self.confidence_threshold
            and depth <= self.max_confirm_distance_m
        )
        if not confirmed_candidate:
            # Suspected target below the confirm threshold: publish it as a
            # candidate so the supervisor can pause and stabilize the robot.
            self.seen_streak = 0
            if candidate_pose is not None:
                self.candidate_pub.publish(candidate_pose)
            self._publish_status(
                False,
                "target_candidate:conf=%.2f,dist=%.2f" % (confidence, depth),
                [f"yolo_conf={confidence:.2f}", f"depth_m={depth:.3f}"],
                inputs,
                candidate_x=(
                    float(candidate_pose.pose.position.x)
                    if candidate_pose is not None
                    else None
                ),
                candidate_y=(
                    float(candidate_pose.pose.position.y)
                    if candidate_pose is not None
                    else None
                ),
                candidate_conf=confidence,
                confirmed=False,
            )
            return

        self.seen_streak += 1
        if self.seen_streak < self.seen_confirm_frames:
            self._publish_status(
                False,
                "target_candidate_unconfirmed",
                [f"yolo_conf={confidence:.2f}"],
                inputs,
                candidate_x=(
                    float(candidate_pose.pose.position.x)
                    if candidate_pose is not None
                    else None
                ),
                candidate_y=(
                    float(candidate_pose.pose.position.y)
                    if candidate_pose is not None
                    else None
                ),
                candidate_conf=confidence,
                confirmed=False,
            )
            return

        if not inputs["odom"]:
            self._publish_annotated(result, depth)
            self._publish_status(
                True,
                "target_detected_waiting_for_odom",
                [
                    f"class={self.target_class}",
                    f"yolo_conf={confidence:.2f}",
                    f"depth_m={depth:.3f}",
                ],
                inputs,
                candidate_x=None,
                candidate_y=None,
                candidate_conf=confidence,
                confirmed=True,
            )
            return

        pose = candidate_pose if candidate_pose is not None else self._estimate_pose_from_depth(u, v, depth)
        pose = self._smooth_pose(pose)
        self.pose_pub.publish(pose)
        self._publish_annotated(result, depth)
        self.get_logger().warn(
            "TARGET CONFIRMED: class=%s conf=%.2f depth=%.2fm map=(%.2f, %.2f)"
            % (
                self.target_class,
                confidence,
                depth,
                pose.pose.position.x,
                pose.pose.position.y,
            )
        )

        if not self.target_seen_emitted:
            self.event_pub.publish(String(data=f"target_seen:{self.target_class}"))
            self.target_seen_emitted = True

        self._publish_status(
            True,
            "target_pose_published",
            [
                f"class={self.target_class}",
                f"yolo_conf={confidence:.2f}",
                f"depth_m={depth:.3f}",
                f"map_x={pose.pose.position.x:.3f}",
                f"map_y={pose.pose.position.y:.3f}",
            ],
            inputs,
            candidate_x=float(pose.pose.position.x),
            candidate_y=float(pose.pose.position.y),
            candidate_conf=confidence,
            confirmed=True,
        )

    @staticmethod
    def _fresh(now_ns: int, received_ns: int, timeout_ns: int) -> bool:
        return received_ns > 0 and now_ns - received_ns <= timeout_ns

    def _publish_annotated(
        self,
        detection: tuple[float, float, float, float, float] | None,
        depth_m: float | None,
    ) -> None:
        if self.rgb_msg is None:
            return
        rgb = _image_to_array(self.rgb_msg)
        if rgb is None:
            return
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if detection is not None:
            u, v, confidence, box_w, box_h = detection
            x1 = max(0, int(round(u - box_w / 2.0)))
            y1 = max(0, int(round(v - box_h / 2.0)))
            x2 = min(bgr.shape[1] - 1, int(round(u + box_w / 2.0)))
            y2 = min(bgr.shape[0] - 1, int(round(v + box_h / 2.0)))
            import cv2

            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.target_class} {confidence:.2f}"
            if depth_m is not None:
                label += f" {depth_m:.2f}m"
            cv2.putText(
                bgr,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        msg = Image()
        msg.header = self.rgb_msg.header
        msg.height = int(bgr.shape[0])
        msg.width = int(bgr.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(bgr.shape[1] * 3)
        msg.data = bgr.tobytes()
        self.annotated_pub.publish(msg)

    def _detect_yolo_target(self) -> tuple[float, float, float, float, float] | None:
        """Run ONNX YOLO detection and return (u, v, confidence, box_w, box_h)."""
        if self.rgb_msg is None:
            return None

        rgb = _image_to_array(self.rgb_msg)
        if rgb is None:
            return None

        blob, scale, pad_x, pad_y = preprocess_rgb(rgb, self.imgsz)
        outputs = self.ort_session.run(None, {self.ort_input_name: blob})
        detections = decode_yolo_output(
            outputs[0],
            class_names=self.class_names,
            confidence_threshold=self.candidate_confidence_threshold,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            image_size=(rgb.shape[1], rgb.shape[0]),
        )

        best: tuple[float, float, float, float, float] | None = None
        for class_name, confidence, u, v, box_w, box_h in detections:
            mapped_class = self.class_map.get(class_name, class_name)
            if mapped_class != self.target_class:
                continue
            if best is None or confidence > best[2]:
                best = (u, v, confidence, box_w, box_h)

        if best is None:
            return None
        return best

    def _depth_to_meters(self, msg: Image) -> np.ndarray | None:
        """Convert a RealSense depth image to meters as a 2D float array."""
        depth = _depth_image_to_array(msg)
        if depth is None:
            return None
        if msg.encoding == "16UC1":
            return depth * float(self.get_parameter("depth_units").value)
        # ROS 32FC1 depth images are already expressed in metres.
        return depth

    def _distance_from_depth(
        self,
        u: float,
        v: float,
        box_w: float,
        box_h: float,
    ) -> float | None:
        """Return the median depth (m) inside the central ROI of a detection box."""
        if self.camera_info is None or self.depth_msg is None:
            return None

        depth = self._depth_to_meters(self.depth_msg)
        if depth is None:
            return None

        height, width = depth.shape
        roi_scale = max(0.05, float(self.get_parameter("depth_roi_scale").value))
        bounds = _depth_roi_bounds(
            u,
            v,
            box_w,
            box_h,
            roi_scale,
            (int(self.camera_info.width), int(self.camera_info.height)),
            (width, height),
        )
        if bounds is None:
            return None
        x1, x2, y1, y2 = bounds

        roi = depth[y1 : y2 + 1, x1 : x2 + 1]
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _estimate_pose_from_depth(self, u: float, v: float, depth: float) -> PoseStamped:
        """Project a depth pixel through the camera to base to map frame."""
        assert self.odom is not None
        assert self.camera_info is not None

        fx = float(self.camera_info.k[0]) if self.camera_info.k[0] else 571.0
        fy = float(self.camera_info.k[1]) if self.camera_info.k[1] else fx
        cx = float(self.camera_info.k[2]) if self.camera_info.k[2] else 319.5
        cy = float(self.camera_info.k[5]) if self.camera_info.k[5] else 239.5

        point_camera = np.array(
            [
                (u - cx) / fx * depth,
                (v - cy) / fy * depth,
                depth,
            ],
            dtype=np.float64,
        )

        roll = math.radians(float(self.get_parameter("camera_roll_offset_deg").value))
        pitch = math.radians(float(self.get_parameter("camera_pitch_offset_deg").value))
        yaw = math.radians(float(self.get_parameter("camera_yaw_offset_deg").value))
        rotation = _euler_rotation_matrix(yaw, pitch, roll) @ _CAMERA_OPTICAL_TO_BASE
        translation = np.array(
            [
                float(self.get_parameter("camera_to_base_x").value),
                float(self.get_parameter("camera_to_base_y").value),
                float(self.get_parameter("camera_to_base_z").value),
            ],
            dtype=np.float64,
        )
        point_base = rotation @ point_camera + translation

        robot_x = float(self.odom.pose.pose.position.x)
        robot_y = float(self.odom.pose.pose.position.y)
        robot_yaw = _yaw_from_quaternion(self.odom.pose.pose.orientation)
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        target_x = robot_x + cos_yaw * point_base[0] - sin_yaw * point_base[1]
        target_y = robot_y + sin_yaw * point_base[0] + cos_yaw * point_base[1]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = target_x
        msg.pose.position.y = target_y
        msg.pose.position.z = float(self.odom.pose.pose.position.z) + float(point_base[2])
        msg.pose.orientation.w = 1.0
        return msg

    def _smooth_pose(self, msg: PoseStamped) -> PoseStamped:
        """Apply exponential smoothing to pose estimate."""
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)

        if self.smoothed_target_xy is None:
            self.smoothed_target_xy = (x, y)
            return msg

        old_x, old_y = self.smoothed_target_xy
        if (
            self.pose_smoothing_reset_distance_m > 0.0
            and math.hypot(x - old_x, y - old_y)
            > self.pose_smoothing_reset_distance_m
        ):
            self.get_logger().info(
                "Target pose smoothing reset after %.2fm map-space jump."
                % math.hypot(x - old_x, y - old_y),
                throttle_duration_sec=1.0,
            )
            self.smoothed_target_xy = (x, y)
            return msg

        alpha = self.pose_smoothing_alpha
        smooth_x = alpha * x + (1.0 - alpha) * old_x
        smooth_y = alpha * y + (1.0 - alpha) * old_y
        self.smoothed_target_xy = (smooth_x, smooth_y)

        msg.pose.position.x = smooth_x
        msg.pose.position.y = smooth_y
        return msg

    def _publish_status(
        self,
        detected: bool,
        reason: str,
        sources: list[str],
        inputs: dict[str, bool],
        candidate_x: float | None = None,
        candidate_y: float | None = None,
        candidate_conf: float | None = None,
        confirmed: bool = False,
    ) -> None:
        """Publish detection status for debugging."""
        if detected:
            self.lost_streak = 0

        payload = {
            "target_detected": detected,
            "reason": reason,
            "seen_streak": self.seen_streak,
            "lost_streak": self.lost_streak,
            "sources": sources,
            "inputs": inputs,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
            "candidate_confidence": candidate_conf,
            "confirmed": confirmed,
        }
        self.status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def _image_to_array(msg: Image) -> np.ndarray | None:
    """Convert ROS Image message to numpy array."""
    if msg.encoding not in ("rgb8", "bgr8"):
        return None

    data = np.frombuffer(msg.data, dtype=np.uint8)
    row_bytes = int(msg.width) * 3
    step = int(msg.step) if int(msg.step) > 0 else row_bytes
    expected = int(msg.height) * step
    if data.size < expected or step < row_bytes:
        return None
    rows = data[:expected].reshape((int(msg.height), step))
    array = rows[:, :row_bytes].reshape((int(msg.height), int(msg.width), 3))
    if msg.encoding == "bgr8":
        array = array[:, :, ::-1]

    return array


def _depth_image_to_array(msg: Image) -> np.ndarray | None:
    """Decode a mono depth image while respecting per-row padding."""
    if msg.encoding == "16UC1":
        dtype = np.dtype("<u2" if not msg.is_bigendian else ">u2")
    elif msg.encoding == "32FC1":
        dtype = np.dtype("<f4" if not msg.is_bigendian else ">f4")
    else:
        return None

    row_bytes = int(msg.width) * dtype.itemsize
    step = int(msg.step) if int(msg.step) > 0 else row_bytes
    expected = int(msg.height) * step
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size < expected or step < row_bytes:
        return None
    rows = raw[:expected].reshape((int(msg.height), step))
    pixels = np.ascontiguousarray(rows[:, :row_bytes])
    return pixels.view(dtype).reshape((int(msg.height), int(msg.width))).astype(
        np.float64
    )


def _stamp_seconds(msg: Image) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def _stamps_within(first: Image, second: Image, tolerance_sec: float) -> bool:
    return abs(_stamp_seconds(first) - _stamp_seconds(second)) <= max(0.0, tolerance_sec)


def _euler_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build Rz(yaw) @ Ry(pitch) @ Rx(roll) for camera-to-base rotation."""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)

    rz = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _yaw_from_quaternion(q: Any) -> float:
    """Extract yaw angle from quaternion."""
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def main() -> None:
    rclpy.init()
    node = YoloTargetPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
