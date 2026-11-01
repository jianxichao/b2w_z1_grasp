"""Gemini 2 与 YOLO 目标话题节点，不控制机械臂。"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from robot_arm.app import build_core
from robot_arm.config import load_config
from robot_arm.manipulation import Pose, Target
from robot_arm.manipulation.safety import SafetyError
from robot_arm.vision import OrbbecCamera, TargetProvider, YoloTracker, load_eye_in_hand, stable_target
from robot_arm.vision.target_provider import TrackedTarget


def _transform_matrix(transform: Any) -> np.ndarray:
    """ROS Transform 转齐次矩阵。"""
    translation, rotation = transform.translation, transform.rotation
    quaternion = np.asarray([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("TF 四元数无效")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = (translation.x, translation.y, translation.z)
    return matrix


def _status_payload(stage: str, target: TrackedTarget | None = None, reason: str = "") -> str:
    return json.dumps(
        {
            "stage": stage,
            "target_id": None if target is None else f"{target.class_name}#{target.track_id}",
            "class_name": None if target is None else target.class_name,
            "confidence": None if target is None else target.confidence,
            "camera_xyz_m": None if target is None else target.camera_xyz.tolist(),
            "arm_base_xyz_m": None if target is None else target.base_xyz.tolist(),
            "reason": reason,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class VisionTargetNode(Node):
    """发布当前候选和稳定目标；抓取必须由 Action 客户端另行触发。"""

    def __init__(self) -> None:
        super().__init__("vision_target_node")
        self._declare_parameters()
        root = str(self.get_parameter("robot_config_root").value).strip()
        self.config = load_config(Path(root) if root else None)
        self._mode = str(self.get_parameter("mode").value)
        if self._mode not in {"dry-run", "real"}:
            raise ValueError("mode 只能是 dry-run 或 real")
        _, self._safety = build_core(self.config)
        self._arm_base_frame = str(self.get_parameter("arm_base_frame").value)
        self._tcp_frame = str(self.get_parameter("tcp_frame").value)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._candidate_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("candidate_topic").value), qos
        )
        self._target_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_topic").value), qos
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), qos
        )
        self._image_pub = self.create_publisher(
            CompressedImage,
            str(self.get_parameter("annotated_image_topic").value),
            1,
        )
        self._raw_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("annotated_raw_image_topic").value),
            1,
        )
        self.provider = self._build_provider()
        policy = self.config.vision["event_policy"]
        self._stable_frames = int(policy["seen_confirm_frames"])
        self._max_spread_m = float(policy["max_position_spread_m"])
        self._history: deque[list[TrackedTarget]] = deque(maxlen=self._stable_frames)
        self._arm_busy: bool | None = None
        self.create_subscription(
            String,
            str(self.get_parameter("arm_status_topic").value),
            self._on_arm_status,
            qos,
        )
        rate = max(0.5, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self._observe)
        self._publish_status("STARTING", reason="等待 Gemini 2 图像和机械臂 TF")

    def _declare_parameters(self) -> None:
        self.declare_parameter("robot_config_root", "")
        self.declare_parameter("mode", "dry-run")
        self.declare_parameter("candidate_topic", "/robot_arm/target_candidate")
        self.declare_parameter("target_topic", "/robot_arm/target_pose")
        self.declare_parameter("status_topic", "/robot_arm/target_status")
        self.declare_parameter(
            "annotated_image_topic", "/robot_arm/annotated/compressed"
        )
        self.declare_parameter(
            "annotated_raw_image_topic", "/robot_arm/annotated_image"
        )
        self.declare_parameter("arm_status_topic", "/arm/status")
        self.declare_parameter("arm_base_frame", "arm_base")
        self.declare_parameter("tcp_frame", "arm_tcp")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("tf_timeout_s", 0.2)
        self.declare_parameter("demo_mapping", True)
        self.declare_parameter("show", True)
        self.declare_parameter("show_depth", True)

    def _build_provider(self) -> TargetProvider:
        camera_cfg = self.config.camera["camera"]
        camera = OrbbecCamera(
            width=int(camera_cfg["width"]),
            height=int(camera_cfg["height"]),
            fps=int(camera_cfg["fps"]),
            depth_width=int(camera_cfg.get("depth_width", camera_cfg["width"])),
            depth_height=int(camera_cfg.get("depth_height", 800)),
            min_depth_m=float(camera_cfg["min_depth_m"]),
            max_depth_m=float(camera_cfg["max_depth_m"]),
            calibrated_camera_matrix=np.asarray(self.config.calibration["camera_matrix"]),
            calibration_width=int(camera_cfg.get("calibration_width", camera_cfg["width"])),
            calibration_height=int(camera_cfg.get("calibration_height", camera_cfg["height"])),
        )
        vision, model = self.config.vision, self.config.vision["model"]
        mapping = vision["demo_mapping"]
        use_mapping = bool(self.get_parameter("demo_mapping").value) and bool(mapping["enabled"])
        weights = str(model["weights"])
        local_weights = self.config.resolve(weights)
        tracker = YoloTracker(
            weights=str(local_weights) if local_weights.is_file() else weights,
            confidence=float(model["confidence"]),
            device=str(model["device"]),
            tracker=str(model["tracker"]),
            image_size=int(model.get("image_size", 416)),
            class_map=dict(mapping["class_map"]) if use_mapping else {},
            target_classes=set(vision["target_classes"]),
        )
        policy = vision["event_policy"]
        return TargetProvider(
            camera,
            tracker,
            load_eye_in_hand(self.config.resolve("config/eye_in_hand_calibration.json")),
            self._base_T_tcp,
            target_class=str(policy["target_class"]),
            acquire_confidence=float(policy["acquire_confidence"]),
            stable_frames=int(policy["seen_confirm_frames"]),
            max_spread_m=float(policy["max_position_spread_m"]),
            lock_timeout_s=float(policy["lock_timeout_s"]),
            show=bool(self.get_parameter("show").value),
            show_depth=bool(self.get_parameter("show_depth").value),
            distortion=self.config.calibration["distortion_coefficients"],
            continuous_tracking=False,
            frame_callback=self._publish_annotated_image,
        )

    def _publish_annotated_image(self, image: np.ndarray) -> None:
        try:
            import cv2
        except ImportError:
            return
        stamp = self.get_clock().now().to_msg()
        raw = Image()
        raw.header.stamp = stamp
        raw.header.frame_id = "camera_color_optical_frame"
        raw.height, raw.width = (int(value) for value in image.shape[:2])
        raw.encoding = "bgr8"
        raw.is_bigendian = 0
        raw.step = raw.width * 3
        raw.data = np.ascontiguousarray(image).tobytes()
        self._raw_image_pub.publish(raw)

        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            return
        message = CompressedImage()
        message.header.stamp = stamp
        message.header.frame_id = "camera_color_optical_frame"
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self._image_pub.publish(message)

    def _base_T_tcp(self) -> np.ndarray:
        transform = self._tf_buffer.lookup_transform(
            self._arm_base_frame,
            self._tcp_frame,
            Time(),
            timeout=Duration(seconds=float(self.get_parameter("tf_timeout_s").value)),
        )
        return _transform_matrix(transform.transform)

    def _observe(self) -> None:
        try:
            observations = self.provider.observe()
            if self._arm_busy is not False:
                self._history.clear()
                self._publish_status("WAITING_ARM", reason="等待机械臂空闲")
                return
            self._history.append(observations)
            candidate = max(observations, key=lambda item: item.confidence, default=None)
            if candidate is None:
                self._publish_status("SEARCHING", reason="当前帧没有满足条件的 RGB-D 候选")
                return
            self._candidate_pub.publish(self._pose(candidate))
            stable = stable_target(self._history, self._stable_frames, self._max_spread_m)
            if stable is None:
                self._publish_status("TRACKING", candidate, "等待目标连续稳定")
                return
            target = Target(
                f"{stable.class_name}#{stable.track_id}",
                Pose.from_xyz(stable.base_xyz, self._arm_base_frame),
                stable.confidence,
                stable.timestamp,
            )
            try:
                self._safety.validate_target(target)
            except SafetyError as exc:
                self._publish_status("REJECTED", stable, str(exc))
                return
            # Publish a heartbeat while the target remains stable.  The
            # navigation supervisor deliberately accepts only a fresh pose
            # acquired after the base has stopped at the target; publishing
            # only once per track could leave it waiting on a stale latched
            # sample forever.
            self._target_pub.publish(self._pose(stable))
            self._publish_status("LOCKED", stable)
        except TransformException as exc:
            self._publish_status("WAITING_TF", reason=str(exc))
        except (ImportError, RuntimeError, TimeoutError, ValueError) as exc:
            # A transient USB2/frame-sync failure otherwise leaves the Orbbec
            # pipeline permanently wedged. Close it here; the next timer tick
            # performs a clean lazy restart.
            if isinstance(exc, (RuntimeError, TimeoutError)):
                self.provider.camera.close()
                self._history.clear()
            self._publish_status("ERROR", reason=str(exc))
            self.get_logger().error(str(exc), throttle_duration_sec=2.0)

    def _on_arm_status(self, message: String) -> None:
        try:
            self._arm_busy = bool(json.loads(message.data)["busy"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._arm_busy = None

    def _pose(self, target: TrackedTarget) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._arm_base_frame
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            float(value) for value in target.base_xyz
        )
        message.pose.orientation.w = 1.0
        return message

    def _publish_status(
        self, stage: str, target: TrackedTarget | None = None, reason: str = ""
    ) -> None:
        message = String()
        message.data = _status_payload(stage, target, reason)
        self._status_pub.publish(message)

    def close(self) -> None:
        self.provider.close()


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node: VisionTargetNode | None = None
    try:
        node = VisionTargetNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
