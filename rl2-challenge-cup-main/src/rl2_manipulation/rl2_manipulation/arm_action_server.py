"""ARX X5 抓取 Action、关节状态和 TF 统一节点。"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster
from tf2_ros import TransformException, TransformListener

from rl2_interfaces.action import GraspTarget
from robot_arm.app import build_core
from robot_arm.config import load_config
from robot_arm.hardware import Arx5Backend
from robot_arm.manipulation import Pose, Target
from robot_arm.manipulation.state_machine import GraspStateMachine, State


PROGRESS = {
    State.CHECK: 0.00,
    State.HOME: 0.05,
    State.SEARCH_POSE: 0.10,
    State.LOCK_TARGET: 0.15,
    State.PLAN: 0.20,
    State.PREGRASP: 0.30,
    State.GRASP: 0.40,
    State.CLOSE: 0.48,
    State.VERIFY_GRASP: 0.52,
    State.LIFT: 0.60,
    State.TRANSIT: 0.70,
    State.PLACE: 0.80,
    State.OPEN: 0.88,
    State.VERIFY_PLACE: 0.92,
    State.RETREAT: 0.96,
    State.DONE: 1.00,
}


def _quaternion(rotation: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """旋转矩阵转 ROS 四元数 (x, y, z, w)。"""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("旋转矩阵无效")
    values, vectors = np.linalg.eigh(
        np.array(
            [
                [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[1, 0] + matrix[0, 1], matrix[2, 0] + matrix[0, 2], matrix[1, 2] - matrix[2, 1]],
                [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 0] - matrix[0, 2]],
                [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[0, 1] - matrix[1, 0]],
                [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0], matrix.trace()],
            ]
        )
        / 3.0
    )
    quaternion = vectors[:, int(np.argmax(values))]
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]


def _rpy_quaternion(rpy: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) * 0.5 for value in rpy)
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    )
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class _OneTarget:
    def __init__(self, target: Target) -> None:
        self.target = target

    def lock(self) -> Target:
        return self.target

    def close(self) -> None:
        pass


class ArmRuntimeNode(Node):
    """一个进程独占 CAN，同时提供 Action、JointState 和 TF。"""

    def __init__(self) -> None:
        super().__init__("arm_runtime_node")
        self._declare_parameters()
        mode = str(self.get_parameter("mode").value)
        if mode not in {"dry-run", "real"}:
            raise ValueError("mode 只能是 dry-run 或 real")

        root_value = str(self.get_parameter("robot_config_root").value).strip()
        self.config = load_config(Path(root_value) if root_value else None)
        self.planner, self.safety = build_core(self.config)
        self.arm = Arx5Backend(
            self.config.root,
            self.config.hardware,
            dry_run=mode == "dry-run",
            kinematics=self.planner.kinematics,
        )
        self._connected = False
        self._busy = False
        self._fatal_error = ""
        self._goal_lock = threading.Lock()
        self._sdk_lock = threading.RLock()
        self._active_target_id = ""
        self._last_state = State.CHECK

        try:
            self.arm.connect()
            self._connected = True
            ok, reason = self.arm.self_check()
            if not ok:
                raise RuntimeError(reason or "机械臂自检失败")
        except Exception:
            if self._connected:
                self.arm.close()
                self._connected = False
            raise

        action_name = str(self.get_parameter("action_name").value)
        callback_group = ReentrantCallbackGroup()
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), latched_qos
        )
        self._target_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_topic").value), latched_qos
        )
        self._joint_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_state_topic").value), 10
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_static_transforms()
        self._action = ActionServer(
            self,
            GraspTarget,
            action_name,
            execute_callback=self._execute,
            goal_callback=self._accept_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=callback_group,
        )
        self._home_service = self.create_service(
            Trigger,
            str(self.get_parameter("home_service").value),
            self._home,
            callback_group=callback_group,
        )
        rate = max(1.0, float(self.get_parameter("state_rate_hz").value))
        self.create_timer(1.0 / rate, self._publish_robot_state, callback_group=callback_group)
        self._publish_status("READY")
        self.get_logger().info(
            f"{action_name} 已启动，mode={mode}，CAN={self.config.hardware['can_interface']}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("mode", "dry-run")
        self.declare_parameter("robot_config_root", "")
        self.declare_parameter("action_name", "/arm/grasp_target")
        self.declare_parameter("joint_state_topic", "/arm/joint_states")
        self.declare_parameter("status_topic", "/arm/status")
        self.declare_parameter("target_topic", "/arm/target_pose")
        self.declare_parameter("home_service", "/arm/home")
        self.declare_parameter("state_rate_hz", 20.0)
        self.declare_parameter("base_parent_frame", "base_link")
        self.declare_parameter("arm_base_frame", "arm_base")
        self.declare_parameter("tcp_frame", "arm_tcp")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("mount_calibrated", False)
        self.declare_parameter("mount_xyz_m", [0.0, 0.0, 0.0])
        self.declare_parameter("mount_rpy_rad", [0.0, 0.0, 0.0])
        self.declare_parameter("tf_timeout_s", 0.2)

    def _home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Explicit operator recovery; never races with an active grasp."""
        with self._goal_lock:
            if self._busy:
                response.success = False
                response.message = "mechanical arm is busy; home rejected"
                return response
            if self._fatal_error or not self._connected:
                response.success = False
                response.message = self._fatal_error or "mechanical arm is not connected"
                return response
            self._busy = True
            self._active_target_id = "operator_home"

        final_stage, final_error = "HOMING", ""
        self._publish_status(final_stage)
        try:
            with self._sdk_lock:
                result = self.arm.home()
            response.success = bool(result.reached and not result.cancelled)
            response.message = "home reached" if response.success else (result.reason or "home not reached")
            final_stage = "READY" if response.success else "HOME_FAILED"
            final_error = "" if response.success else response.message
        except RuntimeError as exc:
            response.success = False
            response.message = str(exc)
            final_stage, final_error = "HOME_FAILED", str(exc)
        finally:
            with self._goal_lock:
                self._busy = False
            self._publish_status(final_stage, final_error)
            self._active_target_id = ""
        return response

    def _accept_goal(self, request: GraspTarget.Goal) -> GoalResponse:
        frame = request.target_pose.header.frame_id.strip()
        with self._goal_lock:
            if self._busy or self._fatal_error or not self._connected:
                return GoalResponse.REJECT
            if not request.target_id.strip() or not frame:
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle: Any) -> GraspTarget.Result:
        result = GraspTarget.Result()
        target_id = goal_handle.request.target_id.strip()
        self._active_target_id = target_id
        final_stage, final_error = "RUNNING", ""
        try:
            source_stamp = goal_handle.request.target_pose.header.stamp
            stamp_s = None
            if int(source_stamp.sec) != 0 or int(source_stamp.nanosec) != 0:
                stamp_s = float(source_stamp.sec) + float(source_stamp.nanosec) * 1e-9
            pose = self._to_arm_base(goal_handle.request.target_pose)
            self._target_pub.publish(pose)
            target = Target(
                target_id,
                Pose.from_xyz((pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)),
                confidence=1.0,
                # CLI 手动目标通常使用零时间戳，表示“现在提交的固定目标”；
                # 带有效 ROS 时间戳的视觉目标仍由 target_max_age_s 保护。
                stamp_s=stamp_s,
            )

            def on_state(state: State) -> None:
                self._last_state = state
                self._publish_status(state.value)
                feedback = GraspTarget.Feedback()
                feedback.stage = state.value
                feedback.progress = float(PROGRESS.get(state, 0.0))
                goal_handle.publish_feedback(feedback)

            machine = GraspStateMachine(
                self.arm,
                _OneTarget(target),
                self.planner,
                self.safety,
                on_state,
                max_retries=0,
                manage_resources=False,
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            )
            # ARX5 SDK 由本 Action 独占；状态定时器在运动时只读取反馈缓存。
            with self._sdk_lock:
                outcome = machine.run()
            final_stage, final_error = self._last_state.value, outcome.error
            result.success = outcome.success
            result.message = "抓取完成" if outcome.success else outcome.error
            if outcome.success:
                goal_handle.succeed()
            elif outcome.cancelled:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            if outcome.fatal:
                self._fatal_error = outcome.error
                with self._sdk_lock:
                    self.arm.close()
                self._connected = False
            return result
        except (RuntimeError, ValueError, TransformException) as exc:
            final_stage, final_error = "REJECTED", str(exc)
            result.success = False
            result.message = str(exc)
            goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._busy = False
            self._publish_status(final_stage, final_error)
            self._active_target_id = ""

    def _to_arm_base(self, source: PoseStamped) -> PoseStamped:
        target_frame = str(self.get_parameter("arm_base_frame").value)
        source_frame = source.header.frame_id.strip()
        if source_frame == target_frame:
            result = PoseStamped()
            result.header.stamp = self.get_clock().now().to_msg()
            result.header.frame_id = target_frame
            result.pose = source.pose
            return result
        transform = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=float(self.get_parameter("tf_timeout_s").value)),
        )
        result = do_transform_pose_stamped(source, transform)
        result.header.stamp = self.get_clock().now().to_msg()
        return result

    def _publish_robot_state(self) -> None:
        if not self._connected:
            return
        try:
            busy = self._busy
            if busy:
                joints = self.arm.latest_feedback_joints()
                transform = self.planner.kinematics.forward_matrix(joints)
            else:
                with self._sdk_lock:
                    transform = self.arm.base_T_eef()
                    joints = self.arm.latest_feedback_joints()
            now = self.get_clock().now().to_msg()
            message = JointState()
            message.header.stamp = now
            message.header.frame_id = str(self.get_parameter("arm_base_frame").value)
            message.name = [f"joint{index}" for index in range(1, 7)]
            message.position = list(joints)
            self._joint_pub.publish(message)

            self._tf.sendTransform(
                self._transform_message(
                    str(self.get_parameter("arm_base_frame").value),
                    str(self.get_parameter("tcp_frame").value),
                    transform[:3, 3],
                    _quaternion(transform[:3, :3]),
                    now,
                )
            )
        except RuntimeError as exc:
            self._fatal_error = str(exc)
            self._publish_status("HARDWARE_FAULT", self._fatal_error)
            if self._connected:
                with self._sdk_lock:
                    self.arm.close()
                self._connected = False

    def _publish_static_transforms(self) -> None:
        now = self.get_clock().now().to_msg()
        calibration = self.config.calibration["camera_to_eef"]
        camera = self._transform_message(
            str(self.get_parameter("tcp_frame").value),
            str(self.get_parameter("camera_frame").value),
            calibration["translation"],
            _quaternion(calibration["rotation_matrix"]),
            now,
        )
        transforms = [camera]
        if bool(self.get_parameter("mount_calibrated").value):
            transforms.insert(
                0,
                self._transform_message(
                    str(self.get_parameter("base_parent_frame").value),
                    str(self.get_parameter("arm_base_frame").value),
                    self.get_parameter("mount_xyz_m").value,
                    _rpy_quaternion(self.get_parameter("mount_rpy_rad").value),
                    now,
                ),
            )
        else:
            self.get_logger().warn(
                "未发布 base_link -> arm_base：请先测量安装位姿并设置 mount_calibrated=true。"
            )
        self._static_tf.sendTransform(transforms)

    @staticmethod
    def _transform_message(
        parent: str,
        child: str,
        xyz: Sequence[float],
        quaternion: Sequence[float],
        stamp: Any,
    ) -> TransformStamped:
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = (
            float(value) for value in xyz
        )
        (
            message.transform.rotation.x,
            message.transform.rotation.y,
            message.transform.rotation.z,
            message.transform.rotation.w,
        ) = (float(value) for value in quaternion)
        return message

    def _publish_status(self, stage: str, error: str = "") -> None:
        message = String()
        message.data = json.dumps(
            {
                "stage": stage,
                "target_id": self._active_target_id,
                "busy": self._busy,
                "connected": self._connected,
                "fatal_error": self._fatal_error,
                "error": error,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._status_pub.publish(message)

    def close(self) -> None:
        if hasattr(self, "_action"):
            self._action.destroy()
            del self._action
        if self._connected:
            with self._sdk_lock:
                self.arm.close()
            self._connected = False


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node: ArmRuntimeNode | None = None
    try:
        node = ArmRuntimeNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
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
