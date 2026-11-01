"""ROS2 adapter for the Unitree Go2W official SLAM navigation service.

The official service uses the standard Unitree ROS2 request/response topics
with JSON parameters. This node also relays the official status topics into
normal ROS2 topics used by rl2_nav.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import Future
from typing import Dict, Optional

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from rl2_hardware.pointcloud_utils import quaternion_from_yaw, yaw_from_quaternion
from rl2_interfaces.msg import UnitreeNavStatus
from rl2_interfaces.srv import UnitreeEndMapping, UnitreeInitPose, UnitreeNavigatePose


API_START_MAPPING = 1801
API_END_MAPPING = 1802
# Unitree's installed ``slam_operate`` service registers 1803 as
# ``chargeInit``.  On the installed Go2-W software this *initializes* the
# recharge workflow by cancelling the active mapping task; it does not start
# the physical docking approach.  It remains opt-in because cancelling an
# active mapping run is a material state change.
API_RECHARGE_INIT = 1803
API_INIT_POSE = 1804
API_NAVIGATE_POSE = 1102
API_PAUSE_NAVIGATION = 1201
API_RESUME_NAVIGATION = 1202
API_CLOSE_SLAM = 1901


class UnitreeSlamClientNode(Node):
    """Bridge Unitree SLAM API calls and status topics to the ROS2 mission stack."""

    def __init__(self) -> None:
        super().__init__("unitree_slam_client")

        self.declare_parameter("request_topic", "/api/slam_operate/request")
        self.declare_parameter("response_topic", "/api/slam_operate/response")
        self.declare_parameter("slam_info_topic", "/slam_info")
        self.declare_parameter("slam_key_info_topic", "/slam_key_info")
        self.declare_parameter(
            "official_odom_topic", "/unitree/slam_mapping/odom"
        )
        self.declare_parameter("official_odom_timeout_s", 0.5)
        # ``/slam_info`` is useful for official-service status, but its
        # currentPose can belong to a different SLAM session after a service
        # restart/relocalization.  Mixing it into the native mapping odometry
        # stream makes a map-frame reset look like a robot teleport to Nav2.
        # Keep that fallback opt-in only.
        self.declare_parameter("allow_slam_info_pose_fallback", False)
        self.declare_parameter("current_pose_topic", "/slam/current_pose")
        self.declare_parameter("odom_topic", "/slam/odom")
        self.declare_parameter("arrived_topic", "/unitree_slam/arrived")
        self.declare_parameter("command_status_topic", "/unitree_slam/command_status")
        self.declare_parameter("map_frame", "slam_map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("pcd_path", "/home/unitree/rl2.pcd")
        self.declare_parameter("slam_type", "indoor")
        self.declare_parameter("navigate_mode", 1)
        self.declare_parameter("navigate_speed", 0.5)
        # A normal navigation launch must never initiate a charging attempt.
        self.declare_parameter("recharge_command_enabled", False)
        self.declare_parameter("response_timeout_s", 5.0)

        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._pcd_path = str(self.get_parameter("pcd_path").value)
        self._slam_type = str(self.get_parameter("slam_type").value)
        self._navigate_mode = int(self.get_parameter("navigate_mode").value)
        self._navigate_speed = float(self.get_parameter("navigate_speed").value)
        self._official_odom_timeout_s = float(
            self.get_parameter("official_odom_timeout_s").value
        )

        self._pending: Dict[int, Future] = {}
        self._pending_api: Dict[int, int] = {}
        self._pending_lock = threading.Lock()
        self._next_request_id = int(time.monotonic_ns() & 0x7FFFFFFF) or 1
        self._id_lock = threading.Lock()
        self._active_request_id = 0
        self._awaiting_arrival = False
        self._last_current_pose: Optional[PoseStamped] = None
        self._last_official_odom_time = None

        self._api_available = False
        self._Request = None
        self._Response = None
        self._req_pub = None
        self._init_unitree_api()

        self._current_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("current_pose_topic").value),
            10,
        )
        self._odom_pub = self.create_publisher(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            10,
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._arrived_pub = self.create_publisher(
            UnitreeNavStatus,
            str(self.get_parameter("arrived_topic").value),
            10,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("command_status_topic").value),
            10,
        )

        self.create_subscription(
            String,
            str(self.get_parameter("slam_info_topic").value),
            self._on_slam_info,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("slam_key_info_topic").value),
            self._on_slam_key_info,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("official_odom_topic").value),
            self._on_official_odom,
            qos_profile_sensor_data,
        )

        self._init_services()

        self.get_logger().info(
            "unitree_slam_client ready: request=%s response=%s api_available=%s"
            % (
                str(self.get_parameter("request_topic").value),
                str(self.get_parameter("response_topic").value),
                self._api_available,
            )
        )

    def _init_unitree_api(self) -> None:
        try:
            from unitree_api.msg import Request, Response
        except ImportError:
            self.get_logger().error(
                "unitree_api ROS2 messages not found. Source the unitree_ros2 "
                "CycloneDDS workspace before launching this node."
            )
            return

        self._Request = Request
        self._Response = Response
        self._api_available = True
        self._req_pub = self.create_publisher(
            Request,
            str(self.get_parameter("request_topic").value),
            10,
        )
        self.create_subscription(
            Response,
            str(self.get_parameter("response_topic").value),
            self._on_api_response,
            10,
        )

    def _init_services(self) -> None:
        self.create_service(Trigger, "/unitree_slam/start_mapping", self._on_start_mapping)
        self.create_service(
            Trigger,
            "/unitree_slam/recharge_init",
            self._on_recharge_init,
        )
        self.create_service(
            UnitreeEndMapping,
            "/unitree_slam/end_mapping",
            self._on_end_mapping,
        )
        self.create_service(
            UnitreeInitPose,
            "/unitree_slam/init_pose",
            self._on_init_pose,
        )
        self.create_service(
            UnitreeNavigatePose,
            "/unitree_slam/navigate_pose",
            self._on_navigate_pose,
        )
        self.create_service(Trigger, "/unitree_slam/pause_navigation", self._on_pause)
        self.create_service(Trigger, "/unitree_slam/resume_navigation", self._on_resume)
        self.create_service(Trigger, "/unitree_slam/shutdown", self._on_shutdown)

    def _next_id(self) -> int:
        with self._id_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id

    def _send_request(self, api_id: int, parameter: str) -> Optional[int]:
        if not self._api_available or self._req_pub is None:
            self._publish_status(False, api_id, "unitree_api messages unavailable")
            return None

        request_id = self._next_id()
        request = self._Request()
        request.header.identity.id = request_id
        request.header.identity.api_id = api_id
        request.parameter = parameter

        with self._pending_lock:
            self._pending[request_id] = Future()
            self._pending_api[request_id] = api_id

        self._req_pub.publish(request)
        self.get_logger().info("Sent Unitree SLAM request api_id=%d id=%d" % (api_id, request_id))
        return request_id

    def _on_api_response(self, msg) -> None:
        request_id = int(msg.header.identity.id)
        api_id = int(msg.header.identity.api_id)
        transport_code = int(msg.header.status.code)
        response_data = str(msg.data)
        semantic_success = True
        semantic_code = 0
        semantic_info = response_data
        try:
            decoded = json.loads(response_data)
            if isinstance(decoded, dict):
                semantic_success = _as_bool(decoded.get("succeed", True))
                semantic_code = int(decoded.get("errorCode", 0))
                semantic_info = str(decoded.get("info", response_data))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        success = transport_code == 0 and semantic_success and semantic_code == 0
        code = transport_code if transport_code != 0 else semantic_code
        with self._pending_lock:
            future = self._pending.pop(request_id, None)
            self._pending_api.pop(request_id, None)

        if future is None:
            self.get_logger().debug(
                "Ignoring Unitree response not owned by this client: api_id=%d id=%d"
                % (api_id, request_id)
            )
            return

        payload = {
            "succeed": success,
            "errorCode": code,
            "info": semantic_info,
        }
        if not success:
            self.get_logger().error(
                "Unitree SLAM request failed: api_id=%d id=%d code=%d data=%s"
                % (api_id, request_id, code, response_data)
            )
        future.set_result(payload)
        if api_id == API_NAVIGATE_POSE and request_id == self._active_request_id:
            self._awaiting_arrival = success
        self._publish_status(
            success,
            api_id,
            payload["info"],
            request_id=request_id,
            error_code=code,
        )

    def _on_slam_info(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return

        info_type = payload.get("type")
        data = payload.get("data") or {}
        if info_type in ("pos_info", "mapping_info"):
            pose_data = data.get("currentPose")
            if pose_data:
                self._publish_current_pose(pose_data)

        if info_type == "ctrl_info":
            is_arrived = _as_bool(data.get("is_arrived", False))
            if is_arrived and self._awaiting_arrival:
                self._publish_arrived(self._active_request_id, "ctrl_info")
                self._awaiting_arrival = False

    def _on_slam_key_info(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if payload.get("type") != "task_result":
            return
        data = payload.get("data") or {}
        if _as_bool(data.get("is_arrived", False)) and self._awaiting_arrival:
            self._publish_arrived(self._active_request_id, "task_result")
            self._awaiting_arrival = False

    def _publish_current_pose(self, pose_data: dict) -> None:
        # The native 20 Hz mapping odometry contains both pose and twist and is
        # the primary Nav2 source.  Keep slam_info only as a failover when that
        # stream disappears.
        if not bool(
            self.get_parameter("allow_slam_info_pose_fallback").value
        ):
            return

        if self._last_official_odom_time is not None:
            age = (
                self.get_clock().now() - self._last_official_odom_time
            ).nanoseconds * 1.0e-9
            if age <= self._official_odom_timeout_s:
                return

        try:
            x = float(pose_data["x"])
            y = float(pose_data["y"])
            z = float(pose_data.get("z", 0.0))
            qx = float(pose_data.get("q_x", 0.0))
            qy = float(pose_data.get("q_y", 0.0))
            qz = float(pose_data.get("q_z", 0.0))
            qw = float(pose_data.get("q_w", 1.0))
        except (KeyError, TypeError, ValueError):
            return

        now = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self._map_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self._current_pose_pub.publish(pose)

        odom = Odometry()
        odom.header.stamp = now
        # During official API 1801 mapping, currentPose is already expressed in
        # slam_map.  Expose that same contract to Nav2 instead of inventing an
        # unconnected slam_odom frame.
        odom.header.frame_id = self._map_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose = pose.pose
        self._odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = self._map_frame
        transform.child_frame_id = self._base_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation = pose.pose.orientation
        self._tf_broadcaster.sendTransform(transform)
        self._last_current_pose = pose

    def _on_official_odom(self, msg: Odometry) -> None:
        """Restamp and expose native Unitree mapping odometry to Nav2.

        The native child frame is ``livox_frame``.  Nav2 only needs planar
        base motion here, so x/y and yaw are exposed as ``base_footprint``;
        the fixed lidar-to-base offset is intentionally excluded from the 2D
        navigation pose.
        """
        now = self.get_clock().now()
        self._last_official_odom_time = now
        stamp = now.to_msg()

        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        planar_orientation = quaternion_from_yaw(yaw)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self._map_frame
        pose.pose.position.x = float(msg.pose.pose.position.x)
        pose.pose.position.y = float(msg.pose.pose.position.y)
        pose.pose.position.z = 0.0
        pose.pose.orientation = planar_orientation
        self._current_pose_pub.publish(pose)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._map_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose = pose.pose
        odom.pose.covariance = msg.pose.covariance
        odom.twist = msg.twist
        self._odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self._map_frame
        transform.child_frame_id = self._base_frame
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.rotation = planar_orientation
        self._tf_broadcaster.sendTransform(transform)
        self._last_current_pose = pose

    def _publish_arrived(self, request_id: int, source: str) -> None:
        msg = UnitreeNavStatus()
        msg.request_id = int(request_id)
        msg.is_arrived = True
        msg.stamp = self.get_clock().now().to_msg()
        msg.info = source
        self._arrived_pub.publish(msg)
        self.get_logger().info("Official SLAM reported arrival for request %d" % request_id)

    def _publish_status(
        self,
        success: bool,
        api_id: int,
        info: str,
        *,
        request_id: int = 0,
        error_code: int = 0,
    ) -> None:
        payload = {
            "succeed": bool(success),
            "api_id": int(api_id),
            "request_id": int(request_id),
            "error_code": int(error_code),
            "info": str(info),
        }
        self._status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    @staticmethod
    def _queued_message(action: str, request_id: Optional[int]) -> str:
        if request_id is None:
            return "unitree_api unavailable"
        return json.dumps(
            {"state": "queued", "action": action, "request_id": int(request_id)},
            separators=(",", ":"),
        )

    def _on_start_mapping(self, request, response) -> Trigger.Response:
        del request
        parameter = json.dumps({"data": {"slam_type": self._slam_type}}, separators=(",", ":"))
        request_id = self._send_request(API_START_MAPPING, parameter)
        response.success = request_id is not None
        response.message = self._queued_message("start_mapping", request_id)
        return response

    def _on_recharge_init(self, request, response) -> Trigger.Response:
        """Initialize the official recharge workflow after an explicit opt-in.

        This is deliberately not advertised as a ``dock`` command: the
        installed official service acknowledges API 1803 with e.g. ``Cancel
        mapping successfully`` and does not move the robot.  The separate
        official charge-navigation API must not be guessed or invoked until
        its request schema has been verified for this software version.
        """
        del request
        if not _as_bool(self.get_parameter("recharge_command_enabled").value):
            response.success = False
            response.message = (
                "recharge command blocked: launch with "
                "recharge_command_enabled:=true only after completing the "
                "ArUco/dock preflight"
            )
            self._publish_status(
                False,
                API_RECHARGE_INIT,
                "blocked locally: recharge_command_enabled is false",
            )
            return response

        # The installed Unitree chargeInit handler accepts an empty JSON
        # request.  It initializes recharge / cancels mapping, but does not
        # command physical docking on this robot software version.
        request_id = self._send_request(API_RECHARGE_INIT, "{}")
        response.success = request_id is not None
        response.message = self._queued_message("recharge_init", request_id)
        return response

    def _on_end_mapping(self, request, response) -> UnitreeEndMapping.Response:
        address = request.address if request.address else self._pcd_path
        parameter = json.dumps({"data": {"address": address}}, separators=(",", ":"))
        request_id = self._send_request(API_END_MAPPING, parameter)
        response.success = request_id is not None
        response.message = self._queued_message("end_mapping", request_id)
        return response

    def _on_init_pose(self, request, response) -> UnitreeInitPose.Response:
        pose = request.pose.pose
        address = request.address if request.address else self._pcd_path
        data = {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "q_x": pose.orientation.x,
            "q_y": pose.orientation.y,
            "q_z": pose.orientation.z,
            "q_w": pose.orientation.w,
            "address": address,
        }
        parameter = json.dumps({"data": data}, separators=(",", ":"))
        request_id = self._send_request(API_INIT_POSE, parameter)
        response.success = request_id is not None
        response.message = self._queued_message("init_pose", request_id)
        return response

    def _on_navigate_pose(self, request, response) -> UnitreeNavigatePose.Response:
        pose = request.pose.pose
        mode = request.mode if request.mode >= 0 else self._navigate_mode
        speed = request.speed if request.speed > 0.0 else self._navigate_speed
        target = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "q_x": float(pose.orientation.x),
            "q_y": float(pose.orientation.y),
            "q_z": float(pose.orientation.z),
            "q_w": float(pose.orientation.w),
        }
        parameter = json.dumps(
            {"data": {"targetPose": target, "mode": int(mode), "speed": float(speed)}},
            separators=(",", ":"),
        )
        request_id = self._send_request(API_NAVIGATE_POSE, parameter)
        response.success = request_id is not None
        response.request_id = int(request_id or 0)
        response.message = self._queued_message("navigate_pose", request_id)
        if request_id is not None:
            self._active_request_id = request_id
            self._awaiting_arrival = False
        return response

    def _on_pause(self, request, response) -> Trigger.Response:
        del request
        request_id = self._send_request(API_PAUSE_NAVIGATION, "{}")
        response.success = request_id is not None
        response.message = self._queued_message("pause_navigation", request_id)
        return response

    def _on_resume(self, request, response) -> Trigger.Response:
        del request
        request_id = self._send_request(API_RESUME_NAVIGATION, "{}")
        response.success = request_id is not None
        response.message = self._queued_message("resume_navigation", request_id)
        return response

    def _on_shutdown(self, request, response) -> Trigger.Response:
        del request
        request_id = self._send_request(API_CLOSE_SLAM, "{}")
        response.success = request_id is not None
        response.message = self._queued_message("shutdown", request_id)
        return response


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnitreeSlamClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
