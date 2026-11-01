"""Fail-safe ROS 2 Twist bridge for the Unitree GO2-W sport service."""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Optional

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from unitree_api.msg import Request, Response
from unitree_go.msg import SportModeState


API_STOP_MOVE = 1003
API_BALANCE_STAND = 1002
API_STAND_UP = 1004
API_STAND_DOWN = 1005
API_MOVE = 1008


def clamp(value: float, limit: float) -> float:
    """Clamp a scalar symmetrically around zero."""
    return max(-abs(limit), min(abs(limit), value))


def move_parameter(vx: float, vy: float, yaw_rate: float) -> str:
    """Return the JSON payload expected by Unitree SportClient::Move."""
    return json.dumps(
        {"x": float(vx), "y": float(vy), "z": float(yaw_rate)},
        separators=(",", ":"),
    )


def response_succeeded(value: object) -> bool:
    """Interpret the JSON ``succeed`` field without treating "false" as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def posture_exclusive_active(now_ns: int, exclusive_until_ns: int) -> bool:
    """Return whether a one-shot posture action still owns the Sport API."""
    return int(exclusive_until_ns) > 0 and int(now_ns) < int(exclusive_until_ns)


class Go2WSportCmdBridge(Node):
    """Translate a gated Twist stream into Unitree high-level sport requests.

    Merely launching this node never publishes a sport request.  A caller must
    explicitly enable it through ``~/enable``.  Once enabled, stale velocity
    input or stale sport state produces API 1003 (StopMove).
    """

    def __init__(self) -> None:
        super().__init__("go2w_sport_cmd_bridge")

        self.declare_parameter("input_topic", "/cmd_vel_sport")
        self.declare_parameter("request_topic", "/api/sport/request")
        self.declare_parameter("response_topic", "/api/sport/response")
        self.declare_parameter("sport_state_topic", "/lf/sportmodestate")
        self.declare_parameter("status_topic", "/go2w_sport_cmd_bridge/status")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("input_timeout_sec", 0.30)
        self.declare_parameter("sport_state_timeout_sec", 0.50)
        self.declare_parameter("max_linear_x", 0.40)
        self.declare_parameter("max_linear_y", 0.50)
        self.declare_parameter("max_angular_z", 0.50)
        self.declare_parameter("allow_reverse", False)
        self.declare_parameter("zero_epsilon", 1.0e-3)
        self.declare_parameter("stop_heartbeat_sec", 0.50)
        self.declare_parameter("stop_repeat_count", 3)
        # StandDown/StandUp are one-shot Sport actions.  A stale zero-velocity
        # watchdog must not immediately overwrite them with API 1003 while the
        # firmware is still executing the posture transition.  Explicit
        # stop_move and disabling the bridge always preempt this window.
        self.declare_parameter("stand_down_exclusive_sec", 3.0)
        self.declare_parameter("stand_up_exclusive_sec", 4.0)
        self.declare_parameter("balance_stand_exclusive_sec", 2.0)
        self.declare_parameter("enabled_on_start", False)

        self._lock = threading.Lock()
        self._enabled = bool(self.get_parameter("enabled_on_start").value)
        self._ever_enabled = self._enabled
        self._target: Optional[tuple[float, float, float]] = None
        self._last_input_ns = 0
        self._last_sport_state_ns = 0
        self._last_stop_ns = 0
        self._move_active = False
        self._posture_exclusive_until_ns = 0
        self._last_gate_reason = ""
        # Match Unitree's BaseClient: request identities use CLOCK_MONOTONIC
        # nanoseconds, not wall-clock epoch microseconds.
        self._last_request_id = 0
        self._pending: dict[int, tuple[int, int]] = {}

        self._request_pub = self.create_publisher(
            Request, str(self.get_parameter("request_topic").value), 10
        )
        # TRANSIENT_LOCAL keeps the last status latched so a late
        # `topic echo --once` reader can distinguish "gate disabled" from
        # "bridge process dead" even though a disabled bridge is silent.
        status_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), status_qos
        )
        self.create_subscription(
            Response,
            str(self.get_parameter("response_topic").value),
            self._on_response,
            10,
        )
        self.create_subscription(
            SportModeState,
            str(self.get_parameter("sport_state_topic").value),
            self._on_sport_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("input_topic").value),
            self._on_cmd_vel,
            10,
        )
        self.create_service(SetBool, "~/enable", self._on_enable)
        self.create_service(Trigger, "~/stop_move", self._on_stop_move)
        self.create_service(Trigger, "~/stand_down", self._on_stand_down)
        self.create_service(Trigger, "~/stand_up", self._on_stand_up)
        self.create_service(Trigger, "~/balance_stand", self._on_balance_stand)

        rate = max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.create_timer(1.0 / rate, self._on_timer)

        self.get_logger().warn(
            "GO2-W sport bridge ready but %s; no request is published until enabled."
            % ("ENABLED" if self._enabled else "DISABLED")
        )
        self._publish_gate_status(self._enabled)
        if self._enabled:
            # enabled_on_start is supported for testing, but still clears any
            # old velocity before accepting a fresh input sample.
            self._publish_stop("startup enabled")

    def _on_cmd_vel(self, msg: Twist) -> None:
        values = (msg.linear.x, msg.linear.y, msg.angular.z)
        if not all(math.isfinite(float(value)) for value in values):
            self.get_logger().error("Rejected non-finite cmd_vel_sport")
            with self._lock:
                self._target = None
                self._last_input_ns = 0
            return

        requested_x = float(msg.linear.x)
        if not bool(self.get_parameter("allow_reverse").value):
            requested_x = max(0.0, requested_x)
        vx = clamp(requested_x, float(self.get_parameter("max_linear_x").value))
        vy = clamp(
            float(msg.linear.y),
            float(self.get_parameter("max_linear_y").value),
        )
        yaw = clamp(
            float(msg.angular.z),
            float(self.get_parameter("max_angular_z").value),
        )
        with self._lock:
            self._target = (vx, vy, yaw)
            self._last_input_ns = self.get_clock().now().nanoseconds

    def _on_sport_state(self, msg: SportModeState) -> None:
        del msg
        with self._lock:
            self._last_sport_state_ns = self.get_clock().now().nanoseconds

    def _on_enable(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        with self._lock:
            self._enabled = bool(request.data)
            self._ever_enabled = self._ever_enabled or self._enabled
            self._target = None
            self._last_input_ns = 0
            self._posture_exclusive_until_ns = 0
        if request.data:
            self._publish_stop("bridge enabled; waiting for fresh input", force=True)
            response.message = "sport output enabled; waiting for fresh cmd_vel_sport"
        else:
            self._publish_stop("bridge disabled", force=True)
            response.message = "sport output disabled and StopMove sent"
        self._publish_gate_status(self._enabled)
        response.success = True
        self.get_logger().warn(response.message)
        return response

    def _motion_action_allowed(self) -> bool:
        """Posture actions share the explicit ownership gate with velocity."""
        with self._lock:
            return self._enabled

    def _on_stop_move(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            self._posture_exclusive_until_ns = 0
        self._publish_stop("explicit stop_move service", force=True)
        response.success = True
        response.message = "StopMove queued"
        return response

    def _on_stand_down(self, _request: Trigger.Request, response: Trigger.Response):
        if not self._motion_action_allowed():
            response.success = False
            response.message = "sport output is disabled; StandDown was not sent"
            return response
        self._publish_stop("before StandDown", force=True)
        with self._lock:
            self._posture_exclusive_until_ns = (
                self.get_clock().now().nanoseconds
                + int(
                    max(
                        0.0,
                        float(self.get_parameter("stand_down_exclusive_sec").value),
                    )
                    * 1_000_000_000
                )
            )
        self._publish_request(API_STAND_DOWN, "")
        response.success = True
        response.message = "StandDown queued"
        return response

    def _on_stand_up(self, _request: Trigger.Request, response: Trigger.Response):
        if not self._motion_action_allowed():
            response.success = False
            response.message = "sport output is disabled; StandUp was not sent"
            return response
        with self._lock:
            self._posture_exclusive_until_ns = (
                self.get_clock().now().nanoseconds
                + int(
                    max(
                        0.0,
                        float(self.get_parameter("stand_up_exclusive_sec").value),
                    )
                    * 1_000_000_000
                )
            )
        self._publish_request(API_STAND_UP, "")
        response.success = True
        response.message = "StandUp queued"
        return response

    def _on_balance_stand(
        self, _request: Trigger.Request, response: Trigger.Response
    ):
        """Return StandUp's terminal mode to the locomotion-ready mode.

        GO2-W reports a successful StandUp while remaining in sport mode 6;
        Move (1008) is rejected with status -1 in that mode.  BalanceStand
        (1002) transitions it to mode 1, where velocity commands are accepted.
        """
        if not self._motion_action_allowed():
            response.success = False
            response.message = "sport output is disabled; BalanceStand was not sent"
            return response
        with self._lock:
            self._posture_exclusive_until_ns = (
                self.get_clock().now().nanoseconds
                + int(
                    max(
                        0.0,
                        float(
                            self.get_parameter("balance_stand_exclusive_sec").value
                        ),
                    )
                    * 1_000_000_000
                )
            )
        self._publish_request(API_BALANCE_STAND, "")
        response.success = True
        response.message = "BalanceStand queued"
        return response

    def _on_timer(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            enabled = self._enabled
            target = self._target
            last_input_ns = self._last_input_ns
            last_state_ns = self._last_sport_state_ns
            posture_exclusive_until_ns = self._posture_exclusive_until_ns

        if not enabled:
            return
        if posture_exclusive_active(now_ns, posture_exclusive_until_ns):
            # The posture action owns the Sport API during this bounded
            # interval.  Do not send Move or watchdog StopMove requests that
            # would cancel the transition.  stop_move/disable remain immediate.
            return

        input_timeout_ns = int(
            float(self.get_parameter("input_timeout_sec").value) * 1_000_000_000
        )
        state_timeout_ns = int(
            float(self.get_parameter("sport_state_timeout_sec").value)
            * 1_000_000_000
        )
        input_fresh = (
            target is not None
            and last_input_ns > 0
            and 0 <= now_ns - last_input_ns <= input_timeout_ns
        )
        state_fresh = (
            last_state_ns > 0 and 0 <= now_ns - last_state_ns <= state_timeout_ns
        )

        reason = ""
        if not state_fresh:
            reason = "sport state missing/stale"
        elif not input_fresh:
            reason = "cmd_vel_sport missing/stale"

        if reason:
            self._publish_stop(reason)
            self._report_gate_reason(reason)
            return

        assert target is not None
        vx, vy, yaw = target
        epsilon = abs(float(self.get_parameter("zero_epsilon").value))
        if abs(vx) <= epsilon and abs(vy) <= epsilon and abs(yaw) <= epsilon:
            self._publish_stop("zero command")
            self._report_gate_reason("")
            return

        self._publish_request(API_MOVE, move_parameter(vx, vy, yaw))
        self._move_active = True
        self._report_gate_reason("")

    def _publish_stop(self, reason: str, *, force: bool = False) -> None:
        now_ns = self.get_clock().now().nanoseconds
        heartbeat_ns = int(
            float(self.get_parameter("stop_heartbeat_sec").value) * 1_000_000_000
        )
        if not force and not self._move_active and now_ns - self._last_stop_ns < heartbeat_ns:
            return

        repeat = max(int(self.get_parameter("stop_repeat_count").value), 1) if force else 1
        for _ in range(repeat):
            self._publish_request(API_STOP_MOVE, "")
        self._move_active = False
        self._last_stop_ns = now_ns
        self._publish_status(True, API_STOP_MOVE, 0, "StopMove: %s" % reason)

    def _publish_request(self, api_id: int, parameter: str) -> int:
        request = Request()
        with self._lock:
            request_id = max(time.monotonic_ns(), self._last_request_id + 1)
            self._last_request_id = request_id
            self._pending[request_id] = (
                api_id,
                self.get_clock().now().nanoseconds,
            )
            # Bound diagnostics state if a firmware does not answer every Move.
            if len(self._pending) > 200:
                oldest = sorted(self._pending)[: len(self._pending) - 200]
                for old_id in oldest:
                    self._pending.pop(old_id, None)
        request.header.identity.id = request_id
        request.header.identity.api_id = api_id
        request.parameter = parameter
        self._request_pub.publish(request)
        if api_id in (
            API_MOVE,
            API_BALANCE_STAND,
            API_STAND_UP,
            API_STAND_DOWN,
        ):
            self._status_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "stage": "request",
                            "request_id": int(request_id),
                            "api_id": int(api_id),
                            "parameter": str(parameter),
                            "enabled": bool(self._enabled),
                        },
                        separators=(",", ":"),
                    )
                )
            )
        return request_id

    def _on_response(self, msg: Response) -> None:
        request_id = int(msg.header.identity.id)
        api_id = int(msg.header.identity.api_id)
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        transport_code = int(msg.header.status.code)
        success = transport_code == 0
        info = str(msg.data)
        try:
            decoded = json.loads(info)
            if isinstance(decoded, dict):
                success = success and response_succeeded(decoded.get("succeed", True))
                info = str(decoded.get("info", info))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        # Publish every matched response.  Previously successful Move (1008)
        # responses were hidden, which made a healthy unlocked robot look as if
        # no Move request had been sent at all.
        self._publish_status(success, api_id, transport_code, info, stage="response")

    def _publish_status(
        self,
        success: bool,
        api_id: int,
        error_code: int,
        info: str,
        *,
        stage: str = "local",
    ) -> None:
        payload = {
            "stage": str(stage),
            "succeed": bool(success),
            "api_id": int(api_id),
            "error_code": int(error_code),
            "info": str(info),
            "enabled": bool(self._enabled),
        }
        self._status_pub.publish(
            String(data=json.dumps(payload, separators=(",", ":")))
        )

    def _publish_gate_status(self, enabled: bool) -> None:
        """Latch the enable state so silent-disabled differs from dead."""
        payload = {"stage": "gate", "enabled": bool(enabled)}
        self._status_pub.publish(
            String(data=json.dumps(payload, separators=(",", ":")))
        )

    def _report_gate_reason(self, reason: str) -> None:
        if reason == self._last_gate_reason:
            return
        if reason:
            self.get_logger().error("Sport safety gate forcing StopMove: %s" % reason)
        else:
            self.get_logger().info("Sport safety gate inputs healthy")
        self._last_gate_reason = reason

    def stop_before_shutdown(self) -> None:
        """Send StopMove only if this process previously took ownership."""
        if self._ever_enabled:
            self._publish_stop("node shutdown", force=True)


def main() -> None:
    rclpy.init()
    node = Go2WSportCmdBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            # A launch SIGINT may already have invalidated the rclpy context.
            # Publish the final StopMove only while the context is usable.
            if rclpy.ok():
                try:
                    node.stop_before_shutdown()
                    rclpy.spin_once(node, timeout_sec=0.10)
                except Exception as exc:
                    # SIGINT can invalidate the context between rclpy.ok()
                    # and publish().  A StopMove was already sent when the
                    # output gate was disabled; suppress only this shutdown
                    # race and keep all other errors visible.
                    if "context is invalid" not in str(exc):
                        raise
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
