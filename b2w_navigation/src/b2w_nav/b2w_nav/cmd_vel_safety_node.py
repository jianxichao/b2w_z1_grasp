"""Fail-safe filter between Nav2 velocity output and the real controller."""

from __future__ import annotations

import json
import math
import threading
from typing import Optional

from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_action_status_default,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _approach(current: float, target: float, max_step: float) -> float:
    if target > current:
        return min(target, current + max_step)
    return max(target, current - max_step)


def _angular_output_step(
    current: float,
    target: float,
    max_accel: float,
    dt: float,
    passthrough: bool,
) -> float:
    """Apply one angular output step.

    Nav2's velocity_smoother already applies angular acceleration limits.  A
    second slew limiter in the safety gate adds phase lag at direction changes,
    so real-robot launches may explicitly select passthrough.
    """
    if passthrough:
        return target
    return _approach(current, target, max_accel * dt)


def _minimum_nonzero_magnitude(value: float, minimum: float) -> float:
    """Lift a non-zero command out of a measured actuator deadband."""
    if value == 0.0 or minimum <= 0.0 or abs(value) >= minimum:
        return value
    return math.copysign(minimum, value)


def _odom_step_is_jump(
    previous_xy: Optional[tuple[float, float]],
    current_xy: tuple[float, float],
    threshold_m: float,
) -> bool:
    """Return whether a consecutive odometry step is physically implausible."""
    if previous_xy is None or threshold_m <= 0.0:
        return False
    return math.hypot(
        current_xy[0] - previous_xy[0],
        current_xy[1] - previous_xy[1],
    ) > threshold_m


def _cost_at_xy(
    costmap: OccupancyGrid, x: float, y: float
) -> Optional[int]:
    """Return the occupancy cost at a map-frame point, or None out of bounds."""
    resolution = float(costmap.info.resolution)
    if resolution <= 0.0:
        return None
    col = int(math.floor((x - float(costmap.info.origin.position.x)) / resolution))
    row = int(math.floor((y - float(costmap.info.origin.position.y)) / resolution))
    width = int(costmap.info.width)
    height = int(costmap.info.height)
    if not (0 <= col < width and 0 <= row < height):
        return None
    return int(costmap.data[row * width + col])


class CmdVelSafetyNode(Node):
    """Clamp, slew-limit and timeout Nav2 Twist commands.

    The output stays disabled until ``~/enable`` is called with ``data: true``.
    Disabling immediately publishes a zero Twist.
    """

    def __init__(self) -> None:
        super().__init__("cmd_vel_safety_node")

        self.declare_parameter("input_topic", "/cmd_vel_nav")
        # A dedicated, manually-started closed-loop route may bypass Nav2's
        # planner, but it must not share /cmd_vel_raw with Nav2 controllers.
        # It still passes through this gate and the Sport bridge.
        self.declare_parameter("direct_input_topic", "/cmd_vel_direct")
        self.declare_parameter("recovery_input_topic", "/cmd_vel_recovery")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("max_linear_x", 0.40)
        self.declare_parameter("max_linear_y", 0.50)
        self.declare_parameter("min_effective_linear_y", 0.25)
        self.declare_parameter("allow_lateral_navigation", False)
        self.declare_parameter("recovery_timeout_sec", 0.30)
        self.declare_parameter("max_angular_z", 0.50)
        self.declare_parameter("min_effective_angular_z", 0.20)
        self.declare_parameter("max_linear_accel", 0.50)
        self.declare_parameter("max_linear_y_accel", 0.40)
        self.declare_parameter("max_angular_accel", 0.65)
        self.declare_parameter("angular_passthrough", False)
        self.declare_parameter("input_timeout_sec", 0.30)
        self.declare_parameter("allow_reverse", False)
        # Recovery 通道专属倒车上限；普通导航流仍被 allow_reverse 限制。
        self.declare_parameter("recovery_max_reverse", 0.20)
        self.declare_parameter("require_scan", True)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("scan_timeout_sec", 1.0)
        self.declare_parameter("require_odom", True)
        self.declare_parameter("odom_topic", "/slam/odom")
        self.declare_parameter("odom_timeout_sec", 0.5)
        # A metre-scale step between consecutive 20 Hz odometry samples is a
        # SLAM frame reset, not physical motion.  Disarm instead of following
        # a path expressed in the stale frame.
        self.declare_parameter("odom_jump_stop_distance_m", 1.0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("enabled_on_start", False)
        # Latched gate snapshot for operators, mirroring the
        # slam_pose_jump_guard status pattern: a late `topic echo --once`
        # reader immediately sees whether the output is armed.
        self.declare_parameter("status_topic", "/cmd_vel_safety_node/status")
        # Generic recovery for manually sent Nav2 goals as well as Frontier
        # goals.  It is deliberately conservative: only a sustained
        # translational command with almost no odometry progress can trigger
        # one short lateral pulse, and only when a side is scan-clear.
        self.declare_parameter("stuck_recovery_enabled", False)
        self.declare_parameter("stuck_command_min_speed_mps", 0.12)
        self.declare_parameter("stuck_timeout_sec", 3.0)
        self.declare_parameter("stuck_min_progress_m", 0.05)
        self.declare_parameter("stuck_recovery_speed_mps", 0.20)
        self.declare_parameter("stuck_recovery_duration_sec", 1.0)
        self.declare_parameter("stuck_recovery_side_clearance_m", 0.60)
        self.declare_parameter("stuck_recovery_cooldown_sec", 5.0)
        # The planner can reject a goal before publishing any cmd_vel when the
        # robot's exact global-costmap cell is lethal.  This trigger monitors
        # the Nav2 action and costmap directly, so it does not depend on the
        # three-second command/no-progress watchdog below.
        self.declare_parameter("lethal_start_recovery_enabled", False)
        self.declare_parameter(
            "navigation_status_topic", "/navigate_to_pose/_action/status"
        )
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("global_costmap_timeout_sec", 1.0)
        self.declare_parameter("lethal_start_cost_threshold", 99)
        self.declare_parameter("lethal_start_trigger_delay_sec", 0.15)
        self.declare_parameter("lethal_start_max_attempts", 2)

        input_topic = str(self.get_parameter("input_topic").value)
        direct_input_topic = str(self.get_parameter("direct_input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        publish_rate = max(float(self.get_parameter("publish_rate_hz").value), 1.0)

        self._lock = threading.Lock()
        self._enabled = bool(self.get_parameter("enabled_on_start").value)
        self._odom_jump_monitoring_enabled = True
        self._target: Optional[Twist] = None
        self._direct_target: Optional[Twist] = None
        self._recovery_target: Optional[Twist] = None
        self._last_input_ns = 0
        self._last_direct_input_ns = 0
        self._last_recovery_input_ns = 0
        self._last_scan_ns = 0
        self._last_odom_ns = 0
        self._last_odom_xy = None
        self._odom_xy = None
        self._stuck_window_start_ns = 0
        self._stuck_window_start_xy = None
        self._auto_recovery_until_ns = 0
        self._auto_recovery_direction = 0.0
        self._last_auto_recovery_end_ns = 0
        self._latest_scan = None
        self._latest_costmap: Optional[OccupancyGrid] = None
        self._last_costmap_ns = 0
        self._navigation_active = False
        self._navigation_active_since_ns = 0
        self._lethal_start_attempts = 0
        self._last_output_ns = self.get_clock().now().nanoseconds
        self._output = Twist()
        self._last_gate_reason = ""

        self._publisher = self.create_publisher(Twist, output_topic, 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            status_qos,
        )
        self.create_subscription(Twist, input_topic, self._on_cmd_vel, 10)
        self.create_subscription(Twist, direct_input_topic, self._on_direct_cmd_vel, 10)
        self.create_subscription(
            Twist,
            str(self.get_parameter("recovery_input_topic").value),
            self._on_recovery_cmd_vel,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("global_costmap_topic").value),
            self._on_global_costmap,
            10,
        )
        self.create_subscription(
            GoalStatusArray,
            str(self.get_parameter("navigation_status_topic").value),
            self._on_navigation_status,
            qos_profile_action_status_default,
        )
        self.create_service(SetBool, "~/enable", self._on_enable)
        self.create_service(
            SetBool, "~/odom_jump_monitoring", self._on_odom_jump_monitoring
        )
        self.create_timer(1.0 / publish_rate, self._on_timer)

        self.get_logger().warn(
            "Velocity safety filter ready but %s: %s -> %s"
            % ("ENABLED" if self._enabled else "DISABLED", input_topic, output_topic)
        )
        self._publish_status("startup")

    def _on_cmd_vel(self, msg: Twist) -> None:
        if not all(
            math.isfinite(value)
            for value in (msg.linear.x, msg.linear.y, msg.angular.z)
        ):
            self.get_logger().error("Rejected non-finite cmd_vel")
            return

        target = Twist()
        requested_x = float(msg.linear.x)
        if not bool(self.get_parameter("allow_reverse").value):
            requested_x = max(0.0, requested_x)
        target.linear.x = _clamp(requested_x, float(self.get_parameter("max_linear_x").value))
        if bool(self.get_parameter("allow_lateral_navigation").value):
            requested_y = _clamp(
                float(msg.linear.y),
                float(self.get_parameter("max_linear_y").value),
            )
            target.linear.y = _minimum_nonzero_magnitude(
                requested_y,
                float(self.get_parameter("min_effective_linear_y").value),
            )
        requested_angular_z = _clamp(
            float(msg.angular.z),
            float(self.get_parameter("max_angular_z").value),
        )
        target.angular.z = _clamp(
            _minimum_nonzero_magnitude(
                requested_angular_z,
                float(self.get_parameter("min_effective_angular_z").value),
            ),
            float(self.get_parameter("max_angular_z").value),
        )
        with self._lock:
            self._target = target
            self._last_input_ns = self.get_clock().now().nanoseconds

    def _on_direct_cmd_vel(self, msg: Twist) -> None:
        """Accept a dedicated non-Nav2 command through the same safety gate.

        The command is deliberately clamped identically to /cmd_vel_nav.  A
        fresh direct command has priority over a Nav2 command so a manually
        started direct mission cannot be cancelled by Nav2's idle zero output.
        """
        if not all(
            math.isfinite(value)
            for value in (msg.linear.x, msg.linear.y, msg.angular.z)
        ):
            self.get_logger().error("Rejected non-finite direct cmd_vel")
            return
        target = Twist()
        requested_x = float(msg.linear.x)
        if not bool(self.get_parameter("allow_reverse").value):
            requested_x = max(0.0, requested_x)
        target.linear.x = _clamp(
            requested_x, float(self.get_parameter("max_linear_x").value)
        )
        # Direct missions are forward/rotate only.  Lateral motion remains
        # exclusive to the bounded recovery source.
        target.linear.y = 0.0
        requested_angular_z = _clamp(
            float(msg.angular.z), float(self.get_parameter("max_angular_z").value)
        )
        target.angular.z = _clamp(
            _minimum_nonzero_magnitude(
                requested_angular_z,
                float(self.get_parameter("min_effective_angular_z").value),
            ),
            float(self.get_parameter("max_angular_z").value),
        )
        with self._lock:
            self._direct_target = target
            self._last_direct_input_ns = self.get_clock().now().nanoseconds

    def _on_recovery_cmd_vel(self, msg: Twist) -> None:
        """Accept bounded reverse and lateral motion from the recovery source."""
        if not math.isfinite(float(msg.linear.x)) or not math.isfinite(
            float(msg.linear.y)
        ):
            self.get_logger().error("Rejected non-finite recovery cmd_vel")
            return
        target = Twist()
        requested_x = float(msg.linear.x)
        if requested_x > 0.0:
            requested_x = 0.0  # recovery may only reverse, never drive forward
        target.linear.x = _clamp(
            requested_x,
            float(self.get_parameter("recovery_max_reverse").value),
        )
        requested_y = _clamp(
            float(msg.linear.y),
            float(self.get_parameter("max_linear_y").value),
        )
        target.linear.y = _minimum_nonzero_magnitude(
            requested_y,
            float(self.get_parameter("min_effective_linear_y").value),
        )
        with self._lock:
            self._recovery_target = target
            self._last_recovery_input_ns = self.get_clock().now().nanoseconds

    def _on_scan(self, msg: LaserScan) -> None:
        with self._lock:
            self._last_scan_ns = self.get_clock().now().nanoseconds
            self._latest_scan = msg

    def _on_odom(self, msg: Odometry) -> None:
        current_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        with self._lock:
            self._last_odom_ns = self.get_clock().now().nanoseconds
            previous_xy = self._last_odom_xy
            self._last_odom_xy = current_xy
            self._odom_xy = current_xy
            threshold = float(
                self.get_parameter("odom_jump_stop_distance_m").value
            )
            if previous_xy is None or threshold <= 0.0:
                return
            distance = math.hypot(
                current_xy[0] - previous_xy[0],
                current_xy[1] - previous_xy[1],
            )
            if (
                not self._enabled
                or not self._odom_jump_monitoring_enabled
                or not _odom_step_is_jump(
                    previous_xy, current_xy, threshold
                )
            ):
                return
            self._enabled = False
            self._target = None
            self._direct_target = None
            self._recovery_target = None
            self._last_input_ns = 0
            self._last_direct_input_ns = 0
            self._last_recovery_input_ns = 0
            self._stuck_window_start_ns = 0
            self._stuck_window_start_xy = None
            self._auto_recovery_until_ns = 0
            self._auto_recovery_direction = 0.0
            self._output = Twist()
            self._publisher.publish(self._output)
            self._last_gate_reason = "odom pose jump; output disarmed"
            self.get_logger().error(
                "Odometry pose jumped %.2fm (> %.2fm); output DISARMED. "
                "Rebuild/verify the map before enabling motion again."
                % (distance, threshold)
            )
            self._publish_status(self._last_gate_reason)

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._latest_costmap = msg
            self._last_costmap_ns = self.get_clock().now().nanoseconds

    def _on_navigation_status(self, msg: GoalStatusArray) -> None:
        active = any(
            status.status
            in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
            for status in msg.status_list
        )
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            if active and not self._navigation_active:
                self._navigation_active_since_ns = now_ns
                self._lethal_start_attempts = 0
            elif not active and self._navigation_active:
                self._navigation_active_since_ns = 0
            self._navigation_active = active

    def _on_enable(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        with self._lock:
            self._enabled = bool(request.data)
            # Never reuse a command received before an enable/disable transition.
            self._target = None
            self._direct_target = None
            self._recovery_target = None
            self._last_input_ns = 0
            self._last_direct_input_ns = 0
            self._last_recovery_input_ns = 0
            self._stuck_window_start_ns = 0
            self._stuck_window_start_xy = None
            self._auto_recovery_until_ns = 0
            self._auto_recovery_direction = 0.0
            self._lethal_start_attempts = 0
            self._last_output_ns = self.get_clock().now().nanoseconds
            self._output = Twist()
            self._publisher.publish(self._output)
        response.success = True
        response.message = "cmd_vel output enabled" if request.data else "cmd_vel output disabled"
        self.get_logger().warn(response.message)
        self._publish_status("enable_service" if request.data else "disable_service")
        return response

    def _on_odom_jump_monitoring(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """Pause only odometry discontinuity detection for a commanded posture."""
        with self._lock:
            self._odom_jump_monitoring_enabled = bool(request.data)
            self._last_odom_xy = None
        response.success = True
        response.message = (
            "odom jump monitoring enabled with a fresh baseline"
            if request.data
            else "odom jump monitoring paused for controlled posture"
        )
        self.get_logger().warn(response.message)
        self._publish_status("odom_jump_monitoring:%s" % bool(request.data))
        return response

    def _publish_status(self, reason: str) -> None:
        """Latched one-line snapshot of the gate state for operators.

        Event-driven (startup, enable/disable, monitoring toggles, odom-jump
        auto-disarm); TRANSIENT_LOCAL latching means a late reader still sees
        the latest state without waiting for the next event.
        """
        payload = {
            "enabled": self._enabled,
            "odom_jump_monitoring": self._odom_jump_monitoring_enabled,
            "reason": reason,
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(msg)

    def _on_timer(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            # Disabling publishes one zero in _on_enable(), then releases the
            # output topic.  Continuously publishing zeros while disabled would
            # overwrite another explicitly selected command source and makes
            # "disabled" misleading.
            if not self._enabled:
                self._last_output_ns = now_ns
                return

            dt = max((now_ns - self._last_output_ns) / 1_000_000_000.0, 0.0)
            self._last_output_ns = now_ns
            timeout_ns = int(
                float(self.get_parameter("input_timeout_sec").value) * 1_000_000_000
            )
            normal_fresh = (
                self._target is not None
                and self._last_input_ns > 0
                and now_ns - self._last_input_ns <= timeout_ns
            )
            direct_fresh = (
                self._direct_target is not None
                and self._last_direct_input_ns > 0
                and now_ns - self._last_direct_input_ns <= timeout_ns
            )
            recovery_timeout_ns = int(
                float(self.get_parameter("recovery_timeout_sec").value)
                * 1_000_000_000
            )
            recovery_fresh = (
                self._recovery_target is not None
                and self._last_recovery_input_ns > 0
                and now_ns - self._last_recovery_input_ns <= recovery_timeout_ns
            )
            scan_fresh = self._is_sensor_fresh(
                now_ns, self._last_scan_ns, "require_scan", "scan_timeout_sec"
            )
            odom_fresh = self._is_sensor_fresh(
                now_ns, self._last_odom_ns, "require_odom", "odom_timeout_sec"
            )
            self._finish_auto_recovery(now_ns)
            if not recovery_fresh and self._auto_recovery_until_ns == 0:
                self._update_lethal_start_recovery(
                    now_ns, scan_fresh, odom_fresh
                )
            if not recovery_fresh and self._auto_recovery_until_ns == 0:
                self._update_stuck_recovery(
                    now_ns, normal_fresh, scan_fresh, odom_fresh
                )
            auto_recovery_active = (
                self._auto_recovery_until_ns > now_ns
                and self._auto_recovery_direction != 0.0
            )
            fresh = recovery_fresh or direct_fresh or normal_fresh
            active_target = (
                self._recovery_target
                if recovery_fresh
                else (self._direct_target if direct_fresh else self._target)
            )

            if recovery_fresh:
                assert active_target is not None
                self._output.linear.x = _approach(
                    self._output.linear.x,
                    active_target.linear.x,
                    float(self.get_parameter("max_linear_accel").value) * dt,
                )
                self._output.linear.y = _approach(
                    self._output.linear.y,
                    active_target.linear.y,
                    float(self.get_parameter("max_linear_y_accel").value) * dt,
                )
                self._output.angular.z = 0.0
            elif auto_recovery_active:
                self._output.linear.x = 0.0
                self._output.linear.y = (
                    self._auto_recovery_direction
                    * float(self.get_parameter("stuck_recovery_speed_mps").value)
                )
                self._output.angular.z = 0.0
            elif not fresh or not scan_fresh or not odom_fresh:
                self._output = Twist()
            else:
                assert active_target is not None
                self._output.linear.x = _approach(
                    self._output.linear.x,
                    active_target.linear.x,
                    float(self.get_parameter("max_linear_accel").value) * dt,
                )
                self._output.linear.y = _approach(
                    self._output.linear.y,
                    active_target.linear.y,
                    float(self.get_parameter("max_linear_y_accel").value) * dt,
                )
                self._output.angular.z = _angular_output_step(
                    self._output.angular.z,
                    active_target.angular.z,
                    float(self.get_parameter("max_angular_accel").value),
                    dt,
                    bool(self.get_parameter("angular_passthrough").value),
                )
            self._publisher.publish(self._output)

            reason = ""
            # An automatic escape intentionally runs without a fresh Nav2
            # command in the lethal-start case.  Do not print the misleading
            # "forcing zero" diagnostic while a non-zero escape is active.
            if self._enabled and not auto_recovery_active:
                if not scan_fresh:
                    reason = "scan missing/stale"
                elif not odom_fresh:
                    reason = "odom missing/stale"
                elif not fresh:
                    reason = "cmd_vel_nav, direct, and recovery command missing/stale"
            if reason != self._last_gate_reason:
                if reason:
                    self.get_logger().error("Safety gate forcing zero: %s" % reason)
                elif self._enabled:
                    self.get_logger().info("Safety gate inputs healthy")
                self._last_gate_reason = reason

    def _finish_auto_recovery(self, now_ns: int) -> None:
        if not self._auto_recovery_until_ns:
            return
        if self._auto_recovery_until_ns > now_ns:
            return
        self._last_auto_recovery_end_ns = now_ns
        self._auto_recovery_until_ns = 0
        self._auto_recovery_direction = 0.0
        self._stuck_window_start_ns = 0
        self._stuck_window_start_xy = None

    def _start_auto_lateral_recovery(
        self, now_ns: int, direction: float, reason: str
    ) -> None:
        self._auto_recovery_direction = direction
        self._auto_recovery_until_ns = now_ns + int(
            float(self.get_parameter("stuck_recovery_duration_sec").value) * 1e9
        )
        self.get_logger().warn(
            "%s; starting automatic lateral recovery (%s)."
            % (reason, "left" if direction > 0.0 else "right")
        )

    def _update_lethal_start_recovery(
        self, now_ns: int, scan_fresh: bool, odom_fresh: bool
    ) -> None:
        """Recover when Nav2 cannot produce cmd_vel from a lethal start cell."""
        if not bool(self.get_parameter("lethal_start_recovery_enabled").value):
            return
        if not self._enabled or not self._navigation_active:
            return
        if not scan_fresh or not odom_fresh or self._odom_xy is None:
            return
        if self._navigation_active_since_ns == 0:
            return
        delay_ns = int(
            float(self.get_parameter("lethal_start_trigger_delay_sec").value) * 1e9
        )
        if now_ns - self._navigation_active_since_ns < delay_ns:
            return
        costmap_timeout_ns = int(
            float(self.get_parameter("global_costmap_timeout_sec").value) * 1e9
        )
        if (
            self._latest_costmap is None
            or self._last_costmap_ns <= 0
            or now_ns - self._last_costmap_ns > costmap_timeout_ns
        ):
            return
        cost = _cost_at_xy(
            self._latest_costmap, self._odom_xy[0], self._odom_xy[1]
        )
        threshold = int(self.get_parameter("lethal_start_cost_threshold").value)
        if cost is None or cost < threshold:
            self._lethal_start_attempts = 0
            return
        if self._lethal_start_attempts >= int(
            self.get_parameter("lethal_start_max_attempts").value
        ):
            return
        cooldown_ns = int(
            float(self.get_parameter("stuck_recovery_cooldown_sec").value) * 1e9
        )
        if now_ns - self._last_auto_recovery_end_ns < cooldown_ns:
            return
        direction = self._clearer_side_direction()
        if direction == 0.0:
            self.get_logger().warn(
                "Nav2 start cell is lethal (cost=%d), but neither lateral side "
                "is scan-clear; recovery withheld." % cost
            )
            return
        self._lethal_start_attempts += 1
        self._start_auto_lateral_recovery(
            now_ns,
            direction,
            "Nav2 start cell is lethal (cost=%d, attempt=%d)"
            % (cost, self._lethal_start_attempts),
        )

    def _update_stuck_recovery(
        self, now_ns: int, normal_fresh: bool, scan_fresh: bool, odom_fresh: bool
    ) -> None:
        """Start one bounded lateral pulse when a Nav2 command makes no progress.

        This lives in the final safety gate so RViz-issued goals are covered;
        Frontier's own recovery channel remains independent and takes priority.
        """
        if self._auto_recovery_until_ns > now_ns:
            return
        if not bool(self.get_parameter("stuck_recovery_enabled").value):
            return
        if not self._enabled or not normal_fresh or not scan_fresh or not odom_fresh:
            self._stuck_window_start_ns = 0
            self._stuck_window_start_xy = None
            return
        if self._target is None or self._odom_xy is None:
            return
        command_speed = math.hypot(self._target.linear.x, self._target.linear.y)
        if command_speed < float(self.get_parameter("stuck_command_min_speed_mps").value):
            self._stuck_window_start_ns = 0
            self._stuck_window_start_xy = None
            return
        cooldown_ns = int(
            float(self.get_parameter("stuck_recovery_cooldown_sec").value) * 1e9
        )
        if now_ns - self._last_auto_recovery_end_ns < cooldown_ns:
            return
        if self._stuck_window_start_ns == 0:
            self._stuck_window_start_ns = now_ns
            self._stuck_window_start_xy = self._odom_xy
            return
        elapsed = (now_ns - self._stuck_window_start_ns) / 1e9
        moved = math.hypot(
            self._odom_xy[0] - self._stuck_window_start_xy[0],
            self._odom_xy[1] - self._stuck_window_start_xy[1],
        )
        if elapsed < float(self.get_parameter("stuck_timeout_sec").value):
            return
        if moved >= float(self.get_parameter("stuck_min_progress_m").value):
            self._stuck_window_start_ns = now_ns
            self._stuck_window_start_xy = self._odom_xy
            return
        direction = self._clearer_side_direction()
        self._stuck_window_start_ns = 0
        self._stuck_window_start_xy = None
        if direction == 0.0:
            self.get_logger().warn(
                "Nav2 command made no progress, but neither lateral side is scan-clear; "
                "holding zero instead of forcing recovery."
            )
            return
        self._start_auto_lateral_recovery(
            now_ns,
            direction,
            "Nav2 command made no progress for %.1fs" % elapsed,
        )

    def _clearer_side_direction(self) -> float:
        """Return +1 (left), -1 (right), or 0 using side LaserScan sectors."""
        scan = self._latest_scan
        if scan is None:
            return 0.0
        left = []
        right = []
        for index, value in enumerate(scan.ranges):
            if not math.isfinite(float(value)):
                continue
            angle = float(scan.angle_min) + index * float(scan.angle_increment)
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if math.radians(45.0) <= angle <= math.radians(135.0):
                left.append(float(value))
            elif -math.radians(135.0) <= angle <= -math.radians(45.0):
                right.append(float(value))
        if not left or not right:
            return 0.0
        # The nearest return in each side sector is the collision constraint.
        # Using the farthest ray would incorrectly mark a partly blocked side
        # as clear merely because one ray passed the obstacle.
        left_clear = min(min(left), 10.0)
        right_clear = min(min(right), 10.0)
        threshold = float(self.get_parameter("stuck_recovery_side_clearance_m").value)
        if left_clear < threshold and right_clear < threshold:
            return 0.0
        if left_clear >= threshold and right_clear < threshold:
            return 1.0
        if right_clear >= threshold and left_clear < threshold:
            return -1.0
        return 1.0 if left_clear >= right_clear else -1.0

    def _is_sensor_fresh(
        self,
        now_ns: int,
        last_ns: int,
        required_parameter: str,
        timeout_parameter: str,
    ) -> bool:
        if not bool(self.get_parameter(required_parameter).value):
            return True
        timeout_ns = int(float(self.get_parameter(timeout_parameter).value) * 1_000_000_000)
        return last_ns > 0 and now_ns - last_ns <= timeout_ns


def main() -> None:
    rclpy.init()
    node = CmdVelSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
