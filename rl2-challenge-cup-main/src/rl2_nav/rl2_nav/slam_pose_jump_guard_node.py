"""Coordinate a fail-safe shutdown after an official SLAM frame jump.

The map bridge and velocity gate each have a local jump guard.  This node is
the higher-level coordinator: it cancels navigation, disables all motion
producers, clears state tied to the old map frame, and then latches until an
operator explicitly acknowledges that the new map is stable.
"""

import json
import math
from functools import partial
from typing import Dict, Optional

import rclpy
from action_msgs.srv import CancelGoal
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger


MONITORING = "MONITORING"
RECOVERING = "RECOVERING"
READY_FOR_MANUAL_REARM = "READY_FOR_MANUAL_REARM"


def planar_step(previous_xy, current_xy) -> float:
    """Return the planar distance between two odometry samples."""
    return math.hypot(
        float(current_xy[0]) - float(previous_xy[0]),
        float(current_xy[1]) - float(previous_xy[1]),
    )


def known_cell_count(msg: OccupancyGrid) -> int:
    """Count observed occupancy-grid cells (unknown cells are -1)."""
    return sum(1 for value in msg.data if int(value) >= 0)


class SlamPoseJumpGuard(Node):
    """Latch the navigation stack safe when the SLAM map frame resets."""

    def __init__(self) -> None:
        super().__init__("slam_pose_jump_guard")

        self.declare_parameter("odom_topic", "/slam/odom")
        self.declare_parameter("map_topic", "/navigation_map")
        self.declare_parameter("jump_distance_m", 1.0)
        self.declare_parameter("stable_step_distance_m", 0.10)
        self.declare_parameter("stable_duration_s", 3.0)
        self.declare_parameter("minimum_known_map_cells", 1000)
        self.declare_parameter("status_period_s", 1.0)
        self.declare_parameter("pose_stream_lost_after_s", 5.0)

        self._jump_distance_m = float(self.get_parameter("jump_distance_m").value)
        self._pose_stream_lost_after_s = float(
            self.get_parameter("pose_stream_lost_after_s").value
        )
        self._stable_step_distance_m = float(
            self.get_parameter("stable_step_distance_m").value
        )
        self._stable_duration_s = float(
            self.get_parameter("stable_duration_s").value
        )
        self._minimum_known_map_cells = int(
            self.get_parameter("minimum_known_map_cells").value
        )

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self.create_publisher(
            String, "/rl2_nav/slam_pose_jump_guard/status", status_qos
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            20,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("map_topic").value),
            self._on_map,
            10,
        )
        self.create_service(Trigger, "~/acknowledge", self._on_acknowledge)
        self.create_service(SetBool, "~/monitoring", self._on_monitoring)

        # Do not use Node's private ``_clients`` name: rclpy stores its own
        # client entity list there.
        self._service_clients = {
            "disable_frontier": self.create_client(
                SetBool, "/frontier_explorer_node/enable"
            ),
            "disable_velocity_gate": self.create_client(
                SetBool, "/cmd_vel_safety_node/enable"
            ),
            "disable_sport_output": self.create_client(
                SetBool, "/go2w_sport_cmd_bridge/enable"
            ),
            "cancel_nav2_goals": self.create_client(
                CancelGoal, "/navigate_to_pose/_action/cancel_goal"
            ),
            "reset_occupancy_grid": self.create_client(
                Trigger, "/official_occupancy_grid_bridge/reset"
            ),
            "clear_global_costmap": self.create_client(
                ClearEntireCostmap,
                "/global_costmap/clear_entirely_global_costmap",
            ),
            "clear_local_costmap": self.create_client(
                ClearEntireCostmap,
                "/local_costmap/clear_entirely_local_costmap",
            ),
        }

        self._state = MONITORING
        self._monitoring_enabled = True
        self._previous_xy: Optional[tuple] = None
        self._latest_odom_ns = 0
        self._stable_since_ns = 0
        self._jump_count = 0
        self._recovery_generation = 0
        self._last_jump_distance_m = 0.0
        self._map_received_after_jump = False
        self._known_cells = 0
        self._pose_stream_lost = False
        self._operations: Dict[str, Dict] = {}
        self._last_reason = "startup"

        self.create_timer(0.10, self._recovery_step)
        self.create_timer(
            float(self.get_parameter("status_period_s").value),
            self._publish_status,
        )
        self._publish_status()
        self.get_logger().info(
            "SLAM pose-jump guard monitoring %s (threshold %.2fm)."
            % (str(self.get_parameter("odom_topic").value), self._jump_distance_m)
        )

    def _on_odom(self, msg: Odometry) -> None:
        now_ns = self.get_clock().now().nanoseconds
        current_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        step = 0.0
        if self._previous_xy is not None:
            step = planar_step(self._previous_xy, current_xy)
        self._previous_xy = current_xy
        self._latest_odom_ns = now_ns

        if not self._monitoring_enabled:
            return

        if step > self._jump_distance_m:
            # A relocalizing source can emit several incompatible samples in
            # succession.  The first one has already stopped motion and
            # started all resets; restarting that fan-out for every later
            # sample only races costmap/reset services and delays recovery.
            if self._state == RECOVERING:
                self._last_jump_distance_m = step
                self._stable_since_ns = 0
                self._last_reason = "additional_pose_jump_while_recovering"
                self.get_logger().warn(
                    "Ignoring additional %.3fm SLAM discontinuity while the "
                    "existing recovery is still active." % step,
                    throttle_duration_sec=1.0,
                )
                self._publish_status()
            else:
                self._begin_recovery(step)
            return

        if self._state == RECOVERING:
            if step <= self._stable_step_distance_m:
                if self._stable_since_ns == 0:
                    self._stable_since_ns = now_ns
            else:
                self._stable_since_ns = 0

    def _on_map(self, msg: OccupancyGrid) -> None:
        if self._state != RECOVERING:
            return
        self._map_received_after_jump = True
        self._known_cells = known_cell_count(msg)

    def _begin_recovery(self, step_m: float) -> None:
        self._state = RECOVERING
        self._recovery_generation += 1
        self._jump_count += 1
        self._last_jump_distance_m = step_m
        self._stable_since_ns = 0
        self._map_received_after_jump = False
        self._known_cells = 0
        # A jump can only fire while the odom stream is alive, so every new
        # recovery starts from a live stream.
        self._pose_stream_lost = False
        self._last_reason = "pose_jump_detected"
        self._operations = {
            name: {"future": None, "done": False, "attempts": 0, "error": ""}
            for name in self._service_clients
        }
        self.get_logger().fatal(
            "SLAM pose jumped %.3fm. Disabling motion, canceling navigation, "
            "and clearing old-frame maps. Manual re-arm will be required."
            % step_m
        )
        self._publish_status()
        # Dispatch on the same callback turn when services are already visible.
        self._dispatch_pending_operations()

    @staticmethod
    def _request_for(name: str):
        if name.startswith("disable_"):
            request = SetBool.Request()
            request.data = False
            return request
        if name == "cancel_nav2_goals":
            # Zero UUID and zero timestamp means cancel every active goal.
            return CancelGoal.Request()
        if name == "reset_occupancy_grid":
            return Trigger.Request()
        return ClearEntireCostmap.Request()

    def _dispatch_pending_operations(self) -> None:
        for name, operation in self._operations.items():
            if operation["done"] or operation["future"] is not None:
                continue
            client = self._service_clients[name]
            if not client.service_is_ready():
                continue
            operation["attempts"] += 1
            future = client.call_async(self._request_for(name))
            operation["future"] = future
            future.add_done_callback(
                partial(
                    self._operation_finished,
                    name,
                    self._recovery_generation,
                )
            )

    def _operation_finished(self, name: str, generation: int, future) -> None:
        if generation != self._recovery_generation:
            return
        operation = self._operations.get(name)
        if operation is None:
            return
        operation["future"] = None
        try:
            response = future.result()
            success = bool(getattr(response, "success", True))
            if name == "cancel_nav2_goals":
                # ERROR_NONE=0. ERROR_UNKNOWN_GOAL_ID=2 is also safe: there was
                # no live goal left to cancel.
                success = int(response.return_code) in (0, 2)
            if success:
                operation["done"] = True
                operation["error"] = ""
                if name == "reset_occupancy_grid":
                    # A map received before reset completion still belongs to
                    # the old accumulation.  Require a subsequent map update.
                    self._map_received_after_jump = False
                    self._known_cells = 0
            else:
                operation["error"] = str(getattr(response, "message", "failed"))
        except Exception as exc:  # pragma: no cover - middleware failure path
            operation["error"] = str(exc)

    def _recovery_step(self) -> None:
        if self._state != RECOVERING:
            return
        self._dispatch_pending_operations()

        now_ns = self.get_clock().now().nanoseconds
        odom_fresh = (
            self._latest_odom_ns > 0
            and (now_ns - self._latest_odom_ns) / 1e9 <= 1.0
        )
        odom_age_s = (
            (now_ns - self._latest_odom_ns) / 1e9
            if self._latest_odom_ns > 0
            else float("inf")
        )
        if (
            not self._pose_stream_lost
            and odom_age_s > self._pose_stream_lost_after_s
        ):
            # The dock-side unitree_slam died (or stopped publishing) after
            # the jump.  Nothing on this PC can rebuild the map frame until
            # it is restarted, so say so instead of waiting silently in
            # RECOVERING forever.
            self._pose_stream_lost = True
            self._last_reason = "pose_stream_lost"
            self.get_logger().fatal(
                "Official SLAM pose stream has been silent for %.1fs. Restart "
                "unitree_slam on the dock (./scripts/ensure_dock_slam_stack.sh "
                "--restart) and start a new mapping session; recovery stays "
                "latched until the new map frame is stable." % odom_age_s
            )
            self._publish_status()
        stable_for_s = (
            (now_ns - self._stable_since_ns) / 1e9
            if self._stable_since_ns > 0
            else 0.0
        )
        operations_done = bool(self._operations) and all(
            item["done"] for item in self._operations.values()
        )
        map_ready = (
            self._map_received_after_jump
            and self._known_cells >= self._minimum_known_map_cells
        )
        if (
            operations_done
            and odom_fresh
            and stable_for_s >= self._stable_duration_s
            and map_ready
        ):
            self._state = READY_FOR_MANUAL_REARM
            self._last_reason = "new_map_stable_manual_rearm_required"
            # Reaching this state required fresh odometry and a rebuilt map,
            # so the earlier stream loss has been recovered from.
            self._pose_stream_lost = False
            self.get_logger().warn(
                "New SLAM frame is stable. Motion remains disabled; inspect RViz, "
                "call ~/acknowledge, then explicitly re-enable both motion gates."
            )
            self._publish_status()

    def _on_acknowledge(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if self._state != READY_FOR_MANUAL_REARM:
            response.success = False
            response.message = "guard is not ready for acknowledgement"
            return response
        self._state = MONITORING
        self._last_reason = "operator_acknowledged"
        self._stable_since_ns = 0
        self._operations = {}
        response.success = True
        response.message = (
            "jump acknowledged; motion gates remain disabled and must be enabled manually"
        )
        self._publish_status()
        return response

    def _on_monitoring(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """Pause only jump detection during a commanded whole-body posture.

        Recovery cannot be bypassed with this service. Re-enabling clears the
        comparison baseline so the first post-posture sample cannot be
        compared with the lidar pose from the previous body height.
        """
        if self._state != MONITORING:
            response.success = False
            response.message = "cannot change monitoring while guard is latched"
            return response
        self._monitoring_enabled = bool(request.data)
        self._previous_xy = None
        self._stable_since_ns = 0
        self._last_reason = (
            "monitoring_resumed_after_controlled_posture"
            if request.data
            else "monitoring_paused_for_controlled_posture"
        )
        response.success = True
        response.message = self._last_reason
        self._publish_status()
        self.get_logger().warn(response.message)
        return response

    def _publish_status(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        stable_for_s = (
            max(0.0, (now_ns - self._stable_since_ns) / 1e9)
            if self._stable_since_ns > 0
            else 0.0
        )
        pending = [
            name for name, item in self._operations.items() if not item["done"]
        ]
        payload = {
            "state": self._state,
            "monitoring_enabled": self._monitoring_enabled,
            "reason": self._last_reason,
            "pose_stream_lost": self._pose_stream_lost,
            "jump_count": self._jump_count,
            "last_jump_distance_m": round(self._last_jump_distance_m, 3),
            "stable_for_s": round(stable_for_s, 2),
            "map_received_after_jump": self._map_received_after_jump,
            "known_map_cells": self._known_cells,
            "pending_operations": pending,
            "motion_rearm_allowed": self._state == READY_FOR_MANUAL_REARM,
            "motion_gates_are_automatically_enabled": False,
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamPoseJumpGuard()
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
