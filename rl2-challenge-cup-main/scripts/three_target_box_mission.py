#!/usr/bin/env python3
"""Coordinate a three-target posture mission with the storage box.

The node is inert after startup.  Calling ``~/start`` closes the box first and
only starts the already-running mission supervisor after that command is
accepted.  After three targets, it waits for the supervisor's verified
return-to-base completion and a stable pose window before opening the box.

The storage-box controller has command acknowledgement but no position sensor;
the published status therefore never claims physical lid confirmation.
"""

from __future__ import annotations

from collections import deque
import json
import math
import time
from typing import Deque, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


# Only events that end the mission are allowed to disarm the coordinator.
# Several recoverable supervisor events deliberately contain the word
# ``failed`` (for example return pre-alignment falling back to normal Nav2,
# an approach candidate failing, or target posture failing).  Treating every
# such event as terminal leaves the box closed even when the robot later
# reaches base successfully.
_TERMINAL_MISSION_EVENTS = frozenset(
    {
        "mission_aborted",
        "mission_stopped",
        "mission_stopped_after_target_interaction",
        "mission_slam_lost",
        "mission_slam_lost_abort",
        "mission_motion_gate_closed",
        "mission_motion_gate_closed_abort",
        "mission_scan_motion_stalled",
        "mission_scan_motion_stalled_abort",
        "mission_return_failed",
        "mission_arm_manual_reset_required",
        "mission_failed",
    }
)

# These are explicit operator-requested stops, not coordinator faults.  The
# box staying closed is the intended safe result and the next Start request
# may reuse the coordinator directly.
_OPERATOR_STOP_EVENTS = frozenset({"mission_aborted", "mission_stopped"})


def mission_event_is_terminal_fault(event: str) -> bool:
    """Return whether a task event means the coordinated mission has ended."""
    event_name = str(event).strip().split(":", 1)[0]
    return event_name in _TERMINAL_MISSION_EVENTS


class ThreeTargetBoxMission(Node):
    def __init__(self) -> None:
        super().__init__("three_target_box_mission")
        self.declare_parameter("target_count", 3)
        self.declare_parameter("base_tolerance_m", 0.40)
        self.declare_parameter("stable_hold_s", 2.0)
        self.declare_parameter("stable_displacement_m", 0.03)
        self.declare_parameter("pose_max_age_s", 1.0)

        self._target_count = max(1, int(self.get_parameter("target_count").value))
        self._base_tolerance_m = max(
            0.05, float(self.get_parameter("base_tolerance_m").value)
        )
        self._stable_hold_s = max(
            0.5, float(self.get_parameter("stable_hold_s").value)
        )
        self._stable_displacement_m = max(
            0.005, float(self.get_parameter("stable_displacement_m").value)
        )
        self._pose_max_age_s = max(
            0.2, float(self.get_parameter("pose_max_age_s").value)
        )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self.create_publisher(
            String, "/three_target_box_mission/status", qos
        )
        self.create_subscription(
            String,
            "/mission_supervisor_node/status",
            self._on_mission_status,
            qos,
        )
        self.create_subscription(String, "/task_event", self._on_task_event, 50)
        self.create_subscription(
            PoseStamped, "/slam/current_pose", self._on_pose, 30
        )

        self._box_close = self.create_client(Trigger, "/storage_box/close")
        self._box_open = self.create_client(Trigger, "/storage_box/open")
        self._mission_start = self.create_client(
            Trigger, "/mission_supervisor_node/start"
        )
        self._mission_abort = self.create_client(
            Trigger, "/mission_supervisor_node/abort"
        )
        self._stop_move = self.create_client(
            Trigger, "/go2w_sport_cmd_bridge/stop_move"
        )

        self.create_service(Trigger, "~/start", self._start)
        self.create_service(Trigger, "~/abort", self._abort)
        self.create_service(Trigger, "~/reset", self._reset)
        self.create_timer(0.2, self._tick)

        self._phase = "IDLE"
        self._detail = "coordinator ready; no command sent"
        self._mission_payload: dict = {}
        self._last_event = ""
        self._armed = False
        self._return_confirmed = False
        self._stop_requested = False
        self._opening_requested = False
        self._poses: Deque[Tuple[float, float, float]] = deque(maxlen=200)
        self._publish_status()
        self.get_logger().info(
            "Three-target box coordinator ready and inert; call ~/start explicitly."
        )

    def _set_phase(self, phase: str, detail: str) -> None:
        self._phase = phase
        self._detail = detail
        self._publish_status()
        self.get_logger().info("%s: %s" % (phase, detail))

    def _publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "phase": self._phase,
                "detail": self._detail,
                "armed": self._armed,
                "return_confirmed": self._return_confirmed,
                "mission_state": self._mission_payload.get("state", "UNKNOWN"),
                "reached_targets": self._mission_payload.get("reached_targets", 0),
                "target_goal_count": self._mission_payload.get(
                    "target_goal_count", 0
                ),
                "last_event": self._last_event,
                "box_position_physically_confirmed": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._status_pub.publish(msg)

    def _on_mission_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._mission_payload = payload
        if self._armed and str(payload.get("state", "")) not in ("IDLE", "DONE"):
            if self._phase in ("STARTING_MISSION", "MISSION_RUNNING"):
                self._phase = "MISSION_RUNNING"
                self._detail = "mission active; storage box remains commanded closed"
        if self._armed and self._mission_success_payload(payload):
            self._return_confirmed = True
            if self._phase not in ("OPENING_BOX", "COMPLETE"):
                self._phase = "WAITING_BASE_STABLE"
                self._detail = "base arrival confirmed; waiting for stable pose"
        self._publish_status()

    def _on_task_event(self, msg: String) -> None:
        event = msg.data.strip()
        self._last_event = event
        if self._armed and event.startswith("mission_arrived_base:"):
            self._return_confirmed = True
            if self._phase not in ("OPENING_BOX", "COMPLETE"):
                self._set_phase(
                    "WAITING_BASE_STABLE",
                    "mission_arrived_base received; waiting for stable pose",
                )
                return
        elif self._armed and mission_event_is_terminal_fault(event):
            self._armed = False
            event_name = event.split(":", 1)[0]
            if event_name in _OPERATOR_STOP_EVENTS:
                phase = "ABORTED_BOX_KEPT_CLOSED"
                detail = "operator stop observed; box remains closed: %s" % event
            else:
                # The mission failed, but the coordinator itself behaved
                # correctly and remains reusable.  Preserve the cause without
                # mislabelling the coordinator as broken.
                phase = "INTERRUPTED_BOX_KEPT_CLOSED"
                detail = "mission interrupted; box remains closed: %s" % event
            self._set_phase(
                phase,
                detail,
            )
            return
        self._publish_status()

    def _on_pose(self, msg: PoseStamped) -> None:
        stamp = self.get_clock().now().nanoseconds / 1e9
        self._poses.append(
            (stamp, float(msg.pose.position.x), float(msg.pose.position.y))
        )
        cutoff = stamp - max(5.0, self._stable_hold_s + 1.0)
        while self._poses and self._poses[0][0] < cutoff:
            self._poses.popleft()

    def _mission_success_payload(self, payload: dict) -> bool:
        try:
            return (
                str(payload.get("state", "")) == "DONE"
                and int(payload.get("reached_targets", 0)) >= self._target_count
                and int(payload.get("target_goal_count", 0)) == self._target_count
            )
        except (TypeError, ValueError):
            return False

    def _at_base(self) -> bool:
        robot = self._mission_payload.get("robot_xy")
        base = self._mission_payload.get("base_xy")
        if not (
            isinstance(robot, list)
            and isinstance(base, list)
            and len(robot) >= 2
            and len(base) >= 2
        ):
            return False
        try:
            return math.hypot(
                float(robot[0]) - float(base[0]),
                float(robot[1]) - float(base[1]),
            ) <= self._base_tolerance_m
        except (TypeError, ValueError):
            return False

    def _pose_stable(self) -> bool:
        if not self._poses:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._poses[-1][0] > self._pose_max_age_s:
            return False
        window = [p for p in self._poses if p[0] >= now - self._stable_hold_s]
        if len(window) < 2 or window[0][0] > now - self._stable_hold_s + 0.3:
            return False
        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) <= (
            self._stable_displacement_m
        )

    def _start(self, _request, response):
        mission_state = str(self._mission_payload.get("state", "UNKNOWN"))
        goal_count = int(self._mission_payload.get("target_goal_count", 0) or 0)
        # ABORTED_BOX_KEPT_CLOSED is a safe terminal state, not an active
        # operation.  A new explicit start request may reuse it: all mission
        # flags are cleared below and the box is commanded closed again before
        # the supervisor is started.  Requiring a separate reset here made the
        # normal abort -> start workflow fail with "coordinator busy".
        if self._phase not in (
            "IDLE",
            "COMPLETE",
            "ERROR_BOX_KEPT_CLOSED",
            "ABORTED_BOX_KEPT_CLOSED",
            "INTERRUPTED_BOX_KEPT_CLOSED",
        ):
            response.success = False
            response.message = "coordinator busy: %s" % self._phase
            return response
        if mission_state not in ("IDLE", "DONE"):
            response.success = False
            response.message = "mission supervisor is not idle: %s" % mission_state
            return response
        if goal_count != self._target_count:
            response.success = False
            response.message = "expected target_goal_count=%d, got %d" % (
                self._target_count,
                goal_count,
            )
            return response
        for client, label in (
            (self._box_close, "/storage_box/close"),
            (self._mission_start, "/mission_supervisor_node/start"),
        ):
            if not client.wait_for_service(timeout_sec=0.3):
                response.success = False
                response.message = "required service unavailable: %s" % label
                return response
        self._armed = False
        self._return_confirmed = False
        self._stop_requested = False
        self._opening_requested = False
        self._set_phase("CLOSING_BOX", "closing box before mission start")
        future = self._box_close.call_async(Trigger.Request())
        future.add_done_callback(self._on_box_closed)
        response.success = True
        response.message = (
            "start sequence accepted; mission starts only after box-close acknowledgement"
        )
        return response

    def _on_box_closed(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "box close call failed: %s" % exc)
            return
        if result is None or not result.success:
            detail = "no response" if result is None else result.message
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "box close rejected: %s" % detail)
            return
        self._set_phase("STARTING_MISSION", "box close accepted; starting mission")
        future = self._mission_start.call_async(Trigger.Request())
        future.add_done_callback(self._on_mission_started)

    def _on_mission_started(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "mission start call failed: %s" % exc)
            return
        if result is None or not result.success:
            detail = "no response" if result is None else result.message
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "mission start rejected: %s" % detail)
            return
        self._armed = True
        self._set_phase(
            "MISSION_RUNNING",
            "three-target mission started; storage box remains commanded closed",
        )

    def _tick(self) -> None:
        if not self._armed or not self._return_confirmed:
            return
        if not self._mission_success_payload(self._mission_payload):
            return
        if not self._at_base():
            self._set_phase("WAITING_BASE_POSITION", "DONE received but base tolerance not met")
            return
        if not self._stop_requested:
            if self._stop_move.service_is_ready():
                self._stop_move.call_async(Trigger.Request())
            self._stop_requested = True
            self._set_phase("WAITING_BASE_STABLE", "stop requested; verifying stable pose")
            return
        if not self._pose_stable() or self._opening_requested:
            return
        if not self._box_open.wait_for_service(timeout_sec=0.1):
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "storage-box open service unavailable")
            self._armed = False
            return
        self._opening_requested = True
        self._set_phase("OPENING_BOX", "base arrival and stable stop verified")
        future = self._box_open.call_async(Trigger.Request())
        future.add_done_callback(self._on_box_opened)

    def _on_box_opened(self, future) -> None:
        self._armed = False
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "box open call failed: %s" % exc)
            return
        if result is None or not result.success:
            detail = "no response" if result is None else result.message
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "box open rejected: %s" % detail)
            return
        self._set_phase(
            "COMPLETE",
            "open command accepted after verified base stop; physical lid feedback unavailable",
        )

    def _abort(self, _request, response):
        self._armed = False
        if self._stop_move.service_is_ready():
            self._stop_move.call_async(Trigger.Request())
        if not self._mission_abort.wait_for_service(timeout_sec=0.3):
            self._set_phase("ERROR_BOX_KEPT_CLOSED", "mission abort service unavailable")
            response.success = False
            response.message = self._detail
            return response
        self._mission_abort.call_async(Trigger.Request())
        self._set_phase("ABORTED_BOX_KEPT_CLOSED", "abort requested; box not opened")
        response.success = True
        response.message = "abort requested; storage box remains closed"
        return response

    def _reset(self, _request, response):
        if self._armed:
            response.success = False
            response.message = "cannot reset while mission is armed; abort first"
            return response
        self._return_confirmed = False
        self._stop_requested = False
        self._opening_requested = False
        self._set_phase("IDLE", "coordinator reset; no actuator command sent")
        response.success = True
        response.message = "reset to IDLE"
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThreeTargetBoxMission()
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
