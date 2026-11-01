#!/usr/bin/env python3
"""Start the competition mission from a verified GO2-W remote chord.

The default is a read-only dry run.  Actual service calls require the explicit
``--execute`` option.  The configured chord must be held with neutral sticks,
all readiness inputs must be fresh, and each process permits only one start
attempt so a held or bouncing button cannot retrigger the mission.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from unitree_go.msg import WirelessController


START_MASK_DEFAULT = 0x00C0  # Measured F1 + F2 on 2026-10-08.


def sticks_neutral(values: Tuple[float, float, float, float], limit: float) -> bool:
    return all(abs(float(value)) <= limit for value in values)


def start_chord_pressed(keys: int, mask: int) -> bool:
    # Exact equality rejects F1+F2 combined with another held button.
    return (int(keys) & 0xFFFF) == (int(mask) & 0xFFFF)


@dataclass
class StatusRecord:
    payload: Dict[str, Any] = field(default_factory=dict)
    received_at: float = 0.0

    def fresh(self, now: float, max_age: float) -> bool:
        return self.received_at > 0.0 and 0.0 <= now - self.received_at <= max_age


class CompetitionRemoteTrigger(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("competition_remote_trigger")
        self._execute = bool(args.execute)
        self._start_mask = int(args.start_mask, 0) & 0xFFFF
        self._hold_s = max(0.5, float(args.hold_seconds))
        self._neutral_limit = max(0.01, float(args.neutral_limit))
        self._status_max_age_s = max(0.5, float(args.status_max_age))
        self._controller_timeout_s = max(0.2, float(args.controller_timeout))

        self._controller_received_at = 0.0
        self._keys = 0
        self._sticks = (0.0, 0.0, 0.0, 0.0)
        self._pressed_since = 0.0
        self._release_armed = True
        self._attempted = False
        self._phase = "WAITING_CONTROLLER"
        self._detail = "no wireless-controller sample received"
        self._sequence_busy = False
        self._sequence_index = 0
        self._sequence_due = 0.0
        self._sequence: List[Tuple[str, Any, Callable[[], Any], float]] = []

        self._records = {
            "health": StatusRecord(),
            "guard": StatusRecord(),
            "mission": StatusRecord(),
            "coordinator": StatusRecord(),
            "gate1": StatusRecord(),
            "gate2": StatusRecord(),
        }

        live_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            WirelessController,
            "/wirelesscontroller",
            self._on_controller,
            live_qos,
        )
        for key, topic, qos in (
            ("health", "/rl2_nav/slam_health", live_qos),
            ("guard", "/rl2_nav/slam_pose_jump_guard/status", latched_qos),
            ("mission", "/mission_supervisor_node/status", latched_qos),
            ("coordinator", "/three_target_box_mission/status", latched_qos),
            ("gate1", "/cmd_vel_safety_node/status", latched_qos),
            ("gate2", "/go2w_sport_cmd_bridge/status", latched_qos),
        ):
            self.create_subscription(
                String,
                topic,
                lambda msg, record_key=key: self._on_status(record_key, msg),
                qos,
            )

        self._sport_enable = self.create_client(
            SetBool, "/go2w_sport_cmd_bridge/enable"
        )
        self._balance_stand = self.create_client(
            Trigger, "/go2w_sport_cmd_bridge/balance_stand"
        )
        self._safety_enable = self.create_client(
            SetBool, "/cmd_vel_safety_node/enable"
        )
        self._mission_start = self.create_client(
            Trigger, "/three_target_box_mission/start"
        )
        self._stop_move = self.create_client(
            Trigger, "/go2w_sport_cmd_bridge/stop_move"
        )
        self._mission_abort = self.create_client(
            Trigger, "/three_target_box_mission/abort"
        )

        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, "/competition_remote_trigger/status", status_qos
        )
        self.create_timer(0.05, self._tick)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().warn(
            "Remote trigger started in %s mode; chord=0x%04X, hold=%.1fs."
            % ("EXECUTE" if self._execute else "DRY-RUN", self._start_mask, self._hold_s)
        )

    def _on_controller(self, msg: WirelessController) -> None:
        self._controller_received_at = time.monotonic()
        self._keys = int(msg.keys) & 0xFFFF
        self._sticks = (
            float(msg.lx),
            float(msg.ly),
            float(msg.rx),
            float(msg.ry),
        )

    def _on_status(self, key: str, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._records[key] = StatusRecord(payload, time.monotonic())

    def _readiness_reasons(self, now: float) -> List[str]:
        reasons: List[str] = []
        # These four streams are periodic and must be fresh.  Gate status is
        # intentionally TRANSIENT_LOCAL and event-driven: after an enable or
        # disable it may remain quiet indefinitely, so receipt + live service
        # availability is the correct liveness check for gate1/gate2.
        for key in ("health", "guard", "mission", "coordinator"):
            if not self._records[key].fresh(now, self._status_max_age_s):
                reasons.append("%s_status_stale" % key)
        for key in ("gate1", "gate2"):
            if self._records[key].received_at <= 0.0:
                reasons.append("%s_status_missing" % key)
        if reasons:
            return reasons

        health = self._records["health"].payload
        guard = self._records["guard"].payload
        mission = self._records["mission"].payload
        coordinator = self._records["coordinator"].payload
        gate1 = self._records["gate1"].payload
        gate2 = self._records["gate2"].payload

        if not bool(health.get("slam_pose_ok", False)):
            reasons.append("slam_pose_not_ok")
        if not bool(health.get("navigation_map_ok", False)):
            reasons.append("navigation_map_not_ok")
        if str(guard.get("state", "")) != "MONITORING":
            reasons.append("slam_guard_not_monitoring")
        if not bool(guard.get("monitoring_enabled", False)):
            reasons.append("slam_guard_disabled")
        if str(mission.get("state", "")) not in ("IDLE", "DONE"):
            reasons.append("mission_not_idle")
        try:
            if int(mission.get("target_goal_count", 0)) != 3:
                reasons.append("target_goal_count_not_3")
        except (TypeError, ValueError):
            reasons.append("target_goal_count_invalid")
        if str(coordinator.get("phase", "")) not in (
            "IDLE",
            "COMPLETE",
            "ERROR_BOX_KEPT_CLOSED",
        ):
            reasons.append("coordinator_busy")
        if bool(gate1.get("enabled", False)):
            reasons.append("velocity_gate_already_enabled")
        if bool(gate2.get("enabled", False)):
            reasons.append("sport_gate_already_enabled")
        if not sticks_neutral(self._sticks, self._neutral_limit):
            reasons.append("sticks_not_neutral")
        return reasons

    @staticmethod
    def _set_bool_request(value: bool) -> SetBool.Request:
        request = SetBool.Request()
        request.data = bool(value)
        return request

    @staticmethod
    def _trigger_request() -> Trigger.Request:
        return Trigger.Request()

    def _required_services_ready(self) -> List[str]:
        missing = []
        for name, client in (
            ("sport_enable", self._sport_enable),
            ("balance_stand", self._balance_stand),
            ("safety_enable", self._safety_enable),
            ("mission_start", self._mission_start),
        ):
            if not client.service_is_ready():
                missing.append(name)
        return missing

    def _accept_chord(self, now: float) -> None:
        reasons = self._readiness_reasons(now)
        missing = self._required_services_ready()
        reasons.extend("service_unavailable:%s" % name for name in missing)
        if reasons:
            self._phase = "START_REJECTED"
            self._detail = ",".join(reasons)
            self.get_logger().error("Start chord rejected: %s" % self._detail)
            return
        if not self._execute:
            self._phase = "DRY_RUN_ACCEPTED"
            self._detail = "all checks passed; no service call made"
            self.get_logger().warn(self._detail)
            return

        self._sequence_busy = True
        self._sequence_index = 0
        self._sequence_due = now
        self._sequence = [
            (
                "enable_sport_gate",
                self._sport_enable,
                lambda: self._set_bool_request(True),
                0.3,
            ),
            (
                "request_balance_stand",
                self._balance_stand,
                self._trigger_request,
                1.0,
            ),
            (
                "enable_velocity_gate",
                self._safety_enable,
                lambda: self._set_bool_request(True),
                0.3,
            ),
            (
                "start_three_target_mission",
                self._mission_start,
                self._trigger_request,
                0.0,
            ),
        ]
        self._phase = "START_SEQUENCE"
        self._detail = "validated remote start accepted"
        self.get_logger().warn("Executing validated competition start sequence.")

    def _tick(self) -> None:
        now = time.monotonic()
        if self._sequence_busy:
            self._advance_sequence(now)
            return
        # One process represents one competition-start opportunity.  Preserve
        # its accepted/rejected terminal result even if the event-driven
        # controller stream becomes quiet after the operator releases keys.
        if self._attempted:
            return
        controller_fresh = (
            self._controller_received_at > 0.0
            and now - self._controller_received_at <= self._controller_timeout_s
        )
        if not controller_fresh:
            self._pressed_since = 0.0
            self._phase = "WAITING_CONTROLLER"
            self._detail = "wireless-controller stream stale"
            return

        pressed = start_chord_pressed(self._keys, self._start_mask)
        neutral = sticks_neutral(self._sticks, self._neutral_limit)
        if not pressed:
            self._pressed_since = 0.0
            if self._keys == 0:
                self._release_armed = True
                if not self._attempted:
                    self._phase = "SAFE_READY_CHECKING"
                    reasons = self._readiness_reasons(now)
                    self._detail = "ready" if not reasons else ",".join(reasons)
            return
        if not neutral:
            self._pressed_since = 0.0
            self._phase = "CHORD_BLOCKED"
            self._detail = "start chord held but sticks are not neutral"
            return
        if not self._release_armed or self._attempted:
            return
        if self._pressed_since <= 0.0:
            self._pressed_since = now
            self._phase = "CHORD_HOLDING"
            self._detail = "hold F1+F2 for %.1fs" % self._hold_s
            return
        held = now - self._pressed_since
        self._phase = "CHORD_HOLDING"
        self._detail = "held %.1f/%.1fs" % (held, self._hold_s)
        if held >= self._hold_s:
            self._attempted = True
            self._release_armed = False
            self._accept_chord(now)

    def _advance_sequence(self, now: float) -> None:
        if now < self._sequence_due or self._sequence_index >= len(self._sequence):
            return
        label, client, request_factory, delay_after = self._sequence[self._sequence_index]
        self._phase = "START_SEQUENCE"
        self._detail = label
        future = client.call_async(request_factory())
        self._sequence_due = float("inf")

        def done_callback(done_future) -> None:
            try:
                result = done_future.result()
                success = result is not None and bool(result.success)
                message = "no response" if result is None else str(result.message)
            except Exception as exc:  # noqa: BLE001
                success = False
                message = str(exc)
            if not success:
                self._fail_and_rollback("%s failed: %s" % (label, message))
                return
            self._sequence_index += 1
            if self._sequence_index >= len(self._sequence):
                self._sequence_busy = False
                self._phase = "MISSION_STARTED"
                self._detail = "all four start steps succeeded"
                self.get_logger().warn(self._detail)
                return
            self._sequence_due = time.monotonic() + delay_after

        future.add_done_callback(done_callback)

    def _fail_and_rollback(self, reason: str) -> None:
        self._sequence_busy = False
        self._phase = "START_FAILED_ROLLBACK"
        self._detail = reason
        self.get_logger().error(reason)
        if self._stop_move.service_is_ready():
            self._stop_move.call_async(self._trigger_request())
        if self._safety_enable.service_is_ready():
            self._safety_enable.call_async(self._set_bool_request(False))
        if self._sport_enable.service_is_ready():
            self._sport_enable.call_async(self._set_bool_request(False))
        if self._mission_abort.service_is_ready():
            self._mission_abort.call_async(self._trigger_request())

    def _publish_status(self) -> None:
        payload = {
            "phase": self._phase,
            "detail": self._detail,
            "mode": "execute" if self._execute else "dry_run",
            "start_mask": "0x%04X" % self._start_mask,
            "hold_seconds": self._hold_s,
            "keys": "0x%04X" % self._keys,
            "sticks_neutral": sticks_neutral(self._sticks, self._neutral_limit),
            "one_shot_attempted": self._attempted,
        }
        self._status_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow validated service calls; omitted means read-only dry-run",
    )
    parser.add_argument("--start-mask", default="0x00C0")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--neutral-limit", type=float, default=0.08)
    parser.add_argument("--status-max-age", type=float, default=2.5)
    parser.add_argument("--controller-timeout", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init(args=[])
    node = CompetitionRemoteTrigger(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
