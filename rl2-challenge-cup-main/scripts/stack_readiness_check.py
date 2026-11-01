#!/usr/bin/env python3
"""Bounded ROS readiness checks for stack startup and mission start.

This tool never enables a motion gate, starts a mission, or publishes motion.
In ``startup`` mode it may call ``/storage_box/probe``; that operation only
checks the dock serial transport and does not move the servo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, Iterable, List, Tuple

import rclpy
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger


LATCHED_TOPICS = {
    "supervisor": "/mission_supervisor_node/status",
    "coordinator": "/three_target_box_mission/status",
    "guard": "/rl2_nav/slam_pose_jump_guard/status",
    "gate1": "/cmd_vel_safety_node/status",
    "gate2": "/go2w_sport_cmd_bridge/status",
}
HEALTH_TOPIC = "/rl2_nav/slam_health"
SAFE_COORDINATOR_PHASES = {
    "IDLE",
    "COMPLETE",
    "ERROR_BOX_KEPT_CLOSED",
    "ABORTED_BOX_KEPT_CLOSED",
    "INTERRUPTED_BOX_KEPT_CLOSED",
}


def validate_snapshots(
    snapshots: Dict[str, dict], *, target_count: int, require_gates_enabled: bool
) -> List[str]:
    """Return actionable readiness failures for already-decoded snapshots."""
    failures: List[str] = []
    required_names = ["supervisor", "guard", "gate1", "gate2", "health"]
    if int(target_count) == 3:
        required_names.append("coordinator")
    for name in required_names:
        if name not in snapshots:
            failures.append("missing status topic: %s" % name)
    if failures:
        return failures

    supervisor = snapshots["supervisor"]
    guard = snapshots["guard"]
    gate1 = snapshots["gate1"]
    gate2 = snapshots["gate2"]
    health = snapshots["health"]

    state = str(supervisor.get("state", "UNKNOWN"))
    if state not in ("IDLE", "DONE"):
        failures.append("mission supervisor is busy: %s" % state)
    try:
        actual_count = int(supervisor.get("target_goal_count", 0))
    except (TypeError, ValueError):
        actual_count = 0
    if actual_count != int(target_count):
        failures.append(
            "target count mismatch: expected %d, got %d"
            % (int(target_count), actual_count)
        )

    if int(target_count) == 3:
        coordinator = snapshots["coordinator"]
        phase = str(coordinator.get("phase", "UNKNOWN"))
        armed = bool(coordinator.get("armed", False))
        if armed or phase not in SAFE_COORDINATOR_PHASES:
            failures.append(
                "three-target coordinator is not reusable: phase=%s armed=%s"
                % (phase, armed)
            )

    if str(guard.get("state", "UNKNOWN")) != "MONITORING":
        failures.append("SLAM jump guard is not MONITORING")
    if not bool(guard.get("monitoring_enabled", False)):
        failures.append("SLAM jump monitoring is disabled")
    if bool(guard.get("pose_stream_lost", False)):
        failures.append("SLAM pose stream is lost")

    if not bool(health.get("slam_pose_ok", False)):
        failures.append("SLAM pose is not healthy")
    if not bool(health.get("navigation_map_ok", False)):
        failures.append("navigation map is not healthy")

    expected = bool(require_gates_enabled)
    for label, payload in (("Gate-1", gate1), ("Gate-2", gate2)):
        enabled = bool(payload.get("enabled", False))
        if enabled != expected:
            failures.append(
                "%s must be %s" % (label, "enabled" if expected else "disabled")
            )
    return failures


def _qos(*, latched: bool) -> QoSProfile:
    profile = QoSProfile(depth=1 if latched else 10)
    profile.reliability = ReliabilityPolicy.RELIABLE
    profile.durability = (
        DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE
    )
    return profile


def collect_snapshots(node: Node, timeout_s: float) -> Dict[str, dict]:
    snapshots: Dict[str, dict] = {}
    subscriptions = []

    def callback(name: str):
        def receive(msg: String) -> None:
            try:
                value = json.loads(msg.data)
            except (TypeError, ValueError, json.JSONDecodeError):
                return
            if isinstance(value, dict):
                snapshots[name] = value

        return receive

    for name, topic in LATCHED_TOPICS.items():
        subscriptions.append(
            node.create_subscription(String, topic, callback(name), _qos(latched=True))
        )
    subscriptions.append(
        node.create_subscription(
            String, HEALTH_TOPIC, callback("health"), _qos(latched=False)
        )
    )
    deadline = time.monotonic() + max(0.2, timeout_s)
    expected = len(LATCHED_TOPICS) + 1
    while len(snapshots) < expected and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    for subscription in subscriptions:
        node.destroy_subscription(subscription)
    return snapshots


def call_trigger(node: Node, service: str, timeout_s: float) -> Tuple[bool, str]:
    client = node.create_client(Trigger, service)
    if not client.wait_for_service(timeout_sec=timeout_s):
        node.destroy_client(client)
        return False, "service unavailable"
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        node.destroy_client(client)
        return False, "response timeout"
    try:
        response = future.result()
    except Exception as exc:  # noqa: BLE001
        node.destroy_client(client)
        return False, "service exception: %s" % exc
    node.destroy_client(client)
    if response is None:
        return False, "empty response"
    return bool(response.success), str(response.message)


def services_available(
    node: Node, services: Iterable[Tuple[str, object]], timeout_s: float
) -> List[str]:
    missing = []
    for name, service_type in services:
        client = node.create_client(service_type, name)
        if not client.wait_for_service(timeout_sec=timeout_s):
            missing.append(name)
        node.destroy_client(client)
    return missing


def lifecycle_is_active(node: Node, name: str, timeout_s: float) -> Tuple[bool, str]:
    service = "/%s/get_state" % name.strip("/")
    client = node.create_client(GetState, service)
    if not client.wait_for_service(timeout_sec=timeout_s):
        node.destroy_client(client)
        return False, "get_state service unavailable"
    future = client.call_async(GetState.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        node.destroy_client(client)
        return False, "get_state response timeout"
    try:
        response = future.result()
    except Exception as exc:  # noqa: BLE001
        node.destroy_client(client)
        return False, "get_state exception: %s" % exc
    node.destroy_client(client)
    if response is None:
        return False, "empty get_state response"
    label = str(response.current_state.label)
    return label == "active", label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("startup", "mission-start"))
    parser.add_argument("--target-count", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = Node("rl2_stack_readiness_check")
    try:
        # Do not decide from the first health sample.  Immediately after API
        # 1801, the pose may arrive before the accumulated map reaches its
        # minimum known-cell threshold.  Keep sampling until the whole
        # snapshot is coherent or the bounded deadline expires.
        deadline = time.monotonic() + max(0.5, args.timeout)
        failures: List[str] = ["status snapshots not received"]
        while time.monotonic() < deadline:
            snapshots = collect_snapshots(
                node, min(1.0, max(0.1, deadline - time.monotonic()))
            )
            failures = validate_snapshots(
                snapshots,
                target_count=max(1, args.target_count),
                require_gates_enabled=args.mode == "mission-start",
            )
            if not failures:
                break
            time.sleep(0.1)

        required_services = [
            ("/mission_supervisor_node/start", Trigger),
            ("/mission_supervisor_node/abort", Trigger),
            ("/cmd_vel_safety_node/enable", SetBool),
            ("/go2w_sport_cmd_bridge/enable", SetBool),
        ]
        if args.target_count == 3:
            required_services.extend(
                [
                    ("/three_target_box_mission/start", Trigger),
                    ("/three_target_box_mission/abort", Trigger),
                    ("/three_target_box_mission/reset", Trigger),
                    ("/storage_box/open", Trigger),
                    ("/storage_box/close", Trigger),
                    ("/storage_box/probe", Trigger),
                ]
            )
        for name in services_available(node, required_services, 0.5):
            failures.append("required service unavailable: %s" % name)

        action = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        if not action.wait_for_server(timeout_sec=1.0):
            failures.append("Nav2 action server unavailable: /navigate_to_pose")
        action.destroy()

        for lifecycle_node in ("planner_server", "controller_server", "bt_navigator"):
            active, detail = lifecycle_is_active(node, lifecycle_node, 1.0)
            if not active:
                failures.append(
                    "Nav2 lifecycle node %s is not active: %s"
                    % (lifecycle_node, detail)
                )

        if args.target_count == 3 and not failures:
            ok, detail = call_trigger(node, "/storage_box/probe", args.timeout)
            if not ok:
                failures.append("storage-box probe failed: %s" % detail)

        if failures:
            for failure in failures:
                print("NOT_READY: %s" % failure, flush=True)
            return 1
        print(
            "READY_CHECK_OK: mode=%s target_count=%d"
            % (args.mode, args.target_count),
            flush=True,
        )
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
