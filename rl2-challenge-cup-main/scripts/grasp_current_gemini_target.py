#!/usr/bin/env python3
"""Send exactly one fresh arm-mounted Gemini target to the ARX5 Action.

This helper intentionally has no chassis publishers or safety-gate calls.  It
is for stationary, operator-supervised arm-only validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient

from rl2_interfaces.action import GraspTarget


def main() -> int:
    parser = argparse.ArgumentParser(description="grasp one fresh Gemini arm_base target")
    parser.add_argument("--target-timeout-s", type=float, default=8.0)
    parser.add_argument("--action-timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("grasp_current_gemini_target")
    latest: list[PoseStamped] = []

    def on_target(msg: PoseStamped) -> None:
        if msg.header.frame_id.strip() == "arm_base":
            latest[:] = [copy.deepcopy(msg)]

    node.create_subscription(PoseStamped, "/robot_arm/target_pose", on_target, 10)
    client = ActionClient(node, GraspTarget, "/arm/grasp_target")
    try:
        deadline = time.monotonic() + args.target_timeout_s
        while time.monotonic() < deadline and not latest:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not latest:
            print(json.dumps({"state": "failed", "reason": "fresh_gemini_target_timeout"}))
            return 2
        target = latest[0]
        print(json.dumps({
            "state": "target_locked",
            "x": target.pose.position.x,
            "y": target.pose.position.y,
            "z": target.pose.position.z,
        }), flush=True)
        if not client.wait_for_server(timeout_sec=5.0):
            print(json.dumps({"state": "failed", "reason": "arm_action_unavailable"}))
            return 2
        goal = GraspTarget.Goal()
        goal.target_pose = target
        goal.target_id = "operator_current_gemini_target"
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future, timeout_sec=8.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            print(json.dumps({"state": "failed", "reason": "arm_goal_rejected"}))
            return 2
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=args.action_timeout_s)
        result = result_future.result()
        if result is None:
            handle.cancel_goal_async()
            print(json.dumps({"state": "failed", "reason": "arm_action_timeout"}))
            return 2
        print(json.dumps({"state": "completed" if result.result.success else "failed", "message": result.result.message}))
        return 0 if result.result.success else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
