#!/usr/bin/env python3
"""Publish one explicit base + dock pose for the navigation/arm/return task.

The mission supervisor owns Nav2 and the arm handoff.  This program only
arms the next task by publishing two PoseStamped messages; it never publishes
velocity or enables either motion gate.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient


def pose(frame: str, x: float, y: float, yaw: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start one base -> dock -> Gemini grasp -> return mission."
    )
    parser.add_argument("--base", nargs=3, type=float, metavar=("X", "Y", "YAW"), required=True)
    parser.add_argument("--dock", nargs=3, type=float, metavar=("X", "Y", "YAW"), required=True)
    parser.add_argument("--frame", default="map")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument(
        "--skip-path-preflight",
        action="store_true",
        help="Skip the no-motion Nav2 path check (normally unsafe and unnecessary).",
    )
    parser.add_argument("--path-timeout-s", type=float, default=8.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("manual_grasp_return_starter")
    base_pub = node.create_publisher(PoseStamped, "/manual_grasp_base", 10)
    goal_pub = node.create_publisher(PoseStamped, "/manual_grasp_goal", 10)
    try:
        goal = pose(args.frame, *args.dock)
        if not args.skip_path_preflight:
            planner = ActionClient(node, ComputePathToPose, "/compute_path_to_pose")
            timeout_s = max(0.1, args.path_timeout_s)
            if not planner.wait_for_server(timeout_sec=timeout_s):
                raise RuntimeError("path_preflight_failed: /compute_path_to_pose unavailable")
            plan_request = ComputePathToPose.Goal()
            plan_request.goal = goal
            plan_request.use_start = False
            send_future = planner.send_goal_async(plan_request)
            rclpy.spin_until_future_complete(node, send_future, timeout_sec=timeout_s)
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("path_preflight_failed: planner rejected goal")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_s)
            result = result_future.result()
            if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
                status = "timeout" if result is None else str(result.status)
                raise RuntimeError("path_preflight_failed: planner status=%s" % status)
            path_size = len(result.result.path.poses)
            if path_size < 2:
                raise RuntimeError("path_preflight_failed: empty path")
            print("path_preflight_passed poses=%d" % path_size)
        # Publish the base repeatedly before the goal because they are
        # independent DDS topics and discovery may still be settling.
        base = pose(args.frame, *args.base)
        for _ in range(5):
            base.header.stamp = node.get_clock().now().to_msg()
            base_pub.publish(base)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.10)
        time.sleep(max(0.0, args.settle_s))
        goal.header.stamp = node.get_clock().now().to_msg()
        goal_pub.publish(goal)
        rclpy.spin_once(node, timeout_sec=0.10)
        print(
            "manual_grasp_return_started "
            f"base=({args.base[0]:.3f},{args.base[1]:.3f},{args.base[2]:.3f}) "
            f"dock=({args.dock[0]:.3f},{args.dock[1]:.3f},{args.dock[2]:.3f})"
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
