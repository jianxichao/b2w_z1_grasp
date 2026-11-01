#!/usr/bin/env python3
"""Bounded Nav2 yaw-alignment probe with CSV telemetry.

This tool never arms either motion gate.  The operator must arm them separately
after checking the field.  It always cancels its goal on timeout or shutdown.
"""

import argparse
import csv
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class AngularAlignmentProbe(Node):
    def __init__(self, relative_yaw_deg, timeout_s, output_path, mode, forward_m):
        super().__init__("angular_alignment_probe")
        self.relative_yaw = math.radians(relative_yaw_deg)
        self.timeout_s = timeout_s
        self.output_path = output_path
        self.mode = mode
        self.forward_m = forward_m
        self.odom = None
        self.raw_cmd = Twist()
        self.nav_cmd = Twist()
        self.sport_cmd = Twist()
        self.goal_handle = None
        self.result_future = None
        self.start_monotonic = None
        self.start_yaw = None
        self.target_yaw = None
        self.target_x = None
        self.target_y = None
        self.finished = False
        self.result_received_monotonic = None
        self.outcome = "not_started"
        self.rows = []
        action_type = Spin if mode == "spin" else NavigateToPose
        action_name = "/spin" if mode == "spin" else "/navigate_to_pose"
        self.client = ActionClient(self, action_type, action_name)
        self.create_subscription(
            Odometry, "/slam/odom", self._on_odom, qos_profile_sensor_data
        )
        self.create_subscription(Twist, "/cmd_vel_raw", self._raw, 10)
        self.create_subscription(Twist, "/cmd_vel_nav", self._nav, 10)
        self.create_subscription(Twist, "/cmd_vel_sport", self._sport, 10)
        self.create_timer(0.05, self._sample)

    def _on_odom(self, msg):
        self.odom = msg

    def _raw(self, msg):
        self.raw_cmd = msg

    def _nav(self, msg):
        self.nav_cmd = msg

    def _sport(self, msg):
        self.sport_cmd = msg

    def start(self):
        if self.odom is None:
            raise RuntimeError("/slam/odom unavailable")
        if not self.client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("/navigate_to_pose unavailable")
        pose = self.odom.pose.pose
        self.start_yaw = yaw_from_quaternion(pose.orientation)
        self.target_yaw = normalize_angle(self.start_yaw + self.relative_yaw)
        self.target_x = float(pose.position.x) + self.forward_m * math.cos(
            self.start_yaw
        )
        self.target_y = float(pose.position.y) + self.forward_m * math.sin(
            self.start_yaw
        )
        if self.mode == "spin":
            goal = Spin.Goal()
            goal.target_yaw = float(self.relative_yaw)
            goal.time_allowance.sec = int(math.ceil(self.timeout_s))
        else:
            goal = NavigateToPose.Goal()
            goal.pose = PoseStamped()
            goal.pose.header.frame_id = "map"
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = self.target_x
            goal.pose.pose.position.y = self.target_y
            goal.pose.pose.orientation.z = math.sin(self.target_yaw / 2.0)
            goal.pose.pose.orientation.w = math.cos(self.target_yaw / 2.0)
        self.start_monotonic = time.monotonic()
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)
        self.outcome = "goal_sent"
        self.get_logger().warn(
            "Sent %s goal: forward=%.2fm start_yaw=%.1fdeg "
            "target_yaw=%.1fdeg delta=%.1fdeg"
            % (
                self.mode,
                self.forward_m,
                math.degrees(self.start_yaw),
                math.degrees(self.target_yaw),
                math.degrees(self.relative_yaw),
            )
        )

    def _goal_response(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.outcome = "rejected"
            self.finished = True
            return
        self.result_future = self.goal_handle.get_result_async()
        self.result_future.add_done_callback(self._goal_result)

    def _goal_result(self, future):
        self.outcome = "action_status_%d" % int(future.result().status)
        # Keep sampling for two seconds after Nav2 reports completion so coast,
        # overshoot and stable parking are visible in the same CSV.
        self.result_received_monotonic = time.monotonic()

    def _sample(self):
        if self.start_monotonic is None or self.odom is None:
            return
        elapsed = time.monotonic() - self.start_monotonic
        yaw = yaw_from_quaternion(self.odom.pose.pose.orientation)
        position_error = math.hypot(
            self.target_x - float(self.odom.pose.pose.position.x),
            self.target_y - float(self.odom.pose.pose.position.y),
        )
        self.rows.append(
            [
                elapsed,
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                position_error,
                math.degrees(yaw),
                math.degrees(normalize_angle(self.target_yaw - yaw)),
                self.raw_cmd.linear.x,
                self.nav_cmd.linear.x,
                self.sport_cmd.linear.x,
                self.raw_cmd.angular.z,
                self.nav_cmd.angular.z,
                self.sport_cmd.angular.z,
            ]
        )
        if elapsed >= self.timeout_s and not self.finished:
            self.outcome = "timeout_cancelled"
            self.cancel()
            self.finished = True
        elif (
            self.result_received_monotonic is not None
            and time.monotonic() - self.result_received_monotonic >= 2.0
        ):
            self.finished = True

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "elapsed_s",
                    "odom_x_m",
                    "odom_y_m",
                    "position_error_m",
                    "odom_yaw_deg",
                    "remaining_yaw_deg",
                    "cmd_raw_x",
                    "cmd_nav_x",
                    "cmd_sport_x",
                    "cmd_raw_yaw",
                    "cmd_nav_yaw",
                    "cmd_sport_yaw",
                ]
            )
            writer.writerows(self.rows)
        final_position_error = self.rows[-1][3] if self.rows else float("nan")
        final_yaw_error = self.rows[-1][5] if self.rows else float("nan")
        print(
            "outcome=%s samples=%d final_position_error_m=%.3f "
            "final_yaw_error_deg=%.2f csv=%s"
            % (
                self.outcome,
                len(self.rows),
                final_position_error,
                final_yaw_error,
                self.output_path,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative-yaw-deg", type=float, default=90.0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("spin", "navigate"), default="spin")
    parser.add_argument("--forward-m", type=float, default=0.0)
    parser.add_argument("--confirm-motion", action="store_true")
    args = parser.parse_args()
    if abs(args.relative_yaw_deg) > 180.0:
        raise SystemExit("relative yaw must be within [-180, 180] degrees")
    if args.forward_m < 0.0 or args.forward_m > 0.50:
        raise SystemExit("forward distance must be in [0.0, 0.50] m")
    if not args.confirm_motion:
        print("DRY PREVIEW ONLY: no ROS node or motion command was created.")
        print("requested: mode=%s forward=%.2fm yaw_delta=%.1fdeg timeout=%.1fs" %
              (args.mode, args.forward_m, args.relative_yaw_deg, args.timeout_s))
        return 0
    rclpy.init()
    node = AngularAlignmentProbe(
        args.relative_yaw_deg,
        max(1.0, args.timeout_s),
        args.output,
        args.mode,
        args.forward_m,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        deadline = time.monotonic() + 5.0
        while node.odom is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        node.start()
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        node.outcome = "operator_cancelled"
        node.cancel()
    finally:
        # Give Nav2 up to two seconds to process asynchronous cancellation.
        for _ in range(40):
            executor.spin_once(timeout_sec=0.05)
            if node.result_future is not None and node.result_future.done():
                break
        node.save()
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.outcome == "action_status_4" else 1


if __name__ == "__main__":
    raise SystemExit(main())
