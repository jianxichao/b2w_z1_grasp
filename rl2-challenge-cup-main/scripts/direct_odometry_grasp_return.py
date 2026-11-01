#!/usr/bin/env python3
"""Direct pose-feedback route: base -> dock -> arm grasp -> base.

This is deliberately separate from Nav2.  It never requests a global path or
uses occupancy costmaps to select a route.  It drives a straight, closed-loop
segment from /slam/current_pose to each supplied map-frame coordinate.

Safety remains outside this script: it publishes only /cmd_vel_raw, so both
cmd_vel_safety_node and go2w_sport_cmd_bridge must still be explicitly armed.
Those gates retain obstacle, stale-input, connection-loss and StopMove checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from std_msgs.msg import String
from std_srvs.srv import Trigger

from rl2_interfaces.action import GraspTarget


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_of(pose: PoseStamped) -> float:
    q = pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def make_pose(frame: str, x: float, y: float, yaw: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


@dataclass
class Latest:
    pose: Optional[PoseStamped] = None
    pose_received_s: float = 0.0
    arm_target: Optional[PoseStamped] = None
    arm_target_received_s: float = 0.0


class DirectMission:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        Path(self.args.status_log).parent.mkdir(parents=True, exist_ok=True)
        self.latest = Latest()
        self.node = rclpy.create_node("direct_odometry_grasp_return")
        # Use a dedicated gate input.  Publishing on /cmd_vel_raw would share
        # Nav2 controller/velocity-smoother plumbing and allow idle Nav2
        # output to interfere with this non-Nav2 route.
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel_direct", 10)
        self.status_pub = self.node.create_publisher(String, "/direct_grasp_mission/status", 10)
        self.node.create_subscription(PoseStamped, "/slam/current_pose", self._on_pose, 20)
        self.node.create_subscription(PoseStamped, "/robot_arm/target_pose", self._on_arm_target, 10)
        self.arm_client = ActionClient(self.node, GraspTarget, "/arm/grasp_target")
        self.arm_home_client = self.node.create_client(Trigger, "/arm/home")

    def _on_pose(self, msg: PoseStamped) -> None:
        if (msg.header.frame_id.strip() or self.args.frame) != self.args.frame:
            return
        self.latest.pose = copy.deepcopy(msg)
        self.latest.pose_received_s = time.monotonic()

    def _on_arm_target(self, msg: PoseStamped) -> None:
        if msg.header.frame_id.strip() != "arm_base":
            return
        self.latest.arm_target = copy.deepcopy(msg)
        self.latest.arm_target_received_s = time.monotonic()

    def _status(self, state: str, **extra: object) -> None:
        data: dict[str, object] = {"state": state, **extra}
        line = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(String(data=line))
        print(line, flush=True)
        try:
            with open(self.args.status_log, "a", encoding="utf-8") as stream:
                stream.write("%s %s\n" % (time.strftime("%F %T"), line))
        except OSError as exc:
            print("status_log_write_failed: %s" % exc, file=sys.stderr, flush=True)

    def _stop(self) -> None:
        zero = Twist()
        for _ in range(4):
            self.cmd_pub.publish(zero)
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.04)

    def _require_fresh_pose(self) -> PoseStamped:
        if self.latest.pose is None:
            raise RuntimeError("slam_pose_missing")
        age = time.monotonic() - self.latest.pose_received_s
        if age > self.args.pose_timeout_s:
            raise RuntimeError("slam_pose_stale:%.2fs" % age)
        return self.latest.pose

    def wait_for_pose(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.10)
            try:
                self._require_fresh_pose()
                return
            except RuntimeError:
                pass
        raise RuntimeError("slam_pose_not_ready")

    def drive_to(
        self,
        label: str,
        target: tuple[float, float, float],
        *,
        position_tolerance_m: float | None = None,
        require_final_yaw: bool = True,
    ) -> None:
        target_x, target_y, target_yaw = target
        position_tolerance_m = position_tolerance_m or self.args.position_tolerance_m
        self._status(label + "_started", target_x=target_x, target_y=target_y, target_yaw=target_yaw)
        deadline = time.monotonic() + self.args.drive_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.03)
            pose = self._require_fresh_pose()
            x, y = pose.pose.position.x, pose.pose.position.y
            yaw = yaw_of(pose)
            dx, dy = target_x - x, target_y - y
            distance = math.hypot(dx, dy)
            cmd = Twist()
            if distance > position_tolerance_m:
                heading = math.atan2(dy, dx)
                heading_error = wrap(heading - yaw)
                cmd.angular.z = clamp(
                    self.args.heading_kp * heading_error,
                    -self.args.max_angular_rad_s,
                    self.args.max_angular_rad_s,
                )
                # Rotate first when facing away; otherwise advance only along
                # the desired heading, keeping the route straight.
                if abs(heading_error) < self.args.forward_heading_limit_rad:
                    cmd.linear.x = min(
                        self.args.max_linear_mps,
                        self.args.distance_kp * distance,
                    ) * max(0.0, math.cos(heading_error))
            else:
                if not require_final_yaw:
                    self._stop()
                    self._status(label + "_arrived", x=x, y=y, yaw=yaw, distance_m=round(distance, 3), final_yaw_alignment=False)
                    return
                yaw_error = wrap(target_yaw - yaw)
                if abs(yaw_error) <= self.args.yaw_tolerance_rad:
                    self._stop()
                    self._status(label + "_arrived", x=x, y=y, yaw=yaw)
                    return
                cmd.angular.z = clamp(
                    self.args.heading_kp * yaw_error,
                    -self.args.max_angular_rad_s,
                    self.args.max_angular_rad_s,
                )
            self.cmd_pub.publish(cmd)
            self._status(label + "_driving", distance_m=round(distance, 3), command_x=round(cmd.linear.x, 3), command_yaw=round(cmd.angular.z, 3))
            time.sleep(0.05)
        self._stop()
        raise RuntimeError(label + "_timeout")

    def hold_still(self, seconds: float, state: str) -> None:
        """Keep publishing zero velocity while the operator fine-tunes the base."""
        self._status(state, duration_s=seconds)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            self._stop()
            time.sleep(0.05)
        self._status(state + "_finished")

    def wait_for_arm_target(self) -> PoseStamped:
        self._status("waiting_for_arm_target")
        deadline = time.monotonic() + self.args.arm_target_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.10)
            if (
                self.latest.arm_target is not None
                and time.monotonic() - self.latest.arm_target_received_s <= self.args.arm_target_max_age_s
            ):
                return copy.deepcopy(self.latest.arm_target)
        raise RuntimeError("arm_target_timeout")

    def home_and_hold_arm(self) -> None:
        """Enter the arm's position-control hold before the chassis moves.

        The dock arm starts with gravity compensation but no position command
        until its first motion.  Sending it to the known home pose at the
        stationary start point both verifies the Action runtime and leaves the
        SDK continuously holding the final joint command during navigation.
        """
        self._status("arm_home_started")
        if not self.arm_home_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("arm_home_service_unavailable")
        future = self.arm_home_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.args.arm_home_timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("arm_home_timeout")
        if not response.success:
            raise RuntimeError("arm_home_failed:%s" % response.message)
        self._status("arm_home_succeeded", message=response.message)

    def grasp(self, target: PoseStamped) -> None:
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("arm_action_unavailable")
        # The arm-mounted Gemini's current eye-in-hand estimate may carry a
        # small, repeatable depth-axis bias.  Keep the correction explicit at
        # the mission boundary instead of silently altering the shared vision
        # calibration used by standalone arm tests.
        raw_x = target.pose.position.x
        target.pose.position.x += self.args.arm_target_x_offset_m
        goal = GraspTarget.Goal()
        goal.target_pose = target
        goal.target_id = "direct_odometry_mission"
        self._status(
            "arm_grasp_requested",
            raw_x=raw_x,
            x_offset_m=self.args.arm_target_x_offset_m,
            x=target.pose.position.x,
            y=target.pose.position.y,
            z=target.pose.position.z,
        )
        send_future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=8.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("arm_goal_rejected")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=self.args.arm_action_timeout_s)
        result = result_future.result()
        if result is None:
            handle.cancel_goal_async()
            raise RuntimeError("arm_action_timeout")
        if not result.result.success:
            raise RuntimeError("arm_grasp_failed:%s" % result.result.message)
        self._status("arm_grasp_succeeded", message=result.result.message)

    def run(self) -> int:
        base = tuple(self.args.base)
        dock = tuple(self.args.dock)
        try:
            self._status("waiting_for_slam", base=base, dock=dock)
            self.wait_for_pose()
            if self.args.home_arm_before_drive:
                self.home_and_hold_arm()
            self.drive_to(
                "dock",
                dock,
                position_tolerance_m=self.args.dock_position_tolerance_m,
                require_final_yaw=False,
            )
            self.hold_still(self.args.dock_manual_settle_s, "dock_manual_adjustment")
            self.grasp(self.wait_for_arm_target())
            self.drive_to("return", base)
            self._status("completed")
            return 0
        except RuntimeError as exc:
            self._stop()
            self._status("failed", reason=str(exc))
            return 2
        finally:
            self._stop()
            self.node.destroy_node()
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct SLAM-pose drive -> grasp -> return; no Nav2 path planning.")
    parser.add_argument("--base", nargs=3, type=float, metavar=("X", "Y", "YAW"), required=True)
    parser.add_argument("--dock", nargs=3, type=float, metavar=("X", "Y", "YAW"), required=True)
    parser.add_argument("--frame", default="map")
    parser.add_argument("--max-linear-mps", type=float, default=0.20)
    parser.add_argument("--max-angular-rad-s", type=float, default=0.45)
    parser.add_argument("--position-tolerance-m", type=float, default=0.18)
    parser.add_argument(
        "--dock-position-tolerance-m",
        type=float,
        default=0.35,
        help="Stop this far from the dock coordinate; no final yaw spin is performed.",
    )
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.14)
    parser.add_argument("--forward-heading-limit-rad", type=float, default=0.45)
    parser.add_argument("--heading-kp", type=float, default=1.4)
    parser.add_argument("--distance-kp", type=float, default=0.6)
    parser.add_argument("--pose-timeout-s", type=float, default=0.6)
    parser.add_argument("--startup-timeout-s", type=float, default=10.0)
    parser.add_argument("--drive-timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--dock-manual-settle-s",
        type=float,
        default=7.0,
        help="Zero-velocity pause at the dock for operator radio fine adjustment before Gemini grasping.",
    )
    # The dock Gemini pipeline runs at a low, deliberate rate. Give it time
    # to acquire a stable arm-frame target after the base has settled instead
    # of abandoning an otherwise valid mission after only a few frames.
    parser.add_argument("--arm-target-timeout-s", type=float, default=30.0)
    parser.add_argument("--arm-target-max-age-s", type=float, default=1.0)
    parser.add_argument("--arm-action-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--arm-home-timeout-s",
        type=float,
        default=30.0,
        help="maximum time for the initial home-and-hold request",
    )
    parser.add_argument(
        "--home-arm-before-drive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "explicitly call /arm/home before driving (disabled by default: "
            "some ARX5 gripper firmware revisions enter protection when Home "
            "is repeated after startup)"
        ),
    )
    parser.add_argument(
        "--arm-target-x-offset-m",
        type=float,
        default=0.0,
        help="add this arm_base x correction to Gemini target before grasp",
    )
    parser.add_argument(
        "--status-log",
        default="/tmp/rl2_direct_grasp_mission.log",
        help="append machine-readable mission states here for post-stop diagnosis",
    )
    args = parser.parse_args()
    if args.max_linear_mps <= 0.0 or args.max_linear_mps > 0.30:
        parser.error("--max-linear-mps must be in (0, 0.30]")
    if args.max_angular_rad_s <= 0.0 or args.max_angular_rad_s > 0.60:
        parser.error("--max-angular-rad-s must be in (0, 0.60]")
    rclpy.init()
    return DirectMission(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
