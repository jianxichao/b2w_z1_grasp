#!/usr/bin/env python3
"""Read-only check of the real inputs used by Frontier lateral escape."""

import json
import math
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

from rl2_nav.frontier_explorer_node import (
    choose_lateral_escape_direction,
    side_sector_clearances,
    world_to_cell,
)


class EscapeReadinessCheck(Node):
    def __init__(self):
        super().__init__("wall_escape_readiness_check")
        self.scan = None
        self.costmap = None
        self.cloud = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data
        )
        self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._on_costmap, 10
        )
        self.create_subscription(
            PointCloud2,
            "/unitree/slam_mapping/points",
            self._on_cloud,
            qos_profile_sensor_data,
        )

    def _on_scan(self, msg):
        self.scan = msg

    def _on_costmap(self, msg):
        self.costmap = msg

    def _on_cloud(self, msg):
        self.cloud = msg

    def result(self):
        if self.scan is None or self.costmap is None or self.cloud is None:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.costmap.header.frame_id or "map",
                "base_footprint",
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException:
            return None
        left, right = side_sector_clearances(
            self.scan.ranges,
            float(self.scan.angle_min),
            float(self.scan.angle_increment),
        )
        nearest_range = float("inf")
        nearest_angle = None
        front = float("inf")
        rear = float("inf")
        for index, raw_range in enumerate(self.scan.ranges):
            distance = float(raw_range)
            if not math.isfinite(distance) or distance <= 0.0:
                continue
            angle = float(self.scan.angle_min) + index * float(
                self.scan.angle_increment
            )
            if distance < nearest_range:
                nearest_range = distance
                nearest_angle = angle
            if -math.pi / 4.0 < angle < math.pi / 4.0:
                front = min(front, distance)
            elif angle > 3.0 * math.pi / 4.0 or angle < -3.0 * math.pi / 4.0:
                rear = min(rear, distance)
        robot_x = float(transform.transform.translation.x)
        robot_y = float(transform.transform.translation.y)
        quaternion = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        raw_nearby_points = 0
        raw_left = float("inf")
        raw_right = float("inf")
        for point in point_cloud2.read_points(
            self.cloud, field_names=("x", "y", "z"), skip_nans=True
        ):
            point_x, point_y, point_z = map(float, point)
            delta_x = point_x - robot_x
            delta_y = point_y - robot_y
            base_x = cos_yaw * delta_x + sin_yaw * delta_y
            base_y = -sin_yaw * delta_x + cos_yaw * delta_y
            distance = math.hypot(base_x, base_y)
            if not (-0.20 <= point_z <= 0.60 and 0.20 <= distance <= 8.0):
                continue
            raw_nearby_points += 1
            angle = math.atan2(base_y, base_x)
            if math.pi / 4.0 <= angle <= 3.0 * math.pi / 4.0:
                raw_left = min(raw_left, distance)
            elif -3.0 * math.pi / 4.0 <= angle <= -math.pi / 4.0:
                raw_right = min(raw_right, distance)
        col, row = world_to_cell(
            robot_x,
            robot_y,
            float(self.costmap.info.resolution),
            float(self.costmap.info.origin.position.x),
            float(self.costmap.info.origin.position.y),
        )
        in_bounds = (
            0 <= col < int(self.costmap.info.width)
            and 0 <= row < int(self.costmap.info.height)
        )
        robot_cost = (
            int(self.costmap.data[row * int(self.costmap.info.width) + col])
            if in_bounds
            else None
        )
        direction = choose_lateral_escape_direction(
            left, right, wall_trigger_m=0.55, destination_clearance_m=0.80
        )
        direction_name = "left" if direction > 0 else "right" if direction < 0 else "none"
        return {
            "robot_xy": [round(robot_x, 3), round(robot_y, 3)],
            "global_costmap_start_cost": robot_cost,
            "left_clearance_m": round(left, 3) if math.isfinite(left) else "inf",
            "right_clearance_m": round(right, 3) if math.isfinite(right) else "inf",
            "front_clearance_m": round(front, 3) if math.isfinite(front) else "inf",
            "rear_clearance_m": round(rear, 3) if math.isfinite(rear) else "inf",
            "nearest_range_m": (
                round(nearest_range, 3) if math.isfinite(nearest_range) else "inf"
            ),
            "nearest_angle_deg": (
                round(math.degrees(nearest_angle), 1)
                if nearest_angle is not None
                else None
            ),
            "raw_cloud_filtered_points": raw_nearby_points,
            "raw_cloud_left_m": (
                round(raw_left, 3) if math.isfinite(raw_left) else "inf"
            ),
            "raw_cloud_right_m": (
                round(raw_right, 3) if math.isfinite(raw_right) else "inf"
            ),
            "selected_escape": direction_name,
            "start_cost_triggers_escape": robot_cost is not None and robot_cost >= 99,
            "scan_selects_safe_side": direction != 0.0,
            "ready": robot_cost is not None and robot_cost >= 99 and direction != 0.0,
        }


def main():
    rclpy.init()
    node = EscapeReadinessCheck()
    deadline = time.monotonic() + 8.0
    result = None
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            result = node.result()
            if result is not None:
                break
        if result is None:
            print(json.dumps({"ready": False, "error": "scan/costmap/TF unavailable"}))
            raise SystemExit(2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ready"] else 1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
