"""Health checks for the official B2W SLAM and navigation bridge."""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String

from b2w_interfaces.msg import UnitreeNavStatus


class B2WHealthCheckNode(Node):
    def __init__(self) -> None:
        super().__init__("b2w_health_check_node")

        self.declare_parameter("current_pose_topic", "/slam/current_pose")
        self.declare_parameter("navigation_map_topic", "/navigation_map")
        self.declare_parameter("arrived_topic", "/unitree_slam/arrived")
        self.declare_parameter("command_status_topic", "/unitree_slam/command_status")
        self.declare_parameter("health_topic", "/b2w_nav/slam_health")
        self.declare_parameter("check_rate_hz", 2.0)
        self.declare_parameter("input_timeout_s", 2.0)
        self.declare_parameter("min_known_cells", 200)
        self.declare_parameter("min_known_percent", 0.05)

        self._pose_ok = False
        self._map_ok = False
        self._last_status = ""
        self._last_arrived = ""
        self._last_pose_time = 0.0
        self._last_map_time = 0.0
        self._known_cells = 0
        self._known_percent = 0.0

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("current_pose_topic").value),
            self._on_pose,
            10,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("navigation_map_topic").value),
            self._on_map,
            10,
        )
        self.create_subscription(
            UnitreeNavStatus,
            str(self.get_parameter("arrived_topic").value),
            self._on_arrived,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("command_status_topic").value),
            self._on_command_status,
            10,
        )

        self._health_pub = self.create_publisher(
            String,
            str(self.get_parameter("health_topic").value),
            10,
        )
        rate = float(self.get_parameter("check_rate_hz").value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_health)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose_ok = msg.header.frame_id != ""
        self._last_pose_time = self.get_clock().now().nanoseconds / 1e9

    def _on_map(self, msg: OccupancyGrid) -> None:
        total = len(msg.data)
        self._known_cells = sum(1 for value in msg.data if int(value) >= 0)
        self._known_percent = 100.0 * self._known_cells / total if total else 0.0
        self._map_ok = (
            msg.info.width > 0
            and msg.info.height > 0
            and self._known_cells >= int(self.get_parameter("min_known_cells").value)
            and self._known_percent >= float(self.get_parameter("min_known_percent").value)
        )
        self._last_map_time = self.get_clock().now().nanoseconds / 1e9

    def _on_arrived(self, msg: UnitreeNavStatus) -> None:
        self._last_arrived = json.dumps(
            {"request_id": msg.request_id, "is_arrived": msg.is_arrived},
            separators=(",", ":"),
        )

    def _on_command_status(self, msg: String) -> None:
        self._last_status = msg.data

    def _publish_health(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        timeout = float(self.get_parameter("input_timeout_s").value)
        pose_age = now - self._last_pose_time if self._last_pose_time else -1.0
        map_age = now - self._last_map_time if self._last_map_time else -1.0
        payload = {
            "slam_pose_ok": self._pose_ok and 0.0 <= pose_age <= timeout,
            "navigation_map_ok": self._map_ok and 0.0 <= map_age <= timeout,
            "pose_age_s": round(pose_age, 3),
            "map_age_s": round(map_age, 3),
            "map_known_cells": self._known_cells,
            "map_known_percent": round(self._known_percent, 4),
            "last_command_status": self._last_status,
            "last_arrival": self._last_arrived,
        }
        self._health_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = B2WHealthCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
