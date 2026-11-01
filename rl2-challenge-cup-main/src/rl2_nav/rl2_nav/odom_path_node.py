"""Accumulate an odometry stream into a bounded RViz Path."""

from __future__ import annotations

from collections import deque
import math
from typing import Deque, Optional, Tuple

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger


class OdomPathNode(Node):
    """Publish a decimated, bounded path from `/slam/odom`."""

    def __init__(self) -> None:
        super().__init__("odom_path_node")
        self.declare_parameter("odom_topic", "/slam/odom")
        self.declare_parameter("path_topic", "/slam/trajectory")
        self.declare_parameter("min_distance_m", 0.02)
        self.declare_parameter("max_points", 10000)

        self._min_distance_m = max(
            0.0, float(self.get_parameter("min_distance_m").value)
        )
        self._poses: Deque[PoseStamped] = deque(
            maxlen=max(2, int(self.get_parameter("max_points").value))
        )
        self._last_xy: Optional[Tuple[float, float]] = None
        self._frame_id = ""

        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            Path, str(self.get_parameter("path_topic").value), path_qos
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            20,
        )
        self.create_service(Trigger, "~/reset", self._on_reset)
        self.get_logger().info(
            "Odometry trajectory ready: %s -> %s"
            % (
                self.get_parameter("odom_topic").value,
                self.get_parameter("path_topic").value,
            )
        )

    def _on_odom(self, msg: Odometry) -> None:
        frame_id = msg.header.frame_id or "map"
        if self._frame_id and frame_id != self._frame_id:
            self._clear()
        self._frame_id = frame_id

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        if self._last_xy is not None and math.hypot(
            x - self._last_xy[0], y - self._last_xy[1]
        ) < self._min_distance_m:
            return

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = frame_id
        pose.pose = msg.pose.pose
        self._poses.append(pose)
        self._last_xy = (x, y)
        self._publish(msg.header.stamp)

    def _publish(self, stamp) -> None:
        path = Path()
        path.header.frame_id = self._frame_id or "map"
        path.header.stamp = stamp
        path.poses = list(self._poses)
        self._publisher.publish(path)

    def _clear(self) -> None:
        self._poses.clear()
        self._last_xy = None

    def _on_reset(self, _request, response):
        self._clear()
        response.success = True
        response.message = "trajectory cleared"
        self.get_logger().warn(response.message)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomPathNode()
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
