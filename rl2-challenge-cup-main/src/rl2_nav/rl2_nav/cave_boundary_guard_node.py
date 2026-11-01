"""Overlay a switchable cave-entrance virtual wall on an OccupancyGrid."""

from __future__ import annotations

import copy
import json
import math
from typing import Optional

from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool


def gate_midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def overlay_segment_wall(
    data: list[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    a: tuple[float, float],
    b: tuple[float, float],
    thickness_m: float,
) -> list[int]:
    """Return a copy with the finite A-B segment rasterized as lethal cells."""
    output = list(data)
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if width <= 0 or height <= 0 or resolution <= 0.0 or length <= 1e-6:
        return output
    samples = max(1, int(math.ceil(length / (resolution * 0.35))))
    radius_cells = max(0, int(math.ceil(0.5 * thickness_m / resolution)))
    radius_sq = (0.5 * thickness_m + 0.5 * resolution) ** 2
    for index in range(samples + 1):
        ratio = index / samples
        x = a[0] + ratio * (b[0] - a[0])
        y = a[1] + ratio * (b[1] - a[1])
        cx = int(math.floor((x - origin_x) / resolution))
        cy = int(math.floor((y - origin_y) / resolution))
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                gx, gy = cx + dx, cy + dy
                if not (0 <= gx < width and 0 <= gy < height):
                    continue
                cell_x = origin_x + (gx + 0.5) * resolution
                cell_y = origin_y + (gy + 0.5) * resolution
                if (cell_x - x) ** 2 + (cell_y - y) ** 2 <= radius_sq:
                    output[gy * width + gx] = 100
    return output


class CaveBoundaryGuardNode(Node):
    def __init__(self) -> None:
        super().__init__("cave_boundary_guard_node")
        self.declare_parameter("input_map_topic", "/navigation_map_raw")
        self.declare_parameter("output_map_topic", "/navigation_map")
        self.declare_parameter("gate_topic", "/base_cave_mission/cave_gate")
        self.declare_parameter("enabled_on_start", False)
        self.declare_parameter("wall_thickness_m", 0.30)
        self.declare_parameter("minimum_gate_width_m", 0.20)

        self._enabled = bool(self.get_parameter("enabled_on_start").value)
        self._gate: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_pub = self.create_publisher(
            OccupancyGrid, str(self.get_parameter("output_map_topic").value), latched
        )
        self._status_pub = self.create_publisher(
            String, "/cave_boundary_guard/status", latched
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("input_map_topic").value),
            self._on_map,
            10,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter("gate_topic").value),
            self._on_gate,
            latched,
        )
        self.create_service(SetBool, "~/enable", self._set_enabled)
        self._publish_status("ready")

    def _on_gate(self, msg: PoseArray) -> None:
        if len(msg.poses) < 2:
            self.get_logger().error("Cave gate requires two endpoint poses")
            return
        a = (float(msg.poses[0].position.x), float(msg.poses[0].position.y))
        b = (float(msg.poses[1].position.x), float(msg.poses[1].position.y))
        minimum = float(self.get_parameter("minimum_gate_width_m").value)
        if not all(math.isfinite(v) for v in (*a, *b)) or math.dist(a, b) < minimum:
            self.get_logger().error("Rejected invalid or too-short cave gate")
            return
        self._gate = (a, b)
        self._publish_status("gate_updated")

    def _set_enabled(self, request, response):
        if request.data and self._gate is None:
            response.success = False
            response.message = "cannot enable: cave A/B endpoints unavailable"
            return response
        self._enabled = bool(request.data)
        response.success = True
        response.message = "cave virtual wall enabled" if self._enabled else "cave virtual wall disabled"
        self._publish_status("operator_request")
        return response

    def _on_map(self, msg: OccupancyGrid) -> None:
        if not self._enabled or self._gate is None:
            self._map_pub.publish(msg)
            return
        guarded = copy.deepcopy(msg)
        guarded.data = overlay_segment_wall(
            list(msg.data),
            int(msg.info.width),
            int(msg.info.height),
            float(msg.info.resolution),
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
            self._gate[0],
            self._gate[1],
            float(self.get_parameter("wall_thickness_m").value),
        )
        self._map_pub.publish(guarded)

    def _publish_status(self, reason: str) -> None:
        gate = self._gate
        payload = {
            "enabled": self._enabled,
            "reason": reason,
            "gate_a": gate[0] if gate else None,
            "gate_b": gate[1] if gate else None,
            "midpoint": gate_midpoint(*gate) if gate else None,
            "wall_thickness_m": float(self.get_parameter("wall_thickness_m").value),
        }
        self._status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CaveBoundaryGuardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
