#!/usr/bin/env python3
"""Print free, occupied and unknown cell counts for /navigation_map."""

from collections import Counter

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class MapCellCheck(Node):
    def __init__(self):
        super().__init__("navigation_map_cell_check")
        self.message = None
        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, "/navigation_map", self._receive, qos)

    def _receive(self, message):
        self.message = message


def main():
    rclpy.init()
    node = MapCellCheck()
    deadline = node.get_clock().now().nanoseconds + 5_000_000_000
    while rclpy.ok() and node.message is None:
        if node.get_clock().now().nanoseconds >= deadline:
            break
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.message is None:
        print("ERROR: no /navigation_map received within 5 seconds")
        result = 1
    else:
        counts = Counter(int(value) for value in node.message.data)
        total = len(node.message.data)
        unknown, free, occupied = counts[-1], counts[0], counts[100]
        print(
            "frame=%s total=%d unknown=%d (%.2f%%) free=%d (%.2f%%) "
            "occupied=%d (%.2f%%) known=%d"
            % (
                node.message.header.frame_id,
                total,
                unknown,
                100.0 * unknown / total,
                free,
                100.0 * free / total,
                occupied,
                100.0 * occupied / total,
                total - unknown,
            )
        )
        result = 0
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(result)


if __name__ == "__main__":
    main()
