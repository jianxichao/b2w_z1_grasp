#!/usr/bin/env python3
"""Minimal live viewer for the arm-mounted Gemini annotated image.

Run from a terminal that has sourced the laptop Humble + project overlays.
Close with q or Esc.  This intentionally uses raw ROS Image messages, so it
does not require rqt_image_view or image_transport compressed plugins.
"""

from __future__ import annotations

import sys

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class GeminiLiveView(Node):
    def __init__(self) -> None:
        super().__init__("gemini_live_view")
        self._bridge = CvBridge()
        self._window = "Gemini annotated image (q/Esc to close)"
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        self.create_subscription(
            Image, "/robot_arm/annotated_image", self._on_image, 5
        )
        self.get_logger().info("Waiting for /robot_arm/annotated_image ...")

    def _on_image(self, message: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            cv2.imshow(self._window, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                rclpy.shutdown()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Cannot render Gemini image: %s" % exc)


def main() -> int:
    rclpy.init()
    node = GeminiLiveView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
