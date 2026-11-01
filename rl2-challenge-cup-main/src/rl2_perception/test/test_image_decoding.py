import sys
import unittest
from pathlib import Path

import numpy as np
from sensor_msgs.msg import Image

PACKAGE_SRC = Path(__file__).resolve().parents[1]
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from rl2_perception.yolo_target_perception_node import (
    _depth_roi_bounds,
    _depth_image_to_array,
    _image_to_array,
    _stamps_within,
)


class ImageDecodingTests(unittest.TestCase):
    def test_rgb_decoder_ignores_row_padding(self):
        msg = Image(height=2, width=2, encoding="rgb8", step=8)
        msg.data = bytes([1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 88, 88])
        decoded = _image_to_array(msg)
        np.testing.assert_array_equal(
            decoded,
            np.array(
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                dtype=np.uint8,
            ),
        )

    def test_depth_decoder_ignores_row_padding(self):
        msg = Image(height=2, width=2, encoding="16UC1", step=6)
        rows = [
            np.array([1000, 2000], dtype="<u2").tobytes() + b"xx",
            np.array([3000, 4000], dtype="<u2").tobytes() + b"yy",
        ]
        msg.data = b"".join(rows)
        decoded = _depth_image_to_array(msg)
        np.testing.assert_array_equal(decoded, [[1000.0, 2000.0], [3000.0, 4000.0]])

    def test_rgb_depth_timestamp_tolerance(self):
        first = Image()
        second = Image()
        first.header.stamp.sec = 10
        first.header.stamp.nanosec = 100_000_000
        second.header.stamp.sec = 10
        second.header.stamp.nanosec = 250_000_000
        self.assertTrue(_stamps_within(first, second, 0.20))
        self.assertFalse(_stamps_within(first, second, 0.10))


    def test_depth_roi_scales_color_pixels_to_depth_resolution(self):
        bounds = _depth_roi_bounds(
            u=212.0,
            v=120.0,
            box_w=42.4,
            box_h=24.0,
            roi_scale=0.5,
            color_size=(424, 240),
            depth_size=(640, 480),
        )
        self.assertEqual(bounds, (304, 336, 228, 252))

    def test_depth_roi_keeps_equal_resolution_coordinates(self):
        bounds = _depth_roi_bounds(
            u=100.0,
            v=50.0,
            box_w=40.0,
            box_h=20.0,
            roi_scale=0.5,
            color_size=(640, 480),
            depth_size=(640, 480),
        )
        self.assertEqual(bounds, (90, 110, 45, 55))

    def test_depth_roi_rejects_invalid_dimensions(self):
        self.assertIsNone(
            _depth_roi_bounds(1, 1, 1, 1, 0.5, (0, 240), (640, 480))
        )


if __name__ == "__main__":
    unittest.main()
