import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_SRC = Path(__file__).resolve().parents[1]
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from rl2_perception.onnx_yolo import (
    decode_yolo_output,
    letterbox_square,
    load_class_names,
    preprocess_rgb,
)


class OnnxYoloTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        boxed, scale, pad_x, pad_y = letterbox_square(image, 640)
        self.assertEqual(boxed.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 1.0)
        self.assertEqual((pad_x, pad_y), (0, 80))

        wide = np.zeros((300, 900, 3), dtype=np.uint8)
        boxed, scale, pad_x, pad_y = letterbox_square(wide, 640)
        self.assertEqual(boxed.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 640 / 900, places=6)
        self.assertEqual((pad_x, pad_y), (0, (640 - 214) // 2))

    def test_preprocess_rgb_builds_normalized_chw_blob(self):
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        blob, scale, pad_x, pad_y = preprocess_rgb(rgb, 640)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)
        self.assertLessEqual(float(blob.max()), 1.0)
        self.assertGreaterEqual(float(blob.min()), 0.0)
        self.assertGreater(scale, 0.0)
        self.assertGreaterEqual(pad_x, 0)
        self.assertGreaterEqual(pad_y, 0)

    def test_decode_accepts_channels_first_output(self):
        output = np.zeros((1, 5, 8400), dtype=np.float32)
        output[0, 4, 10] = 0.9  # class score at anchor 10
        output[0, 0, 10] = 100.0  # x center
        output[0, 1, 10] = 50.0  # y center
        output[0, 2, 10] = 40.0  # width
        output[0, 3, 10] = 20.0  # height
        results = decode_yolo_output(
            output,
            class_names=["rock_obstacle"],
            confidence_threshold=0.35,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            image_size=(640, 480),
        )
        self.assertEqual(len(results), 1)
        class_name, confidence, u, v, box_w, box_h = results[0]
        self.assertEqual(class_name, "rock_obstacle")
        self.assertAlmostEqual(confidence, 0.9)
        self.assertAlmostEqual(u, 100.0)
        self.assertAlmostEqual(v, 50.0)
        self.assertAlmostEqual(box_w, 40.0)
        self.assertAlmostEqual(box_h, 20.0)

    def test_decode_filters_below_threshold(self):
        output = np.zeros((8400, 5), dtype=np.float32)
        output[0, 4] = 0.1
        results = decode_yolo_output(
            output,
            class_names=["rock_obstacle"],
            confidence_threshold=0.35,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            image_size=(640, 480),
        )
        self.assertEqual(results, [])

    def test_decode_applies_nms_to_duplicate_boxes(self):
        output = np.zeros((8400, 5), dtype=np.float32)
        # Anchor 0: high-confidence box at (100, 50).
        output[0] = [100.0, 50.0, 40.0, 20.0, 0.9]
        # Anchor 1: lower-confidence overlapping duplicate.
        output[1] = [102.0, 51.0, 42.0, 22.0, 0.8]
        # Anchor 2: separate box far away.
        output[2] = [400.0, 300.0, 30.0, 30.0, 0.7]
        results = decode_yolo_output(
            output,
            class_names=["rock_obstacle"],
            confidence_threshold=0.35,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            image_size=(640, 480),
        )
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0][1], 0.9)
        self.assertAlmostEqual(results[0][2], 100.0)
        self.assertAlmostEqual(results[1][1], 0.7)

    def test_load_class_names_accepts_list_or_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names.json"
            path.write_text('["rock_obstacle"]', encoding="utf-8")
            self.assertEqual(load_class_names(path), ["rock_obstacle"])
            path.write_text('{"names": ["rock_obstacle"]}', encoding="utf-8")
            self.assertEqual(load_class_names(path), ["rock_obstacle"])


if __name__ == "__main__":
    unittest.main()
