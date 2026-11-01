import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1]
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from b2w_perception.pose_task_event_publisher import (
    build_events,
    load_results,
    result_to_task_event,
)


class PoseTaskEventPublisherTests(unittest.TestCase):
    def test_pose_ready_event(self):
        event = result_to_task_event(
            {"status": "ok", "in_grasp_range": False}
        )
        self.assertEqual(event, "target_pose_ready:")

    def test_grasp_range_event(self):
        event = result_to_task_event(
            {"status": "ok", "in_grasp_range": True}
        )
        self.assertEqual(event, "target_in_grasp_range:")

    def test_pose_failed_contains_specific_reason(self):
        statuses = [
            "laser_distance_unavailable",
            "too_few_laser_points",
            "scan_no_return",
            "invalid_bbox",
            "image_size_mismatch",
            "frame_id_mismatch",
        ]

        for status in statuses:
            with self.subTest(status=status):
                event = result_to_task_event({"status": status})
                self.assertEqual(event, f"pose_failed:{status}")

    def test_low_confidence_and_ignored_class_do_not_emit_events(self):
        for status in ["low_confidence", "ignored_class"]:
            with self.subTest(status=status):
                self.assertIsNone(
                    result_to_task_event({"status": status})
                )

    def test_unknown_or_empty_status_does_not_emit_event(self):
        self.assertIsNone(result_to_task_event({"status": "unknown"}))
        self.assertIsNone(result_to_task_event({}))

    def test_load_and_build_events(self):
        payload = [
            {"status": "ok", "in_grasp_range": False},
            {"status": "laser_distance_unavailable", "in_grasp_range": False},
            {"status": "ignored_class", "in_grasp_range": False},
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            results = load_results(path)

        self.assertEqual(
            build_events(results),
            [
                "target_pose_ready:",
                "pose_failed:laser_distance_unavailable",
            ],
        )


if __name__ == "__main__":
    unittest.main()
