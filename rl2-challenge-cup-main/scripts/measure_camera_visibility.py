#!/usr/bin/env python3
"""Collect one stationary target-visibility sample without commanding motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEPTH_PATTERN = re.compile(r"^depth_m=([-+0-9.eE]+)$")


class VisibilitySampler(Node):
    def __init__(self, duration_s: float) -> None:
        super().__init__("camera_visibility_sampler")
        self.duration_s = duration_s
        self.started = time.monotonic()
        self.received_at: list[float] = []
        self.detected: list[bool] = []
        self.confirmed: list[bool] = []
        self.confidences: list[float] = []
        self.depths: list[float] = []
        self.positions: list[tuple[float, float]] = []
        self.reasons: dict[str, int] = {}
        self.create_subscription(
            String, "/target_detection_status", self._on_status, 20
        )

    def _on_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        now = time.monotonic()
        self.received_at.append(now)
        self.detected.append(bool(payload.get("target_detected", False)))
        self.confirmed.append(bool(payload.get("confirmed", False)))
        reason = str(payload.get("reason", "unknown"))
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

        confidence = payload.get("candidate_confidence")
        if confidence is not None:
            try:
                self.confidences.append(float(confidence))
            except (TypeError, ValueError):
                pass
        x = payload.get("candidate_x")
        y = payload.get("candidate_y")
        if x is not None and y is not None:
            try:
                self.positions.append((float(x), float(y)))
            except (TypeError, ValueError):
                pass
        for source in payload.get("sources", []):
            match = DEPTH_PATTERN.match(str(source))
            if match:
                try:
                    self.depths.append(float(match.group(1)))
                except ValueError:
                    pass

    def done(self) -> bool:
        return time.monotonic() - self.started >= self.duration_s

    def summary(self) -> dict:
        total = len(self.received_at)
        result: dict = {
            "samples": total,
            "duration_s": round(self.duration_s, 3),
            "status_rate_hz": round(total / max(self.duration_s, 1e-6), 3),
            "detection_ratio": round(sum(self.detected) / total, 3) if total else 0.0,
            "confirmed_ratio": round(sum(self.confirmed) / total, 3) if total else 0.0,
            "reasons": self.reasons,
        }
        if self.confidences:
            result["confidence"] = {
                "min": round(min(self.confidences), 3),
                "median": round(statistics.median(self.confidences), 3),
                "mean": round(statistics.fmean(self.confidences), 3),
                "max": round(max(self.confidences), 3),
                "strong_ratio_ge_0_70": round(
                    sum(value >= 0.70 for value in self.confidences)
                    / len(self.confidences),
                    3,
                ),
            }
        if self.depths:
            result["depth_m"] = {
                "min": round(min(self.depths), 3),
                "median": round(statistics.median(self.depths), 3),
                "mean": round(statistics.fmean(self.depths), 3),
                "max": round(max(self.depths), 3),
            }
        if self.positions:
            mean_x = statistics.fmean(point[0] for point in self.positions)
            mean_y = statistics.fmean(point[1] for point in self.positions)
            radial_errors = [
                math.hypot(point[0] - mean_x, point[1] - mean_y)
                for point in self.positions
            ]
            result["map_position"] = {
                "mean_x": round(mean_x, 3),
                "mean_y": round(mean_y, 3),
                "max_radius_from_mean_m": round(max(radial_errors), 3),
                "rms_radius_m": round(
                    math.sqrt(statistics.fmean(value * value for value in radial_errors)),
                    3,
                ),
            }
        result["stable"] = bool(
            total >= max(5, int(self.duration_s * 2.0))
            and result["detection_ratio"] >= 0.80
            and result.get("confidence", {}).get("median", 0.0) >= 0.55
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one fixed target pose; this script publishes no commands."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--actual-distance", type=float)
    parser.add_argument("--actual-angle", type=float)
    parser.add_argument(
        "--output",
        default="measure/camera_visibility_results.jsonl",
        help="JSON-lines result file relative to the current directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = VisibilitySampler(max(2.0, args.duration))
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.2)
        result = node.summary()
        result["label"] = args.label
        result["actual_distance_m"] = args.actual_distance
        result["actual_angle_deg"] = args.actual_angle
        result["recorded_at_unix_s"] = round(time.time(), 3)
        print(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            flush=True,
        )

        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"Saved: {output}", flush=True)
        return 0 if result["samples"] else 2
    finally:
        # Save/print before DDS teardown.  Some CycloneDDS versions can take
        # several seconds to tear down an interface after a cable reconnect;
        # that must not hide an otherwise complete measurement.
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
