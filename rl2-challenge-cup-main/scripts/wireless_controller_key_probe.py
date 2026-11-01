#!/usr/bin/env python3
"""Passively report GO2-W wireless-controller key changes.

This diagnostic is deliberately read-only: it creates no publishers or
services and never sends a robot command.  It is used to measure the actual
``keys`` bit mask before a competition start/abort chord is configured.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import WirelessController


def active_bits(value: int) -> list[int]:
    """Return set bit indices in the 16-bit controller key mask."""
    return [index for index in range(16) if value & (1 << index)]


class WirelessControllerKeyProbe(Node):
    def __init__(self, output_path: Optional[str]) -> None:
        super().__init__("wireless_controller_key_probe")
        self._output_path = output_path
        self._started = time.monotonic()
        self._samples = 0
        self._last_keys: Optional[int] = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            WirelessController,
            "/wirelesscontroller",
            self._on_controller,
            qos,
        )
        self.get_logger().info(
            "PASSIVE probe ready; no publisher, service client, or motion command exists."
        )

    @property
    def samples(self) -> int:
        return self._samples

    def _on_controller(self, msg: WirelessController) -> None:
        self._samples += 1
        keys = int(msg.keys) & 0xFFFF
        if keys == self._last_keys:
            return
        self._last_keys = keys
        record = {
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "keys_decimal": keys,
            "keys_hex": "0x%04X" % keys,
            "active_bits": active_bits(keys),
            "lx": round(float(msg.lx), 3),
            "ly": round(float(msg.ly), 3),
            "rx": round(float(msg.rx), 3),
            "ry": round(float(msg.ry), 3),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        print(line, flush=True)
        if self._output_path:
            try:
                with open(self._output_path, "a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except OSError as exc:
                self.get_logger().error("cannot append probe output: %s" % exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only GO2-W wireless-controller key-mask probe."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="seconds to observe; 0 means until Ctrl+C (default: 30)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="optional JSON-lines output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # argparse owns this script's CLI.  Do not pass its non-ROS options to
    # rclpy's ROS argument parser.
    rclpy.init(args=[])
    node = WirelessControllerKeyProbe(args.output or None)
    deadline = None if args.duration <= 0.0 else time.monotonic() + args.duration
    try:
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        samples = node.samples
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if samples == 0:
        print(
            "NO_SAMPLES: wake/pair the remote and verify /wirelesscontroller QoS.",
            flush=True,
        )
        return 2
    print("PASSIVE_PROBE_COMPLETE: received %d samples" % samples, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
