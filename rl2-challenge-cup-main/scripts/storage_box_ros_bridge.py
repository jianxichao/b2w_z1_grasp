#!/usr/bin/env python3
"""Expose the dock-local storage-box serial tool as ROS 2 services on the PC."""

from __future__ import annotations

import json
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger


class StorageBoxBridge(Node):
    def __init__(self) -> None:
        super().__init__("storage_box_bridge")
        self.declare_parameter("dock_host", "unitree@192.168.123.18")
        self.declare_parameter("remote_dir", "/home/unitree/rl2_storage_box")
        self.declare_parameter("ssh_timeout_sec", 8.0)
        self.declare_parameter("serial_retries", 4)
        self._lock = threading.Lock()
        self._commanded_state = "unknown"
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self.create_publisher(String, "/storage_box/status", qos)
        self.create_service(SetBool, "/storage_box/set_open", self._set_open)
        self.create_service(Trigger, "/storage_box/open", self._open)
        self.create_service(Trigger, "/storage_box/close", self._close)
        self.create_service(Trigger, "/storage_box/probe", self._probe)
        self._publish("ready", "bridge started; no servo motion requested")
        self.get_logger().info(
            "Storage-box bridge ready. Startup does not command servo motion."
        )

    def _publish(self, state: str, detail: str) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "state": state,
                "detail": detail,
                "commanded_state": self._commanded_state,
                "physical_position_confirmed": False,
                "transport": "ssh_to_dock_usb_serial",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._status_pub.publish(msg)

    def _remote(self, position: str) -> str:
        if position not in ("probe", "0", "90"):
            raise ValueError("unsupported position")
        host = str(self.get_parameter("dock_host").value)
        remote_dir = str(self.get_parameter("remote_dir").value)
        timeout = float(self.get_parameter("ssh_timeout_sec").value)
        serial_retries = max(0, int(self.get_parameter("serial_retries").value))
        remote_command = "cd %s && ./servo_box_control.py %s --retries %d" % (
            remote_dir, position, serial_retries,
        )
        command = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=2",
            host, remote_command,
        ]
        last_error = "no attempt"
        for attempt in range(1, 4):
            try:
                result = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=timeout, check=False,
                )
                output = result.stdout.strip()
                error = result.stderr.strip()
                if result.returncode == 0:
                    return output.replace("\n", " | ")
                last_error = "rc=%d: %s" % (
                    result.returncode, error or output or "no output",
                )
            except subprocess.TimeoutExpired:
                last_error = "SSH command timed out after %.1fs" % timeout
            if attempt < 3:
                time.sleep(0.5)
        raise RuntimeError("dock command failed after 3 opens: %s" % last_error)

    def _do_probe(self) -> str:
        detail = self._remote("probe")
        self._publish("online", detail)
        return detail

    def _do_command(self, position: str) -> str:
        requested = "open" if position == "90" else "closed"
        self._publish("commanding", "requesting %s" % requested)
        detail = self._remote(position)
        self._commanded_state = requested
        self._publish("command_accepted", detail)
        return detail

    def _handle(self, operation):
        with self._lock:
            try:
                return True, operation()
            except Exception as exc:
                detail = str(exc)
                self._publish("error", detail)
                return False, detail

    def _probe(self, _request, response):
        response.success, response.message = self._handle(self._do_probe)
        return response

    def _set_open(self, request, response):
        position = "90" if request.data else "0"
        response.success, response.message = self._handle(
            lambda: self._do_command(position)
        )
        return response

    def _open(self, _request, response):
        response.success, response.message = self._handle(
            lambda: self._do_command("90")
        )
        return response

    def _close(self, _request, response):
        response.success, response.message = self._handle(
            lambda: self._do_command("0")
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StorageBoxBridge()
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
