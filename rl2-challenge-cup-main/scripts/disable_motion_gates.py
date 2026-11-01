#!/usr/bin/env python3
"""Disable both motion gates without starting/stopping any other process."""

import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


SERVICE_TIMEOUT_S = 3.0
RESPONSE_TIMEOUT_S = 5.0
GATE_1 = "/cmd_vel_safety_node/enable"
GATE_2 = "/go2w_sport_cmd_bridge/enable"


def call_disable(node: Node, service: str):
    client = node.create_client(SetBool, service)
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
        return False, "service unavailable"
    request = SetBool.Request()
    request.data = False
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=RESPONSE_TIMEOUT_S)
    if not future.done():
        return False, "response timeout"
    try:
        response = future.result()
    except Exception as exc:
        return False, "service exception: %s" % exc
    return bool(response.success), response.message or "no message"


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("disable_motion_gates_once")
    failures = 0
    try:
        # Remove the upstream velocity path first, then close the Sport output
        # gate.  Always attempt both calls; Gate-2 disable also sends StopMove.
        for label, service in (("Gate-1 cmd_vel", GATE_1),
                               ("Gate-2 Sport", GATE_2)):
            try:
                ok, message = call_disable(node, service)
            except Exception as exc:
                ok, message = False, "unexpected error: %s" % exc
            print("%s: %s — %s" % (
                label, "DISABLED" if ok else "FAILED", message))
            if not ok:
                failures += 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if failures:
        print("RESULT: disable incomplete (%d gate failure(s))." % failures)
        return 2
    print("RESULT: both motion gates disabled; Sport StopMove requested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
