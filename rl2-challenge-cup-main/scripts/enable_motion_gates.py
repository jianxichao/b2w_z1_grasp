#!/usr/bin/env python3
"""Enable both motion gates as one bounded, fail-safe operator action.

This helper does not publish velocity or start a mission. Gate-2 is enabled
first and Gate-1 second. If Gate-1 cannot be enabled, Gate-2 is immediately
disabled again so the system is not left partially armed.
"""

import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


SERVICE_TIMEOUT_S = 5.0
RESPONSE_TIMEOUT_S = 8.0
GATE_2 = "/go2w_sport_cmd_bridge/enable"
GATE_1 = "/cmd_vel_safety_node/enable"


def call_gate(node: Node, service: str, enabled: bool):
    client = node.create_client(SetBool, service)
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
        return False, "service unavailable"
    request = SetBool.Request()
    request.data = enabled
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
    node = rclpy.create_node("enable_motion_gates_once")
    exit_code = 4
    rollback_required = False
    try:
        ok2, msg2 = call_gate(node, GATE_2, True)
        print("Gate-2 Sport: %s — %s" % (
            "ENABLED" if ok2 else "FAILED", msg2))
        if not ok2:
            exit_code = 2
        else:
            rollback_required = True
            ok1, msg1 = call_gate(node, GATE_1, True)
            print("Gate-1 cmd_vel: %s — %s" % (
                "ENABLED" if ok1 else "FAILED", msg1))
            if ok1:
                rollback_required = False
                exit_code = 0
                print(
                    "RESULT: both motion gates enabled; "
                    "no motion command was sent.")
            else:
                exit_code = 3
    except Exception as exc:
        print("FAILED: unexpected error: %s" % exc)
        exit_code = 4
    finally:
        if rollback_required:
            rollback_ok, rollback_msg = call_gate(node, GATE_2, False)
            print("ROLLBACK Gate-2: %s — %s" % (
                "DISABLED" if rollback_ok else "FAILED", rollback_msg))
            if not rollback_ok:
                exit_code = 5
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
