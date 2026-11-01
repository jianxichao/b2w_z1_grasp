"""Tests for cmd_vel_safety_node's latched gate status topic.

`ros2 param get` cannot read the gate's enabled state (it lives in an
instance attribute), which is exactly why the status topic exists: a late
`topic echo --once` reader must immediately see {enabled,
odom_jump_monitoring, reason} after every state transition.
"""

import json

from b2w_nav.cmd_vel_safety_node import CmdVelSafetyNode


class _Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _make_gate(**overrides):
    node = object.__new__(CmdVelSafetyNode)
    node._enabled = False
    node._odom_jump_monitoring_enabled = True
    node._status_pub = _Recorder()
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_status_payload_fields() -> None:
    node = _make_gate()

    node._publish_status("startup")

    payload = json.loads(node._status_pub.messages[-1].data)
    assert payload == {
        "enabled": False,
        "odom_jump_monitoring": True,
        "reason": "startup",
    }


def test_status_reflects_enable_transition() -> None:
    node = _make_gate()

    node._enabled = True
    node._publish_status("enable_service")

    payload = json.loads(node._status_pub.messages[-1].data)
    assert payload["enabled"] is True
    assert payload["reason"] == "enable_service"


def test_status_reflects_odom_jump_monitoring_pause() -> None:
    node = _make_gate()

    node._odom_jump_monitoring_enabled = False
    node._publish_status("odom_jump_monitoring:False")

    payload = json.loads(node._status_pub.messages[-1].data)
    assert payload["odom_jump_monitoring"] is False
    assert payload["reason"] == "odom_jump_monitoring:False"
