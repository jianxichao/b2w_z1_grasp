"""Tests for the mission supervisor's latched status topic.

The status publisher gives operators (and rl2_stack.sh status) a one-line
JSON snapshot of the mission state machine without following the event
stream.  These tests cover _build_status_payload/_publish_node_status and
that _publish_event feeds the payload's last_event field.
"""

import json
from types import SimpleNamespace

from rl2_nav.mission_supervisor_core import MissionState
from rl2_nav.mission_supervisor_node import MissionSupervisorNode


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


class _Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _make_supervisor(**overrides):
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.IDLE
    node._reached_target_count = 0
    node._target_goal_count = 1
    node._manual_mission_active = False
    node._slam_guard_state = "MONITORING"
    node._motion_gate_enabled = None
    node._pending_targets = []
    node._locked_targets = []
    node._slam_loss_since = 0.0
    node._goal_label = ""
    node._spin_active = False
    node._target_posture_phase = ""
    node._robot_xy = None
    node._base_pose = None
    node._last_event = ""
    node._last_status_pub_s = 0.0
    node._event_pub = _Recorder()
    node._status_pub = _Recorder()
    node.get_logger = lambda: _Logger()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    )
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_status_payload_has_operator_fields() -> None:
    node = _make_supervisor()

    payload = node._build_status_payload(100.0)

    assert payload["state"] == "IDLE"
    assert payload["reached_targets"] == 0
    assert payload["target_goal_count"] == 1
    assert payload["manual_mission"] is False
    assert payload["guard_state"] == "MONITORING"
    assert payload["motion_gate_enabled"] is None
    assert payload["pending_targets"] == 0
    assert payload["pending_target_positions"] == []
    assert payload["confirmed_target_positions"] == []
    assert payload["slam_loss_suspected_s"] == 0.0
    assert payload["active_goal"] == ""
    assert payload["spin_active"] is False
    assert payload["posture_phase"] == ""
    assert payload["robot_xy"] is None
    assert payload["base_xy"] is None
    assert payload["last_event"] == ""


def test_status_payload_rounds_and_reports_suspected_loss() -> None:
    base = SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=1.23456, y=2.0))
    )
    node = _make_supervisor(
        _state=MissionState.SEARCHING,
        _robot_xy=(0.123456, -0.654321),
        _base_pose=base,
        _slam_guard_state="RECOVERING",
        _slam_loss_since=97.5,
        _goal_label="approach",
        _last_event="mission_started:service_start",
        _pending_targets=[(3.12345, -1.23456, 0.87654)],
        _locked_targets=[(1.23456, 2.34567)],
    )

    payload = node._build_status_payload(100.0)

    assert payload["state"] == "SEARCHING"
    assert payload["robot_xy"] == [0.123, -0.654]
    assert payload["base_xy"] == [1.235, 2.0]
    assert payload["guard_state"] == "RECOVERING"
    assert payload["slam_loss_suspected_s"] == 2.5
    assert payload["active_goal"] == "approach"
    assert payload["last_event"] == "mission_started:service_start"
    assert payload["pending_target_positions"] == [[3.123, -1.235, 0.877]]
    assert payload["confirmed_target_positions"] == [[1.235, 2.346]]


def test_publish_event_feeds_last_event() -> None:
    node = _make_supervisor()

    node._publish_event("mission_started:service_start")

    assert node._last_event == "mission_started:service_start"
    assert node._event_pub.messages[-1].data == "mission_started:service_start"

    payload = node._build_status_payload(100.0)
    assert payload["last_event"] == "mission_started:service_start"


def test_publish_node_status_emits_valid_json() -> None:
    node = _make_supervisor(_state=MissionState.SCANNING, _spin_active=True)

    node._publish_node_status(100.0)

    published = node._status_pub.messages[-1].data
    payload = json.loads(published)
    assert payload["state"] == "SCANNING"
    assert payload["spin_active"] is True
