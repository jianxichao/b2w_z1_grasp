import json

from rl2_hardware.go2w_sport_cmd_bridge import (
    API_BALANCE_STAND,
    API_MOVE,
    API_STOP_MOVE,
    API_STAND_DOWN,
    API_STAND_UP,
    Go2WSportCmdBridge,
    clamp,
    move_parameter,
    posture_exclusive_active,
    response_succeeded,
)


def test_sport_api_ids_match_unitree_client():
    assert API_BALANCE_STAND == 1002
    assert API_STOP_MOVE == 1003
    assert API_STAND_UP == 1004
    assert API_STAND_DOWN == 1005
    assert API_MOVE == 1008


def test_clamp_is_symmetric():
    assert clamp(0.2, 0.05) == 0.05
    assert clamp(-0.2, 0.05) == -0.05
    assert clamp(0.03, 0.05) == 0.03


def test_move_payload_preserves_lateral_motion():
    payload = json.loads(move_parameter(0.04, -0.05, -0.08))
    assert payload == {"x": 0.04, "y": -0.05, "z": -0.08}


def test_response_success_parser_handles_string_false():
    assert response_succeeded(True)
    assert response_succeeded("true")
    assert not response_succeeded(False)
    assert not response_succeeded("false")


def test_posture_exclusive_window_blocks_watchdog_until_deadline():
    assert posture_exclusive_active(100, 101)
    assert not posture_exclusive_active(101, 101)
    assert not posture_exclusive_active(102, 101)
    assert not posture_exclusive_active(100, 0)


class _StatusRecorder:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(json.loads(msg.data))


def _make_bridge(**overrides):
    node = object.__new__(Go2WSportCmdBridge)
    node._enabled = False
    node._ever_enabled = False
    node._status_pub = _StatusRecorder()
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_gate_status_latches_enable_state():
    node = _make_bridge()

    node._publish_gate_status(False)
    node._publish_gate_status(True)

    assert node._status_pub.messages[0] == {"stage": "gate", "enabled": False}
    assert node._status_pub.messages[1] == {"stage": "gate", "enabled": True}


def test_status_payload_carries_enabled_field():
    node = _make_bridge()

    node._publish_status(True, API_STOP_MOVE, 0, "StopMove: bridge disabled")

    payload = node._status_pub.messages[-1]
    assert payload["stage"] == "local"
    assert payload["enabled"] is False
    assert payload["api_id"] == API_STOP_MOVE
    assert payload["succeed"] is True

    node._enabled = True
    node._publish_status(False, API_MOVE, -1, "rejected", stage="response")
    payload = node._status_pub.messages[-1]
    assert payload["enabled"] is True
    assert payload["stage"] == "response"
