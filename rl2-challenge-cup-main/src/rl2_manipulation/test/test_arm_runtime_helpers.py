import math

import numpy as np
import pytest

from rl2_manipulation.arm_action_server import PROGRESS, _quaternion, _rpy_quaternion
from rl2_manipulation.vision_target_node import _status_payload, _transform_matrix
from robot_arm.manipulation.state_machine import State


def test_identity_quaternion():
    assert _quaternion(np.eye(3)) == (0.0, 0.0, 0.0, 1.0)


def test_yaw_quaternion():
    actual = _rpy_quaternion((0.0, 0.0, math.pi / 2.0))
    assert actual == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))


def test_action_progress_finishes_at_one():
    assert PROGRESS[State.CHECK] == 0.0
    assert PROGRESS[State.DONE] == 1.0


def test_ros_transform_to_matrix():
    transform = type("Transform", (), {})()
    transform.translation = type("Translation", (), {"x": 1.0, "y": 2.0, "z": 3.0})()
    transform.rotation = type(
        "Rotation", (), {"x": 0.0, "y": 0.0, "z": math.sqrt(0.5), "w": math.sqrt(0.5)}
    )()
    actual = _transform_matrix(transform)
    assert actual[:3, 3] == pytest.approx((1.0, 2.0, 3.0))
    assert np.allclose(
        actual[:3, :3],
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        atol=1e-12,
    )


def test_empty_vision_status_is_json():
    assert _status_payload("SEARCHING") == (
        '{"stage":"SEARCHING","target_id":null,"class_name":null,"confidence":null,'
        '"camera_xyz_m":null,"arm_base_xyz_m":null,"reason":""}'
    )
