import math

import numpy as np
import pytest

from robot_arm.manipulation import (
    Arx5Kinematics,
    GraspPlanner,
    Pose,
    PosePolicy,
    SafetyError,
    SafetyGuard,
    Target,
)


HOME = (0.0, 0.645, 1.040, -0.770, 0.0, 0.0)


def test_fk_ik_round_trip():
    kinematics = Arx5Kinematics()
    transform = kinematics.forward_matrix(HOME)
    result = kinematics.solve(transform[:3, 3], transform[:3, :3], [HOME])
    assert result is not None
    assert result.position_error_m < 0.004


def test_rl2_pose_faces_target_and_dives_lower():
    policy = PosePolicy()
    high = policy.rotation((0.3, 0.0, 0.35))
    low = policy.rotation((0.3, 0.0, 0.08))
    side = policy.rotation((0.0, 0.3, 0.35))
    assert not np.allclose(high, low)
    assert math.atan2(side[1, 0], side[0, 0]) == pytest.approx(math.pi / 2)


def test_plan_has_short_named_flow_and_is_safe():
    kinematics = Arx5Kinematics()
    plan = GraspPlanner(kinematics).plan(Target("x", Pose.from_xyz((0.3, 0.0, 0.2))), HOME)
    assert [command.name for command in plan.commands] == [
        "home", "search", "pregrasp", "grasp", "close", "lift", "transit", "place", "open", "retreat"
    ]
    transit = plan.command("transit")
    assert transit.target_pose is not None
    assert transit.target_pose.position == pytest.approx((0.25, 0.30, 0.355))
    SafetyGuard(kinematics).validate_path(plan.commands)


def test_safety_rejects_bad_target():
    guard = SafetyGuard(Arx5Kinematics())
    with pytest.raises(SafetyError):
        guard.validate_target(Target("behind", Pose.from_xyz((-0.3, 0.0, 0.2))))
