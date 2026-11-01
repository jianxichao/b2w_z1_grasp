"""Dog -> arm target report and return-goal heading regressions (batch 10).

Covers the 2026-09-27 field-feedback batch: the latched target_report JSON
payload (including the transform-free robot-relative bearing), and the return
goal arriving along the approach direction instead of the mission-start
heading.
"""

import json
import math
from types import SimpleNamespace

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion

from rl2_nav.mission_supervisor_node import MissionSupervisorNode


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
        return None


class _ReportRecorder:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _make_report_node(robot_xy, robot_yaw):
    node = object.__new__(MissionSupervisorNode)
    node._arm_integration_enabled = False
    node._locked_targets = []
    node._reached_target_count = 0
    node._enter_confirmed_target_hold = lambda *args, **kwargs: None
    node._arm_report_sequence = 0
    node._arm_report_pub = _ReportRecorder()
    node._map_frame = "map"
    node._robot_xy = robot_xy
    node._robot_yaw = robot_yaw
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=1_700_000_000_000_000_000)
    )
    node.get_logger = lambda: _Logger()
    return node


def test_confirmed_target_report_publishes_map_pose_bearing_and_distance():
    node = _make_report_node((0.5, 0.5), 0.0)

    node._accept_confirmed_target(1.5, 0.5, source="visual", confidence=0.8)
    node._accept_confirmed_target(2.5, 0.5, source="visual", confidence=0.9)

    assert node._locked_targets == [(1.5, 0.5), (2.5, 0.5)]
    assert node._reached_target_count == 2
    assert len(node._arm_report_pub.messages) == 2
    first = json.loads(node._arm_report_pub.messages[0].data)
    second = json.loads(node._arm_report_pub.messages[1].data)
    # Sequential numbering, stable envelope, and the actionable fields the
    # arm stack needs: map position plus transform-free robot-relative hints.
    assert first["seq"] == 1
    assert second["seq"] == 2
    for payload in (first, second):
        assert payload["event"] == "target_confirmed"
        assert payload["frame"] == "map"
        assert payload["robot_xy"] == [0.5, 0.5]
        assert payload["robot_yaw_rad"] == 0.0
    assert first["source"] == "visual"
    assert first["confidence"] == 0.8
    assert first["target_xy"] == [1.5, 0.5]
    # Straight ahead of a zero-yaw robot: bearing 0, distance 1.0m.
    assert first["bearing_rad"] == 0.0
    assert first["distance_m"] == 1.0
    assert second["target_xy"] == [2.5, 0.5]
    assert second["bearing_rad"] == 0.0
    assert second["distance_m"] == 2.0


def test_report_bearing_is_normalized_robot_frame_angle():
    # Robot faces +y (yaw = pi/2); the target at +x is on its right side,
    # so the robot-frame bearing is -pi/2.
    node = _make_report_node((0.0, 0.0), math.pi / 2)
    node._publish_arm_target_report(1.0, 0.0, source="visual", confidence=None)

    payload = json.loads(node._arm_report_pub.messages[0].data)
    assert payload["bearing_rad"] == round(-math.pi / 2, 3)
    assert payload["distance_m"] == 1.0

    # A target behind the robot must normalize into [-pi, pi], not wrap.
    behind = _make_report_node((0.0, 0.0), 0.0)
    behind._publish_arm_target_report(-1.0, 0.0, source="visual", confidence=None)
    behind_payload = json.loads(behind._arm_report_pub.messages[0].data)
    assert behind_payload["bearing_rad"] == round(math.pi, 3)


def test_report_still_publishes_without_robot_pose():
    node = _make_report_node(None, 0.0)
    node._publish_arm_target_report(1.0, 2.0, source="scan", confidence=0.6)

    assert len(node._arm_report_pub.messages) == 1
    payload = json.loads(node._arm_report_pub.messages[0].data)
    assert payload["target_xy"] == [1.0, 2.0]
    assert payload["robot_xy"] is None
    assert payload["bearing_rad"] is None
    assert payload["distance_m"] is None


def _make_return_node(robot_xy):
    node = object.__new__(MissionSupervisorNode)
    base = PoseStamped()
    base.header.frame_id = "map"
    base.pose.position.x = 0.0
    base.pose.position.y = 0.0
    # Mission-start heading: +90 degrees. The historic bug kept this in the
    # goal, so Nav2 rotated in place at the base to restore it.
    start_yaw = math.pi / 2
    base.pose.orientation = Quaternion(
        z=math.sin(start_yaw / 2.0), w=math.cos(start_yaw / 2.0)
    )
    node._base_pose = base
    node._latest_map = None
    node._map_frame = "map"
    node._robot_xy = robot_xy
    node._robot_yaw = 0.0
    node._active_return_xy = None
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=0, to_msg=lambda: Time())
    )
    node.get_logger = lambda: _Logger()
    sent = []
    node._send_goal = lambda goal, label: sent.append((goal, label))
    node._publish_event = lambda *_args, **_kwargs: None
    return node, sent


def test_return_goal_arrives_along_approach_direction_not_start_heading():
    node, sent = _make_return_node((1.0, 1.0))

    node._send_return_goal()

    assert len(sent) == 1
    goal, label = sent[0]
    assert label == "return"
    assert goal.pose.pose.position.x == 0.0
    assert goal.pose.pose.position.y == 0.0
    orientation = goal.pose.pose.orientation
    goal_yaw = math.atan2(
        2.0 * float(orientation.w) * float(orientation.z),
        1.0 - 2.0 * float(orientation.z) ** 2,
    )
    # Approach direction from (1, 1) to the base at (0, 0) is -135 degrees.
    assert abs(goal_yaw - math.atan2(-1.0, -1.0)) < 1e-9
    # The recorded base pose itself stays immutable.
    assert abs(node._base_pose.pose.orientation.z - math.sin(math.pi / 4)) < 1e-9
    assert abs(node._base_pose.pose.orientation.w - math.cos(math.pi / 4)) < 1e-9


def test_return_goal_keeps_recorded_heading_without_robot_pose():
    # No /slam/current_pose yet: fall back to the recorded base heading
    # instead of crashing, matching the historic behavior.
    node, sent = _make_return_node(None)

    node._send_return_goal()

    assert len(sent) == 1
    goal, _label = sent[0]
    orientation = goal.pose.pose.orientation
    goal_yaw = math.atan2(
        2.0 * float(orientation.w) * float(orientation.z),
        1.0 - 2.0 * float(orientation.z) ** 2,
    )
    assert abs(goal_yaw - math.pi / 2) < 1e-9
