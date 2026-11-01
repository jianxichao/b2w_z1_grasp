import math
from types import SimpleNamespace

from nav_msgs.msg import OccupancyGrid

from b2w_nav.slam_pose_jump_guard_node import (
    MONITORING,
    RECOVERING,
    READY_FOR_MANUAL_REARM,
    SlamPoseJumpGuard,
    known_cell_count,
    planar_step,
)


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs

    def fatal(self, *args, **kwargs):
        del args, kwargs


def _make_guard_node(
    *,
    state=RECOVERING,
    now_ns=100_000_000_000,
    latest_odom_ns=99_500_000_000,
    pose_stream_lost_after_s=5.0,
):
    node = object.__new__(SlamPoseJumpGuard)
    node._state = state
    node._monitoring_enabled = True
    node._previous_xy = None
    node._latest_odom_ns = latest_odom_ns
    node._stable_since_ns = 0
    node._jump_count = 0
    node._recovery_generation = 0
    node._last_jump_distance_m = 0.0
    node._map_received_after_jump = False
    node._known_cells = 0
    node._pose_stream_lost = False
    node._pose_stream_lost_after_s = pose_stream_lost_after_s
    node._stable_step_distance_m = 0.10
    node._stable_duration_s = 3.0
    node._minimum_known_map_cells = 1000
    node._operations = {}
    node._service_clients = {}
    node._last_reason = "pose_jump_detected"
    node._publish_status = lambda: None
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=now_ns)
    )
    node.get_logger = lambda: _Logger()
    return node


def test_planar_step_detects_large_frame_jump():
    assert math.isclose(planar_step((1.0, 2.0), (1.3, 2.4)), 0.5)
    assert planar_step((3.9, 5.5), (0.9, 0.5)) > 5.0


def test_known_cell_count_ignores_unknown_cells():
    msg = OccupancyGrid()
    msg.data = [-1, -1, 0, 25, 100]
    assert known_cell_count(msg) == 3


def test_recovery_step_latches_pose_stream_lost():
    node = _make_guard_node(latest_odom_ns=90_000_000_000)

    node._recovery_step()

    assert node._pose_stream_lost is True
    assert node._last_reason == "pose_stream_lost"


def test_recovery_step_keeps_fresh_stream_unlatched():
    node = _make_guard_node(latest_odom_ns=99_500_000_000)

    node._recovery_step()

    assert node._pose_stream_lost is False
    assert node._state == RECOVERING


def test_ready_for_manual_rearm_clears_pose_stream_lost():
    now_ns = 100_000_000_000
    node = _make_guard_node(latest_odom_ns=now_ns - 500_000_000)
    node._pose_stream_lost = True
    node._stable_since_ns = now_ns - 4_000_000_000
    node._map_received_after_jump = True
    node._known_cells = 1200
    node._operations = {"reset_occupancy_grid": {"done": True}}

    node._recovery_step()

    assert node._state == READY_FOR_MANUAL_REARM
    assert node._pose_stream_lost is False


def test_begin_recovery_resets_pose_stream_lost():
    node = _make_guard_node(state=MONITORING)
    node._pose_stream_lost = True

    node._begin_recovery(1.5)

    assert node._state == RECOVERING
    assert node._pose_stream_lost is False
    assert node._last_reason == "pose_jump_detected"
