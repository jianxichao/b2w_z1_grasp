import numpy as np
from geometry_msgs.msg import PoseStamped
from types import SimpleNamespace

from rl2_hardware.official_occupancy_grid_bridge import (
    OfficialOccupancyGridBridge,
    clear_oriented_footprint,
    horizontal_range_mask,
    official_slam_inputs_stale,
    official_slam_stale_reason,
)


class _Parameter:
    def __init__(self, value):
        self.value = value


def _bridge_without_ros(**parameters):
    bridge = OfficialOccupancyGridBridge.__new__(OfficialOccupancyGridBridge)
    bridge._resolution = 0.1
    bridge._origin_x = -1.0
    bridge._origin_y = -1.0
    bridge._width = 20
    bridge._height = 20
    bridge._grid = np.full((20, 20), -1, dtype=np.int8)
    bridge._hits = np.zeros((20, 20), dtype=np.int8)
    values = {
        "ground_free_radius_m": 0.1,
        "input_stale_timeout_s": 2.0,
        "clear_map_on_input_stale": True,
    }
    values.update(parameters)
    bridge.get_parameter = lambda name: _Parameter(values[name])
    bridge.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)
    )
    bridge._last_pose_time_s = 0.0
    bridge._last_cloud_time_s = 0.0
    bridge._stale_input_latched = False
    return bridge


def test_local_cloud_range_is_measured_from_sensor_origin():
    xyz = np.array([[1.0, 0.0, 0.0], [8.1, 0.0, 0.0]])
    assert horizontal_range_mask(xyz, 8.0).tolist() == [True, False]


def test_map_cloud_range_follows_robot_instead_of_map_origin():
    xyz = np.array([[10.0, 0.0, 0.0], [18.1, 0.0, 0.0]])
    mask = horizontal_range_mask(xyz, 8.0, origin_x=10.0, origin_y=0.0)
    assert mask.tolist() == [True, False]


def test_non_positive_range_disables_filter():
    xyz = np.array([[100.0, -100.0, 0.0]])
    assert horizontal_range_mask(xyz, 0.0).tolist() == [True]


def test_stale_guard_requires_both_streams_to_have_been_live():
    assert not official_slam_inputs_stale(0.0, 10.0, 20.0, 2.0)
    assert not official_slam_inputs_stale(10.0, 0.0, 20.0, 2.0)


def test_stale_guard_detects_pose_or_cloud_loss():
    assert official_slam_inputs_stale(10.0, 18.5, 20.0, 2.0)
    assert official_slam_inputs_stale(18.5, 10.0, 20.0, 2.0)
    assert not official_slam_inputs_stale(18.5, 18.5, 20.0, 2.0)


def test_stale_reason_identifies_the_lost_official_stream():
    assert official_slam_stale_reason(10.0, 18.5, 20.0, 2.0) == "pose_stale"
    assert official_slam_stale_reason(18.5, 10.0, 20.0, 2.0) == "cloud_stale"
    assert (
        official_slam_stale_reason(10.0, 10.0, 20.0, 2.0)
        == "pose_and_cloud_stale"
    )


def test_robot_footprint_clear_removes_self_points_but_keeps_wall():
    grid = np.full((20, 20), 100, dtype=np.int8)
    clear_oriented_footprint(
        grid,
        resolution=0.1,
        origin_x=-1.0,
        origin_y=-1.0,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        half_length_m=0.34,
        half_width_m=0.22,
    )
    assert grid[10, 10] == 0
    assert grid[10, 14] == 100
    assert grid[13, 10] == 100


def test_ground_evidence_creates_a_small_free_patch():
    bridge = _bridge_without_ros()
    bridge._mark_ground_free(np.array([[0.05, 0.05, -0.5]]))
    assert bridge._grid[10, 10] == 0
    assert bridge._grid[10, 11] == 0
    assert bridge._grid[0, 0] == -1


def test_live_raycast_clears_cells_but_not_endpoint():
    bridge = _bridge_without_ros()
    bridge._raycast_free(
        robot_cell=(10, 10),
        cols=np.array([15]),
        rows=np.array([10]),
    )
    assert bridge._grid[10, 10] == 0
    assert bridge._grid[10, 14] == 0
    assert bridge._grid[10, 15] == -1


def test_reset_map_clears_accumulated_cells_and_stale_inputs():
    bridge = _bridge_without_ros()
    bridge._grid[4, 7] = 100
    bridge._latest_points = np.ones((1, 3))
    bridge._latest_pose = object()
    bridge.get_logger = lambda: type(
        "Logger", (), {"warn": lambda self, message: None}
    )()
    bridge._reset_map("test")
    assert np.all(bridge._grid == -1)
    assert bridge._latest_points is None
    assert bridge._latest_pose is None


def test_pose_jump_clears_old_frame_map_before_accepting_new_pose():
    bridge = _bridge_without_ros(
        reset_on_pose_jump=True,
        pose_jump_distance_m=1.0,
    )
    previous = PoseStamped()
    previous.pose.position.x = 3.9
    previous.pose.position.y = 5.5
    current = PoseStamped()
    current.pose.position.x = 0.95
    current.pose.position.y = 0.55
    bridge._latest_pose = previous
    bridge._latest_points = np.ones((1, 3))
    bridge._grid[4, 7] = 100
    bridge.get_logger = lambda: type(
        "Logger", (), {"warn": lambda self, message: None}
    )()

    bridge._on_current_pose(current)

    assert np.all(bridge._grid == -1)
    assert bridge._latest_points is None
    assert bridge._latest_pose is current
