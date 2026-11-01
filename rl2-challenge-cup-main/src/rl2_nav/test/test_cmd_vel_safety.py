from nav_msgs.msg import OccupancyGrid

from rl2_nav.cmd_vel_safety_node import (
    _angular_output_step,
    _cost_at_xy,
    _minimum_nonzero_magnitude,
    _odom_step_is_jump,
)


def test_cost_at_xy_reads_robot_cell_and_rejects_out_of_bounds():
    costmap = OccupancyGrid()
    costmap.info.resolution = 0.5
    costmap.info.width = 4
    costmap.info.height = 3
    costmap.info.origin.position.x = -1.0
    costmap.info.origin.position.y = -0.5
    costmap.data = [0] * 12
    costmap.data[1 * 4 + 2] = 99

    assert _cost_at_xy(costmap, 0.1, 0.1) == 99
    assert _cost_at_xy(costmap, -1.1, 0.1) is None
    assert _cost_at_xy(costmap, 1.0, 0.1) is None


def test_angular_passthrough_does_not_add_a_second_phase_lag():
    assert _angular_output_step(0.20, -0.05, 0.20, 0.05, True) == -0.05
    assert _angular_output_step(0.20, -0.20, 0.20, 0.05, False) == 0.19


def test_odom_jump_guard_ignores_normal_motion_and_detects_frame_reset():
    assert not _odom_step_is_jump(None, (4.0, 5.0), 1.0)
    assert not _odom_step_is_jump((0.0, 0.0), (0.02, -0.01), 1.0)
    assert _odom_step_is_jump((3.9, 5.5), (0.95, 0.55), 1.0)


def test_lateral_deadband_is_symmetric():
    assert _minimum_nonzero_magnitude(0.05, 0.25) == 0.25
    assert _minimum_nonzero_magnitude(-0.05, 0.25) == -0.25
    assert _minimum_nonzero_magnitude(0.40, 0.25) == 0.40
    assert _minimum_nonzero_magnitude(0.0, 0.25) == 0.0
