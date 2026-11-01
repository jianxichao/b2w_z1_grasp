import math

from rl2_nav.frontier_explorer_node import (
    choose_lateral_escape_direction,
    side_sector_clearances,
)


def test_escape_moves_away_from_left_wall():
    assert choose_lateral_escape_direction(0.4, 1.2, 0.55, 0.8) == -1.0


def test_escape_moves_away_from_right_wall():
    assert choose_lateral_escape_direction(1.2, 0.4, 0.55, 0.8) == 1.0


def test_escape_refuses_when_both_destinations_are_blocked():
    assert choose_lateral_escape_direction(0.4, 0.5, 0.55, 0.8) == 0.0


def test_escape_chooses_clearer_side_for_front_or_rear_obstacle():
    assert choose_lateral_escape_direction(1.2, 1.0, 0.55, 0.8) == 1.0
    assert choose_lateral_escape_direction(0.9, 1.4, 0.55, 0.8) == -1.0
    assert choose_lateral_escape_direction(1.2, 1.2, 0.55, 0.8) == 1.0


def test_escape_uses_only_safe_side_even_without_close_side_wall():
    assert choose_lateral_escape_direction(1.0, 0.7, 0.55, 0.8) == 1.0
    assert choose_lateral_escape_direction(0.7, 1.0, 0.55, 0.8) == -1.0


def test_side_sector_clearances_follow_ros_angle_sign():
    ranges = [float("inf")] * 5
    # Angles: -180, -90, 0, +90, +180 degrees.
    ranges[1] = 0.45
    ranges[3] = 0.90
    left, right = side_sector_clearances(ranges, -math.pi, math.pi / 2.0)
    assert left == 0.90
    assert right == 0.45
