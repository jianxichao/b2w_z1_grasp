"""Unit tests for the ROS-free mission supervisor core logic."""

import math

import pytest

from rl2_nav.frontier_explorer import ApproachCandidate
from rl2_nav.mission_supervisor_core import (
    MissionState,
    arrival_within_tolerance,
    candidate_is_rejected,
    distance_to_square_edge,
    direct_approach_goal_xy,
    find_farthest_free_point,
    find_nearest_free_point,
    is_frontier_unavailable_event,
    parse_target_seen,
    pick_approach_candidate,
    point_inside_square,
    point_inside_square_with_margin,
    square_fence_bounds,
    should_start_scan_after_frontier_unavailable,
    target_is_locked,
    yaw_to_face_target,
)


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("target_seen:rock_obstacle", True),
        ("target_seen", True),
        ("target_detected", True),
        ("target_lost", False),
        ("frontier_goal_reached", False),
        ("", False),
        ("  target_seen:science_target  ", True),
    ],
)
def test_parse_target_seen(event: str, expected: bool) -> None:
    assert parse_target_seen(event) is expected


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("frontier_exhausted", True),
        ("frontier_waiting_for_reachable_goal", True),
        ("frontier_goal_limit_reached", True),
        ("frontier_goal_started", False),
        ("frontier_goal_aborted", False),
        ("", False),
    ],
)
def test_is_frontier_unavailable_event(event: str, expected: bool) -> None:
    assert is_frontier_unavailable_event(event) is expected


@pytest.mark.parametrize(
    ("state", "since", "now", "delay", "expected"),
    [
        (MissionState.SEARCHING, 10.0, 17.9, 8.0, False),
        (MissionState.SEARCHING, 10.0, 18.0, 8.0, True),
        (MissionState.SEARCHING, 10.0, 11.0, 0.0, True),
        (MissionState.SCANNING, 10.0, 30.0, 8.0, False),
        (MissionState.SEARCHING, 0.0, 30.0, 8.0, False),
    ],
)
def test_should_start_scan_after_frontier_unavailable(
    state, since, now, delay, expected
) -> None:
    assert (
        should_start_scan_after_frontier_unavailable(state, since, now, delay)
        is expected
    )


def test_scan_fallback_respects_minimum_mission_age() -> None:
    assert not should_start_scan_after_frontier_unavailable(
        MissionState.SEARCHING,
        unavailable_since_s=102.0,
        now_s=110.0,
        delay_s=8.0,
        mission_start_s=100.0,
        minimum_mission_age_s=20.0,
    )
    assert should_start_scan_after_frontier_unavailable(
        MissionState.SEARCHING,
        unavailable_since_s=110.0,
        now_s=120.0,
        delay_s=8.0,
        mission_start_s=100.0,
        minimum_mission_age_s=20.0,
    )


@pytest.mark.parametrize(
    ("robot", "target", "standoff", "expected"),
    [
        ((0.0, 0.0), (1.0, 0.0), 0.45, (0.55, 0.0)),
        ((0.0, 0.0), (2.0, 0.0), 0.5, (1.5, 0.0)),
        ((1.0, 1.0), (2.0, 3.0), 0.45, (1.798756, 2.597512)),
        # Already closer than the standoff: fallback stays in place.
        ((0.0, 0.0), (0.3, 0.0), 0.45, (0.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), 0.0, (1.0, 0.0)),  # standoff disabled
    ],
)
def test_direct_approach_goal_xy(
    robot, target, standoff, expected
) -> None:
    x, y = direct_approach_goal_xy(
        robot[0], robot[1], target[0], target[1], standoff
    )
    assert x == pytest.approx(expected[0], abs=1e-5)
    assert y == pytest.approx(expected[1], abs=1e-5)


def test_direct_approach_goal_distance() -> None:
    x, y = direct_approach_goal_xy(0.0, 0.0, 3.0, 4.0, 0.45)
    assert math.hypot(x, y) == pytest.approx(math.hypot(3.0, 4.0) - 0.45, abs=1e-9)


def test_yaw_to_face_target() -> None:
    assert yaw_to_face_target(0.0, 1.0, 0.0, 0.0) == pytest.approx(math.pi / 2.0)
    assert yaw_to_face_target(1.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert yaw_to_face_target(0.0, -1.0, 0.0, 0.0) == pytest.approx(-math.pi / 2.0)


def _candidate(x: float, y: float) -> ApproachCandidate:
    return ApproachCandidate(
        x=x,
        y=y,
        yaw=0.0,
        score=0.0,
        standoff_m=0.45,
        angle_rad=0.0,
    )


def test_pick_approach_candidate_skips_failed() -> None:
    candidates = [_candidate(1.0, 0.0), _candidate(0.0, 1.0)]
    failed = [(1.05, 0.05)]
    picked = pick_approach_candidate(candidates, failed, 0.30)
    assert picked is not None
    assert (picked.x, picked.y) == (0.0, 1.0)


def test_pick_approach_candidate_none() -> None:
    candidates = [_candidate(1.0, 0.0)]
    assert pick_approach_candidate(candidates, [(1.0, 0.0)], 0.30) is None
    assert pick_approach_candidate([], []) is None


@pytest.mark.parametrize(
    ("distance", "standoff", "tolerance", "expected"),
    [
        (0.45, 0.45, 0.20, True),
        (0.30, 0.45, 0.20, True),
        (0.64, 0.45, 0.20, True),
        (0.25, 0.45, 0.20, True),  # clamped minimum 0.05
        (0.01, 0.45, 0.20, False),
        (0.80, 0.45, 0.20, False),
    ],
)
def test_arrival_within_tolerance(
    distance: float, standoff: float, tolerance: float, expected: bool
) -> None:
    assert arrival_within_tolerance(distance, standoff, tolerance) is expected


def test_arrival_hard_maximum_clips_symmetric_tolerance() -> None:
    assert arrival_within_tolerance(0.35, 0.35, 0.08, 0.35) is True
    assert arrival_within_tolerance(0.351, 0.35, 0.08, 0.35) is False
    assert arrival_within_tolerance(0.26, 0.35, 0.08, 0.35) is False


def test_mission_state_has_scan_states() -> None:
    assert MissionState.SCANNING.value == "SCANNING"
    assert MissionState.NAVIGATING_TO_SCAN_POINT.value == "NAVIGATING_TO_SCAN_POINT"
    assert MissionState.CONFIRMING_TARGET.value == "CONFIRMING_TARGET"
    assert MissionState.VERIFYING_TARGET.value == "VERIFYING_TARGET"
    assert MissionState.TARGET_HOLDING.value == "TARGET_HOLDING"


@pytest.mark.parametrize(
    ("point", "locked", "radius", "expected"),
    [
        ((1.0, 0.0), [(1.05, 0.05)], 0.30, True),
        ((1.0, 0.0), [(5.0, 5.0)], 0.30, False),
        ((1.0, 0.0), [], 0.30, False),
        ((1.5, 0.0), [(1.0, 0.0)], 0.80, True),
    ],
)
def test_target_is_locked(point, locked, radius, expected) -> None:
    assert target_is_locked(point[0], point[1], locked, radius) is expected


def _free_grid(width: int, height: int, blocked: set) -> list[int]:
    data = [0] * (width * height)
    for col, row in blocked:
        data[row * width + col] = 100
    return data


def test_find_farthest_free_point_prefers_corner() -> None:
    data = _free_grid(10, 10, set())
    point = find_farthest_free_point(
        width=10,
        height=10,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        data=data,
        robot_x=5.0,
        robot_y=5.0,
        min_distance_m=2.0,
        max_distance_m=8.0,
        stride=1,
    )
    assert point is not None
    assert math.hypot(point[0] - 5.0, point[1] - 5.0) == pytest.approx(
        math.hypot(4.5, 4.5), abs=1e-9
    )


def test_find_farthest_free_point_respects_workspace_bounds() -> None:
    data = _free_grid(10, 10, set())
    point = find_farthest_free_point(
        width=10,
        height=10,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        data=data,
        robot_x=5.0,
        robot_y=5.0,
        min_distance_m=0.0,
        max_distance_m=8.0,
        stride=1,
        workspace_bounds=(2.0, 8.0, 2.0, 8.0),
    )
    assert point is not None
    assert 2.0 <= point[0] <= 8.0
    assert 2.0 <= point[1] <= 8.0


def test_find_farthest_free_point_excludes_obstacles() -> None:
    data = _free_grid(10, 10, {(9, 0), (9, 9), (0, 9)})
    point = find_farthest_free_point(
        width=10,
        height=10,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        data=data,
        robot_x=5.0,
        robot_y=5.0,
        min_distance_m=0.0,
        max_distance_m=8.0,
        stride=1,
    )
    assert point is not None
    assert (point[0], point[1]) == (0.5, 0.5)


def test_find_farthest_free_point_avoids_prior_scan_locations() -> None:
    data = _free_grid(8, 1, set())
    point = find_farthest_free_point(
        width=8,
        height=1,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        data=data,
        robot_x=0.0,
        robot_y=0.0,
        min_distance_m=0.0,
        max_distance_m=8.0,
        stride=1,
        excluded_points=[(7.5, 0.5)],
        excluded_radius_m=1.0,
    )
    assert point == (5.5, 0.5)


def test_find_farthest_free_point_distance_band() -> None:
    data = _free_grid(10, 10, set())
    point = find_farthest_free_point(
        width=10,
        height=10,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        data=data,
        robot_x=0.0,
        robot_y=0.0,
        min_distance_m=5.0,
        max_distance_m=6.0,
        stride=1,
    )
    assert point is not None
    distance = math.hypot(point[0], point[1])
    assert 5.0 <= distance <= 6.0


def test_find_farthest_free_point_empty_and_unknown() -> None:
    assert (
        find_farthest_free_point(
            width=10,
            height=10,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            data=[-1] * 100,
            robot_x=5.0,
            robot_y=5.0,
            min_distance_m=0.0,
            max_distance_m=8.0,
            stride=1,
        )
        is None
    )


def test_find_nearest_free_point_keeps_free_target_cell() -> None:
    data = _free_grid(7, 7, set())
    point = find_nearest_free_point(
        7, 7, 1.0, 0.0, 0.0, data, 3.5, 3.5, 2.0
    )
    assert point == pytest.approx((3.5, 3.5))


def test_find_nearest_free_point_corrects_occupied_target() -> None:
    data = _free_grid(7, 7, {(3, 3)})
    point = find_nearest_free_point(
        7, 7, 1.0, 0.0, 0.0, data, 3.5, 3.5, 2.0
    )
    assert point is not None
    assert math.hypot(point[0] - 3.5, point[1] - 3.5) == pytest.approx(1.0)


def test_find_nearest_free_point_rejects_unknown_neighborhood() -> None:
    assert find_nearest_free_point(
        5, 5, 1.0, 0.0, 0.0, [-1] * 25, 2.5, 2.5, 1.5
    ) is None


def test_candidate_is_rejected() -> None:
    rejected = [(1.0, 0.0, 100.0), (5.0, 5.0, 100.0)]
    # Inside radius and within cooldown
    assert candidate_is_rejected(1.1, 0.0, rejected, 0.8, 105.0, 10.0) is True
    # Outside radius
    assert candidate_is_rejected(3.0, 3.0, rejected, 0.8, 105.0, 10.0) is False
    # Cooldown expired
    assert candidate_is_rejected(1.1, 0.0, rejected, 0.8, 115.0, 10.0) is False
    assert candidate_is_rejected(1.1, 0.0, [], 0.8, 105.0, 10.0) is False


def test_square_fence_bounds() -> None:
    assert square_fence_bounds(0.0, 0.0, 3.5) == (-3.5, 3.5, -3.5, 3.5)
    assert square_fence_bounds(1.0, -2.0, 2.0) == (-1.0, 3.0, -4.0, 0.0)


@pytest.mark.parametrize(
    ("point", "center", "half", "expected"),
    [
        ((0.0, 0.0), (0.0, 0.0), 3.5, True),
        ((3.4, 3.4), (0.0, 0.0), 3.5, True),
        ((3.6, 0.0), (0.0, 0.0), 3.5, False),
        ((0.0, -3.5), (0.0, 0.0), 3.5, True),
        ((0.0, -3.6), (0.0, 0.0), 3.5, False),
        ((5.0, 5.0), (1.0, -2.0), 2.0, False),
    ],
)
def test_point_inside_square(point, center, half, expected) -> None:
    assert (
        point_inside_square(point[0], point[1], center[0], center[1], half)
        is expected
    )


def test_point_inside_square_with_margin() -> None:
    assert point_inside_square_with_margin(2.6, 0.0, 0.0, 0.0, 3.5, 0.8)
    assert not point_inside_square_with_margin(2.8, 0.0, 0.0, 0.0, 3.5, 0.8)
    assert not point_inside_square_with_margin(0.0, 0.0, 0.0, 0.0, 0.8, 0.8)


def test_distance_to_square_edge() -> None:
    # Robot at center: 3.5m to every edge.
    assert distance_to_square_edge(0.0, 0.0, 0.0, 0.0, 3.5) == pytest.approx(3.5)
    # Robot near the +x edge.
    assert distance_to_square_edge(3.0, 0.0, 0.0, 0.0, 3.5) == pytest.approx(0.5)
    # Robot outside the square: negative distance to nearest edge.
    assert distance_to_square_edge(4.0, 0.0, 0.0, 0.0, 3.5) == pytest.approx(-0.5)
    assert (
        find_farthest_free_point(
            width=0,
            height=0,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            data=[],
            robot_x=0.0,
            robot_y=0.0,
        )
        is None
    )
