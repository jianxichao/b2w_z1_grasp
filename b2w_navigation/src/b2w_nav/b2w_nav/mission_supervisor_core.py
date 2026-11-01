"""Pure mission-supervisor logic, kept ROS-free for unit testing."""

from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

from b2w_nav.frontier_explorer import ApproachCandidate


class MissionState(str, Enum):
    """Supervisor state machine used by the minimal mission chain."""

    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    MANUAL_NAVIGATING = "MANUAL_NAVIGATING"
    VERIFYING_TARGET = "VERIFYING_TARGET"
    WAITING_ARM_TARGET = "WAITING_ARM_TARGET"
    GRASPING = "GRASPING"
    TARGET_HOLDING = "TARGET_HOLDING"
    HOLDING = "HOLDING"
    SCANNING = "SCANNING"
    NAVIGATING_TO_SCAN_POINT = "NAVIGATING_TO_SCAN_POINT"
    CONFIRMING_TARGET = "CONFIRMING_TARGET"
    RETURNING = "RETURNING"
    DONE = "DONE"


FRONTIER_UNAVAILABLE_EVENTS = frozenset(
    {
        "frontier_exhausted",
        "frontier_waiting_for_reachable_goal",
        # A finite per-enable goal limit means the explorer has deliberately
        # stopped producing goals.  Treat it as unavailable so the supervisor
        # cannot remain in SEARCHING forever if a non-zero limit is selected
        # for a staged run.
        "frontier_goal_limit_reached",
    }
)


def is_frontier_unavailable_event(event: str) -> bool:
    """Return True when Frontier currently has no goal it can safely send."""
    return (event or "").strip() in FRONTIER_UNAVAILABLE_EVENTS


def should_start_scan_after_frontier_unavailable(
    state: MissionState,
    unavailable_since_s: float,
    now_s: float,
    delay_s: float,
    mission_start_s: float = 0.0,
    minimum_mission_age_s: float = 0.0,
) -> bool:
    """Gate the no-Frontier fallback without relying on a repeated event.

    Frontier intentionally publishes its unavailable event once per continuous
    no-candidate interval.  The supervisor therefore owns the timeout that
    converts that interval into an in-place visual scan.
    """
    if state != MissionState.SEARCHING or unavailable_since_s <= 0.0:
        return False
    if now_s - unavailable_since_s < max(0.0, delay_s):
        return False
    if mission_start_s > 0.0 and now_s - mission_start_s < max(
        0.0, minimum_mission_age_s
    ):
        return False
    return True


def parse_target_seen(event: str) -> bool:
    """Return True when a task event announces a confirmed target."""
    text = (event or "").strip()
    return (
        text == "target_seen"
        or text == "target_detected"
        or text.startswith("target_seen:")
    )


def direct_approach_goal_xy(
    robot_x: float,
    robot_y: float,
    target_x: float,
    target_y: float,
    standoff_m: float,
) -> Tuple[float, float]:
    """Interpolate a standoff goal on the robot -> target ray."""
    if standoff_m <= 0.0:
        return target_x, target_y
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = math.hypot(dx, dy)
    scale = max(0.0, (distance - standoff_m) / max(distance, 1e-6))
    return robot_x + dx * scale, robot_y + dy * scale


def yaw_to_face_target(
    target_x: float,
    target_y: float,
    goal_x: float,
    goal_y: float,
) -> float:
    """Return the yaw at (goal_x, goal_y) that faces (target_x, target_y)."""
    return math.atan2(target_y - goal_y, target_x - goal_x)


def pick_approach_candidate(
    candidates: Sequence[ApproachCandidate],
    failed_goals: Iterable[Tuple[float, float]] = (),
    failed_radius_m: float = 0.30,
) -> Optional[ApproachCandidate]:
    """Return the first candidate not near a previously failed standoff."""
    failed = list(failed_goals)
    for candidate in candidates:
        if any(
            math.hypot(candidate.x - failed_x, candidate.y - failed_y) <= failed_radius_m
            for failed_x, failed_y in failed
        ):
            continue
        return candidate
    return None


def arrival_within_tolerance(
    distance_m: float,
    standoff_m: float,
    tolerance_m: float,
    maximum_m: Optional[float] = None,
) -> bool:
    """Return True when the robot stands inside the accepted arrival ring."""
    minimum = max(standoff_m - tolerance_m, 0.05)
    maximum = standoff_m + tolerance_m
    if maximum_m is not None:
        maximum = min(maximum, max(float(maximum_m), minimum))
    return minimum <= distance_m <= maximum


def target_is_locked(
    x: float,
    y: float,
    locked_targets: Iterable[Tuple[float, float]],
    radius_m: float,
) -> bool:
    """Return True when (x, y) is inside an already handled target."""
    return any(
        math.hypot(x - locked_x, y - locked_y) <= radius_m
        for locked_x, locked_y in locked_targets
    )


def find_farthest_free_point(
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    data: Sequence[int],
    robot_x: float,
    robot_y: float,
    min_distance_m: float = 3.0,
    max_distance_m: float = 8.0,
    free_threshold: int = 65,
    stride: int = 5,
    workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
    excluded_points: Optional[Sequence[Tuple[float, float]]] = None,
    excluded_radius_m: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Return the farthest sampled free cell in a band, avoiding prior points."""
    if width <= 0 or height <= 0 or resolution <= 0 or not data:
        return None
    if max_distance_m <= min_distance_m:
        return None

    best: Optional[Tuple[float, float]] = None
    best_distance = -1.0
    for row in range(0, height, max(1, stride)):
        base = row * width
        for col in range(0, width, max(1, stride)):
            if not 0 <= int(data[base + col]) <= free_threshold:
                continue
            x = origin_x + (col + 0.5) * resolution
            y = origin_y + (row + 0.5) * resolution
            if workspace_bounds is not None:
                min_x, max_x, min_y, max_y = workspace_bounds
                if not (min_x <= x <= max_x and min_y <= y <= max_y):
                    continue
            distance = math.hypot(x - robot_x, y - robot_y)
            if distance < min_distance_m or distance > max_distance_m:
                continue
            if excluded_points and any(
                math.hypot(x - old_x, y - old_y) <= excluded_radius_m
                for old_x, old_y in excluded_points
            ):
                continue
            if distance > best_distance:
                best = (x, y)
                best_distance = distance
    return best


def find_nearest_free_point(
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    data: Sequence[int],
    target_x: float,
    target_y: float,
    search_radius_m: float,
    free_threshold: int = 65,
) -> Optional[Tuple[float, float]]:
    """Return the nearest known-free cell center around a requested target."""
    if (
        width <= 0
        or height <= 0
        or resolution <= 0.0
        or search_radius_m < 0.0
        or len(data) < width * height
    ):
        return None
    center_col = int(math.floor((target_x - origin_x) / resolution))
    center_row = int(math.floor((target_y - origin_y) / resolution))
    radius_cells = int(math.ceil(search_radius_m / resolution))
    best: Optional[Tuple[float, float]] = None
    best_distance = math.inf
    for row in range(
        max(0, center_row - radius_cells),
        min(height, center_row + radius_cells + 1),
    ):
        for col in range(
            max(0, center_col - radius_cells),
            min(width, center_col + radius_cells + 1),
        ):
            value = int(data[row * width + col])
            if not 0 <= value <= free_threshold:
                continue
            x = origin_x + (col + 0.5) * resolution
            y = origin_y + (row + 0.5) * resolution
            distance = math.hypot(x - target_x, y - target_y)
            if distance <= search_radius_m + 0.5 * resolution and distance < best_distance:
                best = (x, y)
                best_distance = distance
    return best


def candidate_is_rejected(
    x: float,
    y: float,
    rejected_candidates: Iterable[Tuple[float, float, float]],
    radius_m: float,
    now: float,
    cooldown_s: float,
) -> bool:
    """Return True when a candidate is inside a recent failed-confirm rejection."""
    return any(
        math.hypot(x - rejected_x, y - rejected_y) <= radius_m
        and now - rejected_at <= cooldown_s
        for rejected_x, rejected_y, rejected_at in rejected_candidates
    )


def square_fence_bounds(
    center_x: float,
    center_y: float,
    half_size_m: float,
) -> Tuple[float, float, float, float]:
    """Return (min_x, max_x, min_y, max_y) for an axis-aligned square fence."""
    return (
        center_x - half_size_m,
        center_x + half_size_m,
        center_y - half_size_m,
        center_y + half_size_m,
    )


def point_inside_square(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    half_size_m: float,
) -> bool:
    """Return True when (x, y) lies inside the axis-aligned square fence."""
    return abs(x - center_x) <= half_size_m and abs(y - center_y) <= half_size_m


def point_inside_square_with_margin(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    half_size_m: float,
    margin_m: float,
) -> bool:
    """Return True only inside a square's usable interior.

    ``margin_m`` is an inward safety band, expressed in metres, shared by
    Frontier, visual goals, and scan fallbacks.
    """
    usable_half_size_m = half_size_m - max(0.0, margin_m)
    return usable_half_size_m > 0.0 and point_inside_square(
        x, y, center_x, center_y, usable_half_size_m
    )


def distance_to_square_edge(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    half_size_m: float,
) -> float:
    """Return the smallest distance from (x, y) to any square boundary."""
    if half_size_m <= 0.0:
        return 0.0
    return min(
        center_x + half_size_m - x,
        x - (center_x - half_size_m),
        center_y + half_size_m - y,
        y - (center_y - half_size_m),
    )
