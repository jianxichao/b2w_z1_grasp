"""Tier-1 path-planning fixes (路径规划分析与优化.md 第一梯队).

Covers the 2026-09-26 field failure modes:

- frontier goals canceled by the supervisor (mode switch to scan/approach)
  used to be counted as navigation failures, feeding the blacklist and the
  self-disable fuse;
- the frontier blacklist radius escalation only applied one cycle late;
- a closed velocity gate left the supervisor idling through per-goal
  timeouts (~10 minutes in the field) instead of aborting;
- approach standoff failures never populated the failed-standoff filter, so
  the 24-way ring kept re-picking the same direction.
"""

from collections import deque
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from rclpy.time import Time
from std_msgs.msg import String

from rl2_nav.frontier_explorer_node import FrontierExplorerNode
from rl2_nav.mission_supervisor_core import MissionState
from rl2_nav.mission_supervisor_node import MissionSupervisorNode


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


# ---------------------------------------------------------------------------
# C1: externally canceled frontier goals are cooperation, not failures
# ---------------------------------------------------------------------------


def _make_frontier_for_result(**overrides):
    node = object.__new__(FrontierExplorerNode)
    node._enabled = True
    node._active_goal_generation = 7
    node._active_goal = object()
    node._active_candidate = SimpleNamespace(x=1.0, y=2.0)
    node._consecutive_failures = 0
    node._failed_goals = deque()
    node._goal_sent_time = Time(nanoseconds=1)
    node._goal_last_progress_time = Time(nanoseconds=1)
    node._goal_best_distance_m = 1.0
    node._goals_attempted_this_enable = 1
    node._max_goals_per_enable = 0
    node._replan_requested = False
    values = {
        "failure_stop_threshold": 99,
        "backup_failure_threshold": 99,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = lambda: _Logger()
    events = []
    node._publish_event = events.append
    node._events = events
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def _goal_future(status: int):
    return SimpleNamespace(result=lambda: SimpleNamespace(status=status))


def test_canceled_goal_is_not_a_failure() -> None:
    node = _make_frontier_for_result()

    node._on_goal_result(_goal_future(GoalStatus.STATUS_CANCELED), 7)

    assert node._consecutive_failures == 0
    assert list(node._failed_goals) == []
    assert node._events == ["frontier_goal_canceled"]
    assert node._active_goal is None
    assert node._active_candidate is None
    # Exploration continues on the next timer tick.
    assert node._replan_requested is True


def test_aborted_goal_still_counts_as_failure() -> None:
    node = _make_frontier_for_result()

    node._on_goal_result(_goal_future(GoalStatus.STATUS_ABORTED), 7)

    assert node._consecutive_failures == 1
    assert list(node._failed_goals) == [(1.0, 2.0)]
    assert "frontier_goal_aborted" in node._events


# ---------------------------------------------------------------------------
# C2: blacklist radius escalation applies to the failure that triggered it
# ---------------------------------------------------------------------------


def _make_frontier_for_watchdog(**overrides):
    node = object.__new__(FrontierExplorerNode)
    node._goal_pending = False
    node._active_goal = SimpleNamespace(cancel_goal_async=lambda: None)
    node._active_candidate = SimpleNamespace(x=0.1, y=0.1)
    node._robot_xy_in_map = lambda: (0.0, 0.0)
    node._consecutive_failures = 2  # one below the escalation threshold
    node._failed_goals = deque()
    node._goal_last_progress_time = Time(nanoseconds=100_000_000_000)
    # Best distance already ~the current distance (0.14 m): no positional
    # progress, so the watchdog reaches its timeout branch.
    node._goal_best_distance_m = 0.1
    node._goal_generation = 5
    node._active_goal_generation = 5
    node._goal_sent_time = Time(nanoseconds=100_000_000_000)
    node._goals_attempted_this_enable = 1
    node._max_goals_per_enable = 0
    node._replan_requested = False
    values = {
        "goal_progress_distance_m": 0.5,
        "goal_timeout_sec": 60.0,
        "escalated_failed_goal_radius_m": 2.0,
        "max_consecutive_failures": 3,
        "failure_stop_threshold": 99,
        "backup_failure_threshold": 99,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = lambda: _Logger()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: Time(nanoseconds=200_000_000_000)
    )
    events = []
    node._publish_event = events.append
    node._maybe_start_escape = lambda: events.append("escape")
    node._events = events
    node._failed_goal_radius_m = 0.65
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_effective_rejected_radius_escalates_at_threshold() -> None:
    node = _make_frontier_for_watchdog()

    assert node._effective_rejected_radius() == 0.65
    node._consecutive_failures = 3
    assert node._effective_rejected_radius() == 2.0


def test_watchdog_blacklist_uses_escalated_radius_immediately() -> None:
    node = _make_frontier_for_watchdog()

    node._check_goal_watchdog()

    # The timeout was the third consecutive failure, so the blacklist entry
    # must already be filtered at the escalated radius, not one cycle later.
    assert node._consecutive_failures == 3
    assert list(node._failed_goals) == [(0.1, 0.1)]
    assert node._effective_rejected_radius() == 2.0
    assert "frontier_goal_timeout" in node._events


# ---------------------------------------------------------------------------
# D1: closed velocity gate aborts the mission instead of idling
# ---------------------------------------------------------------------------


def _make_supervisor(**overrides):
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.SEARCHING
    node._motion_gate_abort_enabled = True
    node._motion_gate_closed_confirm_s = 2.0
    node._motion_gate_enabled = None
    node._motion_gate_reason = ""
    node._motion_gate_closed_since = 0.0
    node._target_posture_phase = ""
    node._sport_stop_client = object()  # evaluated before the stubbed call
    node.get_logger = lambda: _Logger()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    )
    events = []
    stops = []
    resets = []
    node._publish_event = events.append
    node._disable_frontier = lambda: stops.append("disable_frontier")
    node._cancel_active_goal = lambda reason: stops.append("cancel:%s" % reason)
    node._call_trigger = lambda client, label: stops.append(label) or True
    node._reset_to_idle = lambda event: resets.append(event)
    node._events = events
    node._stops = stops
    node._resets = resets
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_gate_closed_aborts_after_debounce() -> None:
    node = _make_supervisor(
        _motion_gate_enabled=False,
        _motion_gate_reason="operator disarm",
    )

    node._check_motion_gate_abort(100.0)
    assert node._resets == []
    assert node._motion_gate_closed_since == 100.0

    node._check_motion_gate_abort(103.0)

    assert node._motion_gate_closed_since == 0.0
    assert node._events == [
        "mission_motion_gate_closed_abort:operator disarm",
    ]
    assert node._stops == [
        "disable_frontier",
        "cancel:motion_gate_closed",
        "sport StopMove after motion gate closed",
    ]
    assert node._resets == ["mission_motion_gate_closed:operator disarm"]


def test_gate_reenabled_cancels_debounce() -> None:
    node = _make_supervisor(
        _motion_gate_enabled=False,
        _motion_gate_closed_since=98.0,
    )

    node._on_motion_gate_status(String(data='{"enabled":true}'))
    node._check_motion_gate_abort(101.0)

    assert node._resets == []
    assert node._motion_gate_closed_since == 0.0


def test_gate_unknown_stays_passive() -> None:
    node = _make_supervisor(_motion_gate_enabled=None)

    node._check_motion_gate_abort(100.0)
    node._check_motion_gate_abort(120.0)

    assert node._resets == []
    assert node._motion_gate_closed_since == 0.0


def test_gate_abort_deferred_during_posture_phase() -> None:
    node = _make_supervisor(
        _motion_gate_enabled=False,
        _target_posture_phase="CROUCH_HOLD",
    )

    node._check_motion_gate_abort(100.0)
    node._check_motion_gate_abort(110.0)
    assert node._resets == []
    assert node._motion_gate_closed_since == 0.0


def test_gate_abort_can_be_disabled() -> None:
    node = _make_supervisor(
        _motion_gate_enabled=False,
        _motion_gate_abort_enabled=False,
        _motion_gate_closed_since=90.0,
    )

    node._check_motion_gate_abort(100.0)

    assert node._resets == []


def test_on_motion_gate_status_parses_latched_payload() -> None:
    node = _make_supervisor()

    node._on_motion_gate_status(
        String(data='{"enabled":false,"reason":"odom pose jump; output disarmed"}')
    )
    assert node._motion_gate_enabled is False
    assert node._motion_gate_reason == "odom pose jump; output disarmed"

    # String booleans are tolerated, non-JSON ignored.
    node._on_motion_gate_status(String(data='{"enabled":"true"}'))
    assert node._motion_gate_enabled is True
    node._on_motion_gate_status(String(data="not json"))
    assert node._motion_gate_enabled is True


# ---------------------------------------------------------------------------
# D2: approach standoff failures feed the filter and retry once
# ---------------------------------------------------------------------------


def _make_supervisor_for_approach(**overrides):
    node = _make_supervisor(_state=MissionState.APPROACHING)
    node._target_pose = SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=1.0, y=1.0))
    )
    node._robot_xy = (0.0, 0.0)
    node._failed_approach_goals = deque()
    node._approach_standoff_attempts = 0
    node._approach_standoff_retry_max = 1
    node._last_approach_goal_xy = (1.45, 1.0)
    node._goal_cancel_settle_s = 0.5
    abandoned = []
    node._abandon_visual_candidate = abandoned.append
    node._abandoned = abandoned
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_approach_failure_blacklists_standoff_and_retries() -> None:
    node = _make_supervisor_for_approach()

    node._on_approach_failed("timeout")

    assert list(node._failed_approach_goals) == [(1.45, 1.0)]
    assert node._last_approach_goal_xy is None
    assert node._approach_standoff_attempts == 1
    assert node._state == MissionState.APPROACHING
    assert node._next_goal_allowed_time == 100.5
    assert node._abandoned == []
    assert "mission_approach_standoff_retry:attempt=1,reason=timeout" in (
        node._events
    )


def test_approach_retry_exhausted_abandons_candidate() -> None:
    node = _make_supervisor_for_approach(_approach_standoff_attempts=1)

    node._on_approach_failed("aborted")

    assert node._approach_standoff_attempts == 1  # no further retry
    assert node._abandoned == ["approach_aborted"]


def test_approach_retry_zero_keeps_no_retry_behavior() -> None:
    node = _make_supervisor_for_approach(_approach_standoff_retry_max=0)

    node._on_approach_failed("timeout")

    assert node._approach_standoff_attempts == 0
    assert node._abandoned == ["approach_timeout"]
    # The blacklist still records the failed standoff for the next candidate.
    assert list(node._failed_approach_goals) == [(1.45, 1.0)]


def test_arrival_not_close_uses_retry_path() -> None:
    node = _make_supervisor_for_approach()

    node._on_approach_failed("approach_not_close")

    assert node._state == MissionState.APPROACHING
    assert node._abandoned == []
    assert list(node._failed_approach_goals) == [(1.45, 1.0)]
