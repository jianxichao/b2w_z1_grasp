"""Regression tests for state transitions that previously stopped exploration."""

from collections import deque
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from b2w_nav.frontier_explorer_node import FrontierExplorerNode
from b2w_nav.mission_supervisor_core import MissionState
from b2w_nav.mission_supervisor_node import (
    MissionSupervisorNode,
    sport_mode_confirms_api,
)


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


def test_frontier_reenable_rearms_no_candidate_event() -> None:
    node = object.__new__(FrontierExplorerNode)
    node._exhausted_reported = True
    node._goals_attempted_this_enable = 4
    node._max_goals_per_enable = 0
    node._consecutive_failures = 3
    node._wall_escape_attempts = 2
    node._backup_attempts = 1
    node._wall_escape_start_xy = (1.0, 2.0)
    events = []
    node.get_parameter = lambda name: SimpleNamespace(value=0)
    node._publish_event = events.append
    node.get_logger = lambda: _Logger()

    response = SimpleNamespace(success=False, message="")
    node._on_enable(SimpleNamespace(data=True), response)

    assert node._exhausted_reported is False
    assert events == ["frontier_exploration_enabled"]
    assert response.success is True


def test_frontier_candidate_requires_two_consistent_observations() -> None:
    node = object.__new__(FrontierExplorerNode)
    node._pending_candidate = None
    node._pending_candidate_cycles = 0
    values = {
        "candidate_stability_required_cycles": 2,
        "candidate_stability_radius_m": 0.6,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = lambda: _Logger()
    events = []
    node._publish_event = events.append
    candidate = SimpleNamespace(x=1.0, y=2.0)

    assert node._candidate_is_stable(candidate) is False
    assert node._candidate_is_stable(SimpleNamespace(x=1.2, y=2.1)) is True
    assert events == ["frontier_candidate_stabilizing"]


def test_frontier_candidate_switch_resets_stability_counter() -> None:
    node = object.__new__(FrontierExplorerNode)
    node._pending_candidate = None
    node._pending_candidate_cycles = 0
    values = {
        "candidate_stability_required_cycles": 2,
        "candidate_stability_radius_m": 0.4,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = lambda: _Logger()
    events = []
    node._publish_event = events.append

    assert node._candidate_is_stable(SimpleNamespace(x=1.0, y=1.0)) is False
    assert node._candidate_is_stable(SimpleNamespace(x=-1.0, y=-1.0)) is False
    assert node._pending_candidate_cycles == 1
    assert events == [
        "frontier_candidate_stabilizing",
        "frontier_candidate_stabilizing",
    ]


def test_stabilizing_candidate_cancels_old_unavailable_timer() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.SEARCHING
    node._auto_start = False
    calls = []
    node._clear_frontier_unavailable = lambda: calls.append("clear")

    node._on_task_event(SimpleNamespace(data="frontier_candidate_stabilizing"))

    assert calls == ["clear"]


def test_visual_candidate_requires_consistent_frames_before_preempting_frontier() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.SEARCHING
    node._locked_targets = []
    node._ignored_target_radius_m = 0.8
    node._rejected_candidates = deque(maxlen=50)
    node._candidate_confidence_threshold = 0.30
    node._candidate_reject_radius_m = 0.8
    node._candidate_retry_cooldown_s = 10.0
    node._visual_candidate_stability_radius_m = 0.75
    node._visual_candidate_required_frames = 3
    node._pending_visual_candidate = None
    node._pending_visual_candidate_frames = 0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=123_000_000_000)
    )
    events = []
    starts = []
    node._publish_event = events.append
    node._start_visual_approach = lambda *args: starts.append(args)

    node._observe_visual_candidate(1.0, 2.0, 0.50)
    node._observe_visual_candidate(1.2, 2.1, 0.55)
    assert starts == []
    node._observe_visual_candidate(1.1, 2.0, 0.60)

    assert len(starts) == 1
    assert starts[0][3] == "status"
    assert events == [
        "mission_visual_candidate_stabilizing:1/3",
        "mission_visual_candidate_stabilizing:2/3",
    ]


def test_visual_candidate_position_jump_restarts_stability_window() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.SEARCHING
    node._locked_targets = []
    node._ignored_target_radius_m = 0.8
    node._rejected_candidates = deque(maxlen=50)
    node._candidate_confidence_threshold = 0.30
    node._candidate_reject_radius_m = 0.8
    node._candidate_retry_cooldown_s = 10.0
    node._visual_candidate_stability_radius_m = 0.40
    node._visual_candidate_required_frames = 3
    node._pending_visual_candidate = None
    node._pending_visual_candidate_frames = 0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=123_000_000_000)
    )
    node._publish_event = lambda _event: None
    starts = []
    node._start_visual_approach = lambda *args: starts.append(args)

    node._observe_visual_candidate(1.0, 2.0, 0.50)
    node._observe_visual_candidate(3.0, 2.0, 0.55)
    node._observe_visual_candidate(3.1, 2.0, 0.55)

    assert starts == []
    assert node._pending_visual_candidate_frames == 2


def test_single_target_candidate_failure_stops_without_resuming_frontier() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.APPROACHING
    node._candidate_xy = None
    node._rejected_candidates = deque()
    node._stop_after_target_interaction = True
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=123_000_000_000)
    )
    canceled = []
    events = []
    resets = []
    node._cancel_active_goal = canceled.append
    node._publish_event = events.append
    node._reset_to_idle = resets.append
    node.get_logger = lambda: _Logger()

    node._abandon_visual_candidate("approach_not_close")

    assert canceled == ["visual_candidate_approach_not_close"]
    assert events == [
        "mission_visual_candidate_abandoned:approach_not_close"
    ]
    assert resets == [
        "mission_single_target_stopped:candidate_approach_not_close"
    ]


def test_visual_target_needs_three_high_confidence_frames_after_arrival() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.VERIFYING_TARGET
    node._target_pose = None
    node._target_loss_confidence_threshold = 0.25
    node._target_loss_required_frames = 5
    node._target_confirm_confidence_threshold = 0.60
    node._target_confirm_required_frames = 3
    node._target_high_confidence_frames = 0
    node._target_low_confidence_frames = 0
    confirmed = []
    node._confirm_visual_target = confirmed.append

    node._observe_approach_confidence(0.65)
    node._observe_approach_confidence(0.70)
    assert confirmed == []
    node._observe_approach_confidence(0.75)

    assert confirmed == [0.75]


def test_arm_integration_hands_off_to_gemini_at_geometric_standoff() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._arm_integration_enabled = True
    node._approach_standoff_m = 0.45
    node._approach_arrival_tolerance_m = 0.25
    node._robot_xy = (0.55, 2.0)
    node._target_pose = PoseStamped()
    node._target_pose.pose.position.x = 1.0
    node._target_pose.pose.position.y = 2.0
    events = []
    accepted = []
    node._publish_event = events.append
    node._accept_confirmed_target = lambda x, y, **kwargs: accepted.append(
        (x, y, kwargs)
    )

    node._on_approach_succeeded()

    assert accepted == [
        (1.0, 2.0, {"source": "visual_standoff", "confidence": None})
    ]
    assert events == [
        "mission_arm_handoff_started:reason=standoff_reached,x=1.00,y=2.00"
    ]


def test_rviz_manual_goal_arrival_hands_off_to_arm_camera() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._goal_label = "manual_grasp"
    node._target_pose = PoseStamped()
    node._target_pose.pose.position.x = 1.25
    node._target_pose.pose.position.y = -0.50
    events = []
    handoffs = []
    node._publish_event = events.append
    node._begin_arm_grasp = lambda x, y, **kwargs: handoffs.append(
        (x, y, kwargs)
    )

    node._on_goal_succeeded()

    assert handoffs == [
        (1.25, -0.50, {"source": "manual_rviz", "confidence": None})
    ]
    assert events == ["manual_grasp_goal_arrived:x=1.25,y=-0.50"]


def test_approach_tolerates_temporary_visual_loss_before_grace_deadline() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.APPROACHING
    node._target_pose = None
    node._target_loss_confidence_threshold = 0.25
    node._target_loss_required_frames = 2
    node._target_confirm_confidence_threshold = 0.60
    node._target_confirm_required_frames = 3
    node._target_high_confidence_frames = 0
    node._target_low_confidence_frames = 0
    node._approach_visual_loss_deadline = 110.0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    )
    abandoned = []
    node._abandon_visual_candidate = abandoned.append

    node._observe_approach_confidence(0.10)
    node._observe_approach_confidence(None)

    assert abandoned == []


def test_approach_abandons_only_after_visual_loss_grace_deadline() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.APPROACHING
    node._target_pose = None
    node._target_loss_confidence_threshold = 0.25
    node._target_loss_required_frames = 2
    node._target_confirm_confidence_threshold = 0.60
    node._target_confirm_required_frames = 3
    node._target_high_confidence_frames = 0
    node._target_low_confidence_frames = 1
    node._approach_visual_loss_deadline = 100.0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    )
    abandoned = []
    node._abandon_visual_candidate = abandoned.append

    node._observe_approach_confidence(None)

    assert abandoned == ["confidence_lost"]


def test_locked_approach_ignores_near_field_camera_loss() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.APPROACHING
    node._approach_target_locked = True
    node._target_pose = None
    node._target_loss_confidence_threshold = 0.25
    node._target_loss_required_frames = 1
    node._target_high_confidence_frames = 0
    node._target_low_confidence_frames = 0
    abandoned = []
    node._abandon_visual_candidate = abandoned.append

    node._observe_approach_confidence(None)

    assert abandoned == []
    assert node._target_low_confidence_frames == 0


def test_locked_approach_geometric_arrival_confirms_without_new_camera_frame() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._approach_target_locked = True
    node._arm_integration_enabled = False
    node._approach_standoff_m = 0.45
    node._approach_arrival_tolerance_m = 0.25
    node._robot_xy = (0.55, 2.0)
    node._target_pose = PoseStamped()
    node._target_pose.pose.position.x = 1.0
    node._target_pose.pose.position.y = 2.0
    events = []
    accepted = []
    node._publish_event = events.append
    node._accept_confirmed_target = lambda x, y, **kwargs: accepted.append(
        (x, y, kwargs)
    )

    node._on_approach_succeeded()

    assert accepted == [
        (1.0, 2.0, {"source": "locked_visual_standoff", "confidence": None})
    ]
    assert events == [
        "mission_target_geometric_arrival_confirmed:locked_visual,x=1.00,y=2.00"
    ]


def _make_arrival_node(**overrides):
    """Approach-arrival fixture: frozen anchor at (1.0, 2.0), ring [0.10, 0.80]."""
    from builtin_interfaces.msg import Time

    node = object.__new__(MissionSupervisorNode)
    node._approach_target_locked = True
    node._arm_integration_enabled = False
    node._approach_standoff_m = 0.45
    node._approach_arrival_tolerance_m = 0.35
    node._approach_topup_max = 1
    node._approach_topup_used = 0
    node._approach_direct_topup = False
    node._approach_topup_max_distance_m = 1.6
    node._state = MissionState.APPROACHING
    node._target_pose = PoseStamped()
    node._target_pose.pose.position.x = 1.0
    node._target_pose.pose.position.y = 2.0
    node._robot_xy = (0.0, 2.0)  # 1.0m away: outside the ring, within top-up range
    node.events = []
    node._publish_event = node.events.append
    node.accepted = []
    node._accept_confirmed_target = lambda x, y, **kwargs: node.accepted.append(
        (x, y)
    )
    node.failed = []
    node._on_approach_failed = node.failed.append
    node.get_logger = lambda: _Logger()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000, to_msg=lambda: Time())
    )
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_approach_topup_walks_one_direct_hop_when_nav2_stops_short() -> None:
    node = _make_arrival_node()

    node._on_approach_succeeded()

    # No confirmation and no failure yet: a direct top-up walk is requested.
    assert node.accepted == []
    assert node.failed == []
    assert node._state == MissionState.APPROACHING
    assert node._approach_topup_used == 1
    assert node._approach_direct_topup is True
    assert node.events == [
        "mission_approach_topup_started:attempt=1,dist=1.000,max=1.600"
    ]


def test_locked_near_field_stall_enters_geometric_topup_early() -> None:
    node = _make_arrival_node(
        _robot_xy=(0.50, 2.0),  # 0.50m away: near field, outside 0.35m edge
        _approach_arrival_max_distance_m=0.35,
    )
    node._goal_pending = False
    node._goal_active = True
    node._goal_label = "approach"
    node._goal_handle = None
    node._goal_generation = 1
    node._goal_start_time = 90.0
    node._approach_timeout_s = 25.0
    node._approach_near_stall_distance_m = 0.70
    node._approach_near_stall_timeout_s = 4.0
    node._approach_progress_epsilon_m = 0.04
    node._approach_best_distance_m = 0.51
    node._approach_last_progress_time = 95.0

    node._check_timeouts(100.0)

    assert node._goal_active is False
    assert node._approach_topup_used == 1
    assert node._approach_direct_topup is True
    assert node.events == [
        "mission_approach_near_field_stall:dist=0.500,no_progress=5.0s",
        "mission_approach_near_field_stall_geometric_recheck",
        "mission_approach_topup_started:attempt=1,dist=0.500,max=1.600",
    ]


def test_topup_goal_is_sent_directly_toward_the_frozen_anchor() -> None:
    node = _make_arrival_node()
    node._approach_direct_topup = True
    node._approach_standoff_m = 0.45
    node._map_frame = "map"
    # A truthy map must NOT route the top-up through the standoff ring picker.
    node._latest_map = SimpleNamespace(info=SimpleNamespace(width=4))
    ring_calls = []
    node._map_standoff_goal = lambda *args: (
        ring_calls.append(args),
        ((9.0, 9.0), 0.0),
    )
    node._last_approach_goal_xy = None
    sent = []
    node._send_goal = lambda goal, label: sent.append(
        (label, goal.pose.pose.position.x, goal.pose.pose.position.y)
    )
    node._publish_goal_pose = lambda goal: None

    node._send_approach_goal()

    assert ring_calls == []
    # Goal sits on the robot->target ray, 0.45m short of the anchor.
    assert sent[0][0] == "approach"
    assert abs(sent[0][1] - 0.55) < 1e-9
    assert abs(sent[0][2] - 2.0) < 1e-9
    assert node._approach_direct_topup is False  # one-shot flag consumed
    assert abs(node._last_approach_goal_xy[0] - 0.55) < 1e-9
    assert abs(node._last_approach_goal_xy[1] - 2.0) < 1e-9


def test_topup_budget_exhausted_falls_back_to_failure() -> None:
    node = _make_arrival_node(_approach_topup_used=1)

    node._on_approach_succeeded()

    assert node.accepted == []
    assert node.failed == ["approach_not_close"]
    assert node._approach_direct_topup is False
    assert (
        "mission_approach_failed:approach_not_close:dist=1.000,max=0.800"
        in node.events
    )


def test_topup_skipped_beyond_max_distance() -> None:
    node = _make_arrival_node(_robot_xy=(-1.5, 2.0))  # 2.5m: beyond 1.6m bound

    node._on_approach_succeeded()

    assert node.failed == ["approach_not_close"]
    assert node._approach_topup_used == 0


def test_topup_not_triggered_when_too_close() -> None:
    node = _make_arrival_node(_robot_xy=(0.92, 2.0))  # 0.08m: below the 0.10m edge

    node._on_approach_succeeded()

    # Backing away from an obstacle-level approach is riskier than failing;
    # too-close keeps the plain failure path.
    assert node.failed == ["approach_not_close"]
    assert node._approach_topup_used == 0


def test_sport_mode_confirms_posture_api_mapping() -> None:
    assert sport_mode_confirms_api(1005, 5)
    assert sport_mode_confirms_api(1004, 6)
    assert sport_mode_confirms_api(1002, 1)
    assert not sport_mode_confirms_api(1004, 5)
    assert not sport_mode_confirms_api(1008, 1)


def test_measured_sport_mode_falls_back_when_response_is_missing() -> None:
    from unitree_go.msg import SportModeState

    node = object.__new__(MissionSupervisorNode)
    node._target_posture_expected_api = 1004
    node._target_posture_ack = None
    node._target_posture_ack_info = ""
    node.get_logger = lambda: _Logger()
    msg = SportModeState()
    msg.mode = 6

    node._on_sport_mode_state(msg)

    assert node._target_posture_ack is True
    assert node._target_posture_ack_info == "confirmed_by_sport_mode:6"


def test_posture_advances_only_after_matching_firmware_response() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._target_posture_expected_api = 1005
    node._target_posture_ack = None
    node._target_posture_ack_info = ""

    node._on_sport_status(
        String(data='{"stage":"request","api_id":1005,"succeed":true}')
    )
    assert node._target_posture_ack is None

    node._on_sport_status(
        String(data='{"stage":"response","api_id":1004,"succeed":true}')
    )
    assert node._target_posture_ack is None

    node._on_sport_status(
        String(
            data='{"stage":"response","api_id":1005,'
            '"succeed":true,"info":"accepted"}'
        )
    )
    assert node._target_posture_ack is True
    assert node._target_posture_ack_info == "accepted"


def test_posture_guard_settle_requests_pre_crouch_stand() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._target_posture_phase = "POSTURE_GUARDS_SETTLE"
    node._target_posture_ack_timeout_s = 2.0
    node._sport_stand_up_client = object()
    calls = []
    events = []
    node._call_trigger = lambda client, label: calls.append(label) or True
    node._publish_event = events.append

    node._step_target_posture(10.0)

    assert node._target_posture_phase == "PRE_CROUCH_STAND_WAIT_ACK"
    assert node._target_posture_expected_api == 1004
    assert calls == ["sport StandUp"]
    assert events == ["mission_target_pre_crouch_stand_requested"]


def test_pre_crouch_settle_requests_stand_down() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._target_posture_phase = "PRE_CROUCH_STAND_SETTLE"
    node._target_posture_ack_timeout_s = 2.0
    node._sport_stand_down_client = object()
    calls = []
    events = []
    node._call_trigger = lambda client, label: calls.append(label) or True
    node._publish_event = events.append

    node._step_target_posture(10.0)

    assert node._target_posture_phase == "CROUCH_WAIT_ACK"
    assert node._target_posture_expected_api == 1005
    assert calls == ["sport StandDown"]
    assert events == ["mission_target_crouch_requested"]


def test_scan_exhaustion_resumes_frontier_instead_of_returning() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._scan_attempts = 3
    node._scan_attempts_max = 3
    node._reached_target_count = 0
    node._target_goal_count = 3
    events = []
    resumed = []
    node._publish_event = events.append
    node._resume_search = resumed.append
    node.get_logger = lambda: _Logger()

    node._advance_scan()

    assert events == ["mission_scan_attempts_exhausted"]
    assert resumed == ["scan_attempts_exhausted"]


def test_return_guard_blocks_early_return() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.HOLDING
    node._reached_target_count = 2
    node._target_goal_count = 3
    resumed = []
    node._resume_search = resumed.append
    node.get_logger = lambda: _Logger()

    node._begin_return()

    assert node._state == MissionState.HOLDING
    assert resumed == ["return_guard"]


def test_successful_return_transition_runs_prealignment_first() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.HOLDING
    node._reached_target_count = 3
    node._target_goal_count = 3
    events = []
    starts = []
    node._publish_event = events.append
    node._start_return_alignment_or_goal = lambda: starts.append("prealign")
    node.get_logger = lambda: _Logger()

    node._begin_return()

    assert node._state == MissionState.RETURNING
    assert events == ["mission_return_started"]
    assert starts == ["prealign"]


def test_arm_enabled_confirmation_waits_for_grasp_before_counting() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._arm_integration_enabled = True
    node._locked_targets = []
    node._reached_target_count = 0
    handoffs = []
    node._begin_arm_grasp = lambda *args, **kwargs: handoffs.append((args, kwargs))
    # The dog -> arm target report has its own tests; keep this one focused
    # on the grasp-before-counting handshake.
    node._publish_arm_target_report = lambda *args, **kwargs: None

    node._accept_confirmed_target(1.0, 2.0, source="visual", confidence=0.75)

    assert node._locked_targets == [(1.0, 2.0)]
    assert node._reached_target_count == 0
    assert handoffs == [((1.0, 2.0), {"source": "visual", "confidence": 0.75})]


def test_posture_enabled_arrival_is_not_counted_before_posture_success() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._arm_integration_enabled = False
    node._target_posture_enabled = True
    node._locked_targets = []
    node._reached_target_count = 1
    node._pending_target_completion = None
    node._publish_arm_target_report = lambda *args, **kwargs: None
    holds = []
    node._enter_confirmed_target_hold = (
        lambda *args, **kwargs: holds.append((args, kwargs))
    )

    node._accept_confirmed_target(1.0, 2.0, source="visual", confidence=0.75)

    assert node._reached_target_count == 1
    assert node._pending_target_completion == (1.0, 2.0, "visual", 0.75)
    assert node._locked_targets == [(1.0, 2.0)]
    assert holds == [((1.0, 2.0), {"source": "visual", "confidence": 0.75})]


def test_first_posture_rejection_changes_to_balance_and_retries_once() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._pending_target_completion = (1.0, 2.0, "visual", 0.75)
    node._locked_targets = [(1.0, 2.0)]
    node._reached_target_count = 1
    node._target_posture_ack_info = "firmware rejected"
    node._target_posture_retry_count = 0
    node._target_posture_retry_max = 1
    node._target_posture_ack_timeout_s = 5.0
    node._sport_balance_stand_client = object()
    calls = []
    node._call_trigger = lambda client, label: calls.append(label) or True
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=123_000_000_000)
    )
    node.get_logger = lambda: _Logger()
    events = []
    node._publish_event = events.append

    node._fail_target_posture("stand_down_rejected")

    assert node._reached_target_count == 1
    assert node._pending_target_completion == (1.0, 2.0, "visual", 0.75)
    assert node._locked_targets == [(1.0, 2.0)]
    assert node._target_posture_retry_count == 1
    assert node._target_posture_phase == "RETRY_BALANCE_WAIT_ACK"
    assert calls == ["sport BalanceStand before posture retry"]
    assert any("posture_retry_requested:1/1" in event for event in events)


def test_second_posture_failure_records_and_skips_target() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._pending_target_completion = (1.0, 2.0, "visual", 0.75)
    node._locked_targets = [(1.0, 2.0)]
    node._pending_targets = [(1.1, 2.0, 0.8), (5.0, 6.0, 0.7)]
    node._ignored_target_radius_m = 0.8
    node._unserviceable_targets = []
    node._reached_target_count = 1
    node._target_posture_ack_info = "firmware rejected"
    node._target_posture_retry_count = 1
    node._target_posture_retry_max = 1
    node._target_pose = "target"
    node._candidate_xy = (1.0, 2.0)
    node._stop_after_target_interaction = False
    node._slam_jump_guard_monitoring_client = object()
    node._velocity_odom_jump_monitoring_client = object()
    node._map_accumulation_client = object()
    node._call_set_bool = lambda *args, **kwargs: True
    node.get_logger = lambda: _Logger()
    events = []
    resumed = []
    node._publish_event = events.append
    node._resume_search = resumed.append

    node._fail_target_posture("stand_down_rejected")

    assert node._reached_target_count == 1
    assert node._pending_target_completion is None
    assert node._locked_targets == [(1.0, 2.0)]
    assert node._pending_targets == [(5.0, 6.0, 0.7)]
    assert node._unserviceable_targets == [
        (1.0, 2.0, "stand_down_rejected:firmware rejected")
    ]
    assert resumed == ["target_unserviceable"]
    assert any(event.startswith("mission_target_unserviceable:") for event in events)
    assert any("not_counted" in event for event in events)


def test_posture_success_commits_pending_target_once() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._target_posture_phase = "STAND_SETTLE"
    node._pending_target_completion = (1.0, 2.0, "visual", 0.75)
    node._reached_target_count = 1
    node._target_goal_count = 3
    node._slam_jump_guard_monitoring_client = object()
    node._velocity_odom_jump_monitoring_client = object()
    node._map_accumulation_client = object()
    node._call_set_bool = lambda *args, **kwargs: True
    node._clear_target_posture_ack = lambda: None
    finished = []
    node._finish_target_hold = lambda: finished.append(True)
    events = []
    node._publish_event = events.append

    node._step_target_posture(10.0)

    assert node._reached_target_count == 2
    assert node._pending_target_completion is None
    assert "mission_target_posture_finished" in events
    assert any(event.startswith("mission_target_confirmed:2/3") for event in events)
    assert finished == [True]


def test_approach_timeout_rechecks_frozen_target_geometry() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._goal_label = "approach"
    node._target_pose = PoseStamped()
    node._robot_xy = (0.0, 0.0)
    events = []
    reconciled = []
    node._publish_event = events.append
    node._on_approach_succeeded = lambda: reconciled.append(True)

    node._on_goal_failed("timeout")

    assert node._goal_label == ""
    assert events == ["mission_approach_timeout_geometric_recheck"]
    assert reconciled == [True]


def test_successful_arm_grasp_counts_target_then_enters_hold() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._arm_target_xy = (1.0, 2.0)
    node._arm_target_source = "visual"
    node._arm_target_confidence = 0.80
    node._reached_target_count = 1
    node._target_goal_count = 3
    events = []
    holds = []
    node._publish_event = events.append
    node._clear_arm_action_state = lambda: None
    node._enter_confirmed_target_hold = (
        lambda *args, **kwargs: holds.append((args, kwargs))
    )

    node._finish_arm_grasp(True, "done")

    assert node._reached_target_count == 2
    assert events == ["mission_arm_grasp_succeeded:2/3"]
    assert holds == [
        ((1.0, 2.0), {"source": "visual+grasp", "confidence": 0.80})
    ]


def test_failed_arm_grasp_does_not_count_and_resumes_search() -> None:
    node = object.__new__(MissionSupervisorNode)
    node._arm_target_xy = (1.0, 2.0)
    node._arm_target_source = "visual"
    node._arm_target_confidence = 0.80
    node._reached_target_count = 1
    node._target_pose = "target"
    node._candidate_xy = (1.0, 2.0)
    events = []
    resumed = []
    node._publish_event = events.append
    node._clear_arm_action_state = lambda: None
    node._resume_search = resumed.append
    node.get_logger = lambda: _Logger()

    node._finish_arm_grasp(False, "goal_rejected")

    assert node._reached_target_count == 1
    assert events == ["mission_arm_grasp_failed:goal_rejected"]
    assert resumed == ["arm_grasp_failed"]
    assert node._target_pose is None
    assert node._candidate_xy is None
