"""Tests for the mission supervisor's SLAM-loss watchdog and scan recovery.

The failure these guard against was observed on 2026-09-26: the dock-side
unitree_slam stopped publishing, the pose-jump guard latched RECOVERING with
both motion gates closed, and the supervisor kept issuing goals and waiting
out full scan-segment timeouts for over ten minutes.
"""

from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from rl2_nav.mission_supervisor_core import MissionState
from rl2_nav.mission_supervisor_node import MissionSupervisorNode


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


def _make_supervisor(**overrides):
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.SEARCHING
    node._slam_loss_abort_enabled = True
    node._slam_guard_state = "MONITORING"
    node._current_pose_at_s = 0.0
    node._robot_xy = None
    node._mission_start_time = 0.0
    node._slam_pose_max_age_s = 5.0
    node._slam_loss_confirm_s = 2.0
    node._slam_loss_since = 0.0
    node._target_posture_phase = ""
    node._scan_failed_segments_streak = 0
    node._scan_failed_segments_abort_max = 4
    node._scan_segment_index = 0
    node._scan_segments_per_rotation = 6
    node._scan_segment_pause_s = 1.0
    node._scan_best_candidate = None
    node._target_pose = object()
    node._candidate_xy = (1.0, 2.0)
    node._pending_targets = [(2.0, 3.0, 0.8)]
    node._locked_targets = [(4.0, 5.0)]
    node._base_pose = object()
    node._scan_visited_points = {(1.0, 2.0)}
    node._rejected_candidates = {(3.0, 4.0)}
    node._clear_pending_visual_candidate = lambda: None
    node.get_logger = lambda: _Logger()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    )
    node._sport_stop_client = object()
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


def test_guard_recovering_aborts_after_debounce() -> None:
    node = _make_supervisor(_slam_guard_state="RECOVERING")

    node._check_slam_loss_abort(100.0)
    assert node._resets == []
    assert node._slam_loss_since == 100.0

    node._check_slam_loss_abort(102.5)

    assert node._slam_loss_since == 0.0
    assert node._events == [
        "mission_slam_lost_abort:guard_latched:RECOVERING",
    ]
    assert node._stops == [
        "disable_frontier",
        "cancel:mission_slam_lost",
        "sport StopMove after SLAM loss",
    ]
    assert node._resets == ["mission_slam_lost:guard_latched:RECOVERING"]


def test_guard_ready_for_manual_rearm_also_aborts() -> None:
    node = _make_supervisor(_slam_guard_state="READY_FOR_MANUAL_REARM")
    node._slam_loss_since = 98.0

    node._check_slam_loss_abort(101.0)

    assert node._resets == ["mission_slam_lost:guard_latched:READY_FOR_MANUAL_REARM"]


def test_stale_current_pose_aborts() -> None:
    node = _make_supervisor(
        _current_pose_at_s=90.0,
        _robot_xy=(1.0, 2.0),
    )

    node._check_slam_loss_abort(100.0)
    assert node._resets == []

    node._check_slam_loss_abort(103.0)

    assert node._resets == ["mission_slam_lost:pose_stale:age=13.0s"]
    assert node._events == [
        "mission_slam_lost_abort:pose_stale:age=13.0s",
    ]


def test_recovery_inside_debounce_window_does_not_abort() -> None:
    node = _make_supervisor(_slam_guard_state="RECOVERING")

    node._check_slam_loss_abort(100.0)
    assert node._slam_loss_since == 100.0

    # The guard returns to MONITORING and a fresh pose arrives mid-debounce.
    node._slam_guard_state = "MONITORING"
    node._current_pose_at_s = 101.0
    node._robot_xy = (1.0, 2.0)
    node._check_slam_loss_abort(101.5)
    assert node._resets == []
    assert node._slam_loss_since == 0.0

    # A later loss starts a fresh debounce window instead of aborting
    # immediately off the old timestamp.
    node._slam_guard_state = "RECOVERING"
    node._check_slam_loss_abort(110.0)
    assert node._resets == []
    assert node._slam_loss_since == 110.0


def test_posture_phase_defers_abort_until_stand_complete() -> None:
    node = _make_supervisor(
        _slam_guard_state="RECOVERING",
        _target_posture_phase="CROUCH_HOLD",
    )

    node._check_slam_loss_abort(100.0)
    node._check_slam_loss_abort(105.0)
    assert node._resets == []
    assert node._slam_loss_since == 0.0

    node._target_posture_phase = ""
    node._check_slam_loss_abort(200.0)
    assert node._resets == []
    node._check_slam_loss_abort(203.0)

    assert node._resets == ["mission_slam_lost:guard_latched:RECOVERING"]


def test_slam_loss_abort_can_be_disabled() -> None:
    node = _make_supervisor(
        _slam_guard_state="RECOVERING",
        _slam_loss_abort_enabled=False,
        _slam_loss_since=90.0,
    )

    node._check_slam_loss_abort(100.0)

    assert node._resets == []
    assert node._slam_loss_since == 90.0


def test_on_slam_guard_status_parses_latched_state() -> None:
    node = _make_supervisor()

    node._on_slam_guard_status(String(data='{"state":"RECOVERING"}'))
    assert node._slam_guard_state == "RECOVERING"
    assert node._target_pose is None
    assert node._candidate_xy is None
    assert node._pending_targets == []
    assert node._locked_targets == []
    assert node._base_pose is None
    assert node._robot_xy is None
    assert node._current_pose_at_s == 0.0
    assert node._scan_visited_points == set()
    assert node._rejected_candidates == set()

    node._on_slam_guard_status(String(data="not json"))
    assert node._slam_guard_state == "RECOVERING"

    node._on_slam_guard_status(
        String(data='{"state":"MONITORING","monitoring_enabled":true}')
    )
    assert node._slam_guard_state == "MONITORING"


def test_on_current_pose_records_receipt_time() -> None:
    node = _make_supervisor()
    msg = PoseStamped()
    msg.pose.orientation.w = 1.0

    node._on_current_pose(msg)

    assert node._robot_xy == (0.0, 0.0)
    assert node._current_pose_at_s == 100.0


def test_four_consecutive_failed_scan_segments_relocate_instead_of_abort() -> None:
    relocations = []
    node = _make_supervisor(
        _state=MissionState.SCANNING,
        _scan_failed_segments_streak=3,
        _scan_segment_index=5,
        _start_recovery=lambda pending: False,
        _relocate_after_scan_stall=relocations.append,
    )

    node._on_spin_result(GoalStatus.STATUS_ABORTED)

    assert node._events == [
        "mission_scan_motion_stalled_recovery:streak=4",
    ]
    assert node._resets == []
    assert relocations == ["recovery_unavailable"]
    assert node._scan_segment_index == 5


def test_scan_recovery_success_forces_relocation() -> None:
    relocations = []
    node = _make_supervisor(
        _retry_pending_action=MissionSupervisorNode._retry_pending_action,
        _relocate_after_scan_stall=relocations.append,
    )

    node._retry_pending_action(node, "scan")

    assert relocations == ["escape_finished"]


def test_successful_scan_segment_resets_stall_streak() -> None:
    node = _make_supervisor(
        _state=MissionState.SCANNING,
        _scan_failed_segments_streak=3,
        _scan_segment_index=4,
    )

    node._on_spin_result(GoalStatus.STATUS_SUCCEEDED)

    assert node._resets == []
    assert node._scan_failed_segments_streak == 0
    assert node._scan_segment_index == 5
