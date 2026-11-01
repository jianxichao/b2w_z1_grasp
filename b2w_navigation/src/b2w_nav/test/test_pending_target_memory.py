"""Seen-but-not-chosen target memory (20260926修改.md 第四批/第五批).

Field problem: a spin that sees two targets keeps only the highest-confidence
one, and after interacting with it the mission restarts blind frontier search
even though the runner-up's position was already known.

- scan runner-ups are recorded on replacement, and spatially separate
  lower-confidence candidates are recorded too;
- stable scan sightings are remembered; displaced observations during a
  locked approach are treated as RGB-D outliers and cannot create phantom
  targets or redirect the selected route;
- resume consumes the freshest memory entry instead of enabling frontier;
- locked/near-current-approach sightings are not re-recorded.
"""

import math
from collections import deque
from types import SimpleNamespace

from b2w_nav.mission_supervisor_core import MissionState
from b2w_nav.mission_supervisor_node import MissionSupervisorNode


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


def _pose_at(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y))
    )


def _make_node(**overrides):
    node = object.__new__(MissionSupervisorNode)
    node._state = MissionState.SCANNING
    node._locked_targets = []
    node._ignored_target_radius_m = 0.8
    node._target_association_radius_m = 0.50
    node._candidate_xy = None
    node._robot_xy = None
    node._pending_targets = []
    node._scan_best_candidate = None
    node._rejected_candidates = deque()
    node._candidate_reject_radius_m = 0.8
    node._candidate_retry_cooldown_s = 10.0
    node._locked_approach_min_confidence = 0.40
    node._target_pose = None
    node._candidate_confidence_threshold = 0.40
    node._target_loss_confidence_threshold = 0.25
    node._target_loss_required_frames = 10
    node._target_low_confidence_frames = 0
    node._target_high_confidence_frames = 0
    node._target_confirm_confidence_threshold = 0.50
    node._target_confirm_required_frames = 3
    node._approach_visual_loss_grace_s = 12.0
    node._approach_visual_loss_deadline = 0.0
    node._approach_source = ""
    node._memory_reanchor_candidate = None
    node._memory_reanchor_frames = 0
    node._memory_reanchor_used = False
    node._memory_reanchor_required_frames = 4
    node._memory_reanchor_stability_radius_m = 0.35
    node._memory_reanchor_max_correction_m = 1.50
    # Existing unit cases focus on association/re-anchor mechanics.  Keep
    # their synthetic geometry permissive; dedicated cases below exercise
    # the measured camera visibility envelope.
    node._memory_reanchor_min_visible_distance_m = 0.0
    node._memory_reanchor_max_visible_distance_m = 100.0
    node._memory_reanchor_half_fov_rad = math.pi
    node._memory_reanchor_camera_forward_offset_m = 0.15
    node._robot_yaw = 0.0
    values = {"pending_target_memory_enabled": True}
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = lambda: _Logger()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    )
    events = []
    node._publish_event = events.append
    node._events = events
    node._clear_frontier_unavailable = lambda: None
    node._enable_frontier = lambda: events.append("frontier_enabled")
    approaches = []
    node._start_visual_approach = (
        lambda x, y, conf, source: (
            approaches.append((x, y, conf, source)),
            setattr(node, "_candidate_xy", (x, y)),
        )
    )
    node._approaches = approaches
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


# ---------------------------------------------------------------------------
# recording during a scan
# ---------------------------------------------------------------------------


def test_scan_runner_up_is_recorded_on_replacement() -> None:
    node = _make_node()

    node._note_candidate_pose(1.0, 1.0, 0.60)
    node._note_candidate_pose(4.0, 5.0, 0.75)

    assert node._scan_best_candidate == (4.0, 5.0, 0.75)
    assert node._pending_targets == [(1.0, 1.0, 0.60)]


def test_lower_confidence_separate_candidate_is_also_recorded() -> None:
    node = _make_node()

    node._note_candidate_pose(1.0, 1.0, 0.60)
    node._note_candidate_pose(4.0, 5.0, 0.50)

    assert node._scan_best_candidate == (1.0, 1.0, 0.60)
    assert node._pending_targets == [(4.0, 5.0, 0.50)]


def test_near_best_duplicate_is_not_recorded() -> None:
    node = _make_node()

    node._note_candidate_pose(1.0, 1.0, 0.60)
    node._note_candidate_pose(1.3, 0.9, 0.50)  # same object, lower conf
    node._note_candidate_pose(1.2, 1.1, 0.75)  # same object, fresher estimate

    assert node._scan_best_candidate == (1.2, 1.1, 0.75)
    assert node._pending_targets == []


def test_locked_candidate_is_not_noted() -> None:
    node = _make_node()
    node._locked_targets = [(1.0, 1.0)]

    node._note_candidate_pose(1.1, 1.0, 0.90)

    assert node._scan_best_candidate is None
    assert node._pending_targets == []


def test_confidence_free_pose_during_scan_cannot_bypass_stability_filter() -> None:
    node = _make_node(_state=MissionState.SCANNING)
    node._pose_xy_in_map = lambda msg: (5.0, 6.0)
    node._pose_inside_fence = lambda x, y: True

    node._on_target_pose(SimpleNamespace())

    assert node._scan_best_candidate is None
    assert node._pending_targets == []


# ---------------------------------------------------------------------------
# recording during an approach / verification (real callback paths)
# ---------------------------------------------------------------------------


def test_unassociated_sighting_during_approach_is_recorded_not_fed() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        # Grace long expired (now=100s): any refresh would move it forward.
        _approach_visual_loss_deadline=50.0,
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)

    node._observe_approach_confidence(0.80, 5.0, 6.0)

    assert node._pending_targets == [(5.0, 6.0, 0.80)]
    # No new approach was started for the distracting sighting.
    assert node._approaches == []
    # B's detection must not refresh A's grace window.
    assert node._approach_visual_loss_deadline == 50.0
    assert node._target_low_confidence_frames == 1


def test_unassociated_sighting_during_locked_approach_is_ignored() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_target_locked=True,
        _approach_visual_loss_deadline=50.0,
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)

    node._observe_approach_confidence(0.95, 5.0, 6.0)

    assert node._pending_targets == []
    assert node._target_pose.pose.position.x == 2.0
    assert node._target_low_confidence_frames == 0


def test_associated_detection_still_refreshes_approach_grace() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_visual_loss_deadline=50.0,
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)

    node._observe_approach_confidence(0.80, 2.3, 2.2)

    assert node._pending_targets == []
    assert node._approach_visual_loss_deadline == 112.0  # 100 + 12


def test_pending_memory_is_reanchored_once_after_stable_fresh_frames() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_source="pending_memory",
        _approach_visual_loss_deadline=50.0,
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)
    node._approach_standoff_attempts = 1
    node._approach_topup_used = 1
    node._approach_direct_topup = True
    node._last_approach_goal_xy = (1.7, 2.0)
    node._failed_approach_goals = deque([(1.7, 2.0)])
    node._goal_cancel_settle_s = 0.5
    node._pose_message = lambda x, y, yaw: _pose_at(x, y)
    canceled = []
    node._cancel_nav_goal = canceled.append

    for x, y in ((2.85, 2.02), (2.88, 2.00), (2.86, 2.04)):
        node._observe_approach_confidence(0.80, x, y)
        assert node._approach_source == "pending_memory"

    node._observe_approach_confidence(0.82, 2.87, 2.01)

    assert canceled == ["pending_memory_reanchor"]
    assert node._approach_source == "pending_memory_reanchored"
    assert node._approach_target_locked is True
    assert node._memory_reanchor_used is True
    assert math.isclose(node._target_pose.pose.position.x, 2.865, abs_tol=0.02)
    assert math.isclose(node._target_pose.pose.position.y, 2.018, abs_tol=0.02)
    assert node._next_goal_allowed_time == 100.5
    assert any(
        event.startswith("mission_pending_target_reanchored:")
        for event in node._events
    )

    # Once corrected, the anchor is locked again. A displaced estimate in the
    # same single-target stream is treated as a depth outlier, not another
    # target; multi-target memory is collected during scanning.
    node._observe_approach_confidence(0.90, 3.8, 2.0)
    assert math.isclose(node._target_pose.pose.position.x, 2.865, abs_tol=0.02)
    assert node._pending_targets == []


def test_accurate_pending_memory_is_freshly_validated_before_near_field_loss() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_source="pending_memory",
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)
    node._approach_standoff_attempts = 0
    node._approach_topup_used = 0
    node._approach_direct_topup = False
    node._last_approach_goal_xy = None
    node._failed_approach_goals = deque()
    node._goal_cancel_settle_s = 0.5
    node._pose_message = lambda x, y, yaw: _pose_at(x, y)
    canceled = []
    node._cancel_nav_goal = canceled.append

    for x, y in ((2.10, 2.02), (2.08, 2.00), (2.09, 2.01), (2.10, 2.00)):
        node._observe_approach_confidence(0.82, x, y)

    assert canceled == ["pending_memory_reanchor"]
    assert node._approach_source == "pending_memory_reanchored"
    assert math.isclose(node._target_pose.pose.position.x, 2.09, abs_tol=0.02)


def test_normal_visual_lock_never_reanchors_to_foreign_target() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_source="status",
        _approach_target_locked=True,
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)

    for _ in range(5):
        node._observe_approach_confidence(0.90, 2.9, 2.0)

    assert node._target_pose.pose.position.x == 2.0
    assert node._approach_source == "status"
    assert node._pending_targets == []


def test_pending_memory_freezes_inside_measured_near_blind_zone() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_source="pending_memory",
        _robot_xy=(0.0, 0.0),
        _robot_yaw=0.0,
        _memory_reanchor_min_visible_distance_m=0.65,
        _memory_reanchor_max_visible_distance_m=4.0,
        _memory_reanchor_half_fov_rad=math.radians(30.0),
    )
    # Camera is at x=0.15, so this remembered target is only 0.55m away.
    node._target_pose = _pose_at(0.70, 0.0)
    node._candidate_xy = (0.70, 0.0)

    # A stable-looking yellow distractor must not overwrite the near target.
    for _ in range(6):
        node._observe_approach_confidence(0.90, 2.0, 0.0)

    assert node._target_pose.pose.position.x == 0.70
    assert node._memory_reanchor_frames == 0
    assert node._pending_targets == []


def test_pending_memory_freezes_outside_measured_horizontal_fov() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_source="pending_memory",
        _robot_xy=(0.0, 0.0),
        _robot_yaw=0.0,
        _memory_reanchor_min_visible_distance_m=0.65,
        _memory_reanchor_max_visible_distance_m=4.0,
        _memory_reanchor_half_fov_rad=math.radians(30.0),
    )
    # The remembered target is roughly 45deg left of camera heading.
    node._target_pose = _pose_at(1.5, 1.5)
    node._candidate_xy = (1.5, 1.5)

    for _ in range(6):
        node._observe_approach_confidence(0.90, 1.7, 1.4)

    assert node._target_pose.pose.position.x == 1.5
    assert node._target_pose.pose.position.y == 1.5
    assert node._memory_reanchor_frames == 0


def test_uncorrected_memory_arrival_requires_fresh_verification() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_source="pending_memory",
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._robot_xy = (1.65, 2.0)
    node._approach_standoff_m = 0.35
    node._approach_arrival_tolerance_m = 0.35
    node._approach_topup_used = 0
    node._approach_topup_max = 1
    node._approach_topup_max_distance_m = 1.6
    node._approach_target_locked = True
    node._arm_integration_enabled = False
    node._target_verify_timeout_s = 6.0
    accepted = []
    node._accept_confirmed_target = lambda *args, **kwargs: accepted.append(
        (args, kwargs)
    )

    node._on_approach_succeeded()

    assert node._state == MissionState.VERIFYING_TARGET
    assert accepted == []


def test_reanchored_memory_keeps_near_field_geometric_lock() -> None:
    node = _make_node(
        _state=MissionState.APPROACHING,
        _approach_source="pending_memory_reanchored",
    )
    node._target_pose = _pose_at(2.0, 2.0)
    node._robot_xy = (1.65, 2.0)
    node._approach_standoff_m = 0.35
    node._approach_arrival_tolerance_m = 0.35
    node._approach_topup_used = 0
    node._approach_topup_max = 1
    node._approach_topup_max_distance_m = 1.6
    node._approach_target_locked = True
    node._arm_integration_enabled = False
    accepted = []
    node._accept_confirmed_target = lambda *args, **kwargs: accepted.append(
        (args, kwargs)
    )

    node._on_approach_succeeded()

    assert accepted == [
        ((2.0, 2.0), {"source": "locked_visual_standoff", "confidence": None})
    ]


def test_low_confidence_foreign_detection_is_not_recorded() -> None:
    node = _make_node(_state=MissionState.APPROACHING)
    node._target_pose = _pose_at(2.0, 2.0)

    node._observe_approach_confidence(0.10, 5.0, 6.0)

    assert node._pending_targets == []


def test_foreign_detection_cannot_complete_verification() -> None:
    node = _make_node(_state=MissionState.VERIFYING_TARGET)
    node._target_pose = _pose_at(2.0, 2.0)
    confirms = []
    node._confirm_visual_target = lambda conf: confirms.append(conf)

    node._observe_approach_confidence(0.90, 5.0, 6.0)
    node._observe_approach_confidence(0.90, 5.0, 6.0)
    node._observe_approach_confidence(0.90, 5.0, 6.0)

    assert node._target_high_confidence_frames == 0
    assert confirms == []
    assert node._pending_targets == [(5.0, 6.0, 0.90)]  # deduped
    assert node._target_low_confidence_frames == 3


def test_associated_detection_counts_toward_confirmation() -> None:
    node = _make_node(_state=MissionState.VERIFYING_TARGET)
    node._target_pose = _pose_at(2.0, 2.0)

    node._observe_approach_confidence(0.60, 2.2, 2.1)

    assert node._target_high_confidence_frames == 1
    assert node._pending_targets == []


def test_foreign_target_pose_message_is_ignored_and_target_untouched() -> None:
    node = _make_node(_state=MissionState.APPROACHING)
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)
    node._pose_xy_in_map = lambda msg: (5.0, 6.0)
    node._pose_inside_fence = lambda x, y: True

    node._on_target_pose(
        SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=5.0, y=6.0, z=0.0),
                orientation=SimpleNamespace(w=1.0),
            )
        )
    )

    assert node._pending_targets == []
    assert node._target_pose.pose.position.x == 2.0  # A's pose untouched


def test_associated_pose_during_approach_does_not_move_locked_target() -> None:
    node = _make_node(_state=MissionState.APPROACHING)
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)
    node._pose_xy_in_map = lambda msg: (2.3, 2.2)  # 0.36m: within association
    node._pose_inside_fence = lambda x, y: True

    node._on_target_pose(
        SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=2.3, y=2.2, z=0.0),
                orientation=SimpleNamespace(w=1.0),
            )
        )
    )

    # The locked anchor — and with it the arrival ring the Nav2 goal was
    # computed from — must not move while the goal is active (2026-09-27
    # field run: flickering depth estimates walked it mid-drive).
    assert node._target_pose.pose.position.x == 2.0
    assert node._target_pose.pose.position.y == 2.0
    assert node._pending_targets == []


def test_associated_pose_during_verifying_refines_target() -> None:
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Quaternion

    node = _make_node(_state=MissionState.VERIFYING_TARGET, _map_frame="map")
    node._target_pose = _pose_at(2.0, 2.0)
    node._candidate_xy = (2.0, 2.0)
    node._pose_xy_in_map = lambda msg: (2.3, 2.2)
    node._pose_inside_fence = lambda x, y: True
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(
            nanoseconds=100_000_000_000, to_msg=lambda: Time()
        )
    )

    node._on_target_pose(
        SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=2.3, y=2.2, z=0.0),
                orientation=Quaternion(w=1.0),
            )
        )
    )

    # The Nav2 approach goal is already complete; a close-range estimate may
    # refine the recorded target position before confirmation.
    assert node._target_pose.pose.position.x == 2.3
    assert node._target_pose.pose.position.y == 2.2
    assert node._pending_targets == []


def test_record_skips_locked_and_current_approach_and_duplicates() -> None:
    node = _make_node()
    node._locked_targets = [(0.0, 0.0)]
    node._candidate_xy = (2.0, 2.0)
    node._pending_targets = [(4.0, 4.0, 0.5)]

    # Within ignored radius of a locked target.
    node._record_pending_target(0.3, 0.2, 0.6)
    # Near the target currently being approached.
    node._record_pending_target(2.4, 2.2, 0.6)
    # Near an already remembered entry.
    node._record_pending_target(4.2, 4.2, 0.9)

    assert node._pending_targets == [(4.2, 4.2, 0.9)]

    # A genuinely new position is appended.
    node._record_pending_target(7.0, -1.0, 0.55)
    assert node._pending_targets[-1] == (7.0, -1.0, 0.55)


def test_memory_disabled_records_nothing() -> None:
    node = _make_node()
    node.get_parameter = lambda name: SimpleNamespace(value=False)

    node._record_pending_target(5.0, 6.0, None)

    assert node._pending_targets == []


# ---------------------------------------------------------------------------
# consumption on resume
# ---------------------------------------------------------------------------


def test_resume_visits_highest_confidence_memory_before_enabling_frontier() -> None:
    node = _make_node()
    node._pending_targets = [(6.0, 2.0, 0.70), (1.0, 1.0, 0.55)]

    node._resume_search("target_hold_done")

    # Higher confidence wins even when it is not newest; Frontier stays disabled.
    assert node._approaches[0][:3] == (6.0, 2.0, 0.70)
    assert node._approaches[0][3] == "pending_memory"
    assert "frontier_enabled" not in node._events


def test_resume_locks_low_confidence_memory_entries() -> None:
    node = _make_node()
    node._pending_targets = [(3.0, 3.0, 0.0)]  # recorded without confidence

    node._resume_search("target_hold_done")

    # conf lifted to the lock threshold: near-field loss cannot cancel.
    assert node._approaches[0][2] == 0.40


def test_resume_skips_locked_memory_and_falls_back_to_frontier() -> None:
    node = _make_node()
    node._pending_targets = [(3.0, 3.0, 0.55)]
    node._locked_targets = [(3.2, 3.1)]

    node._resume_search("target_hold_done")

    assert node._approaches == []
    assert node._pending_targets == []
    assert "frontier_enabled" in node._events


def test_resume_without_memory_enables_frontier() -> None:
    node = _make_node()

    node._resume_search("scan_attempts_exhausted")

    assert "frontier_enabled" in node._events
    assert node._approaches == []


def test_association_distance_math_matches_design() -> None:
    # Sanity for the fixture radii used above.
    assert math.hypot(0.3, 0.2) < 0.8  # locked-ignore case
    assert math.hypot(4.0 - 4.2, 4.0 - 4.2) <= 0.50  # duplicate case
