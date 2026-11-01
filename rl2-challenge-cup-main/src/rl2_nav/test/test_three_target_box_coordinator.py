"""Regression tests for the storage-box mission event classification."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from std_msgs.msg import String


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "three_target_box_mission.py"
SPEC = spec_from_file_location("three_target_box_mission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_return_prealign_failure_is_recoverable() -> None:
    assert not MODULE.mission_event_is_terminal_fault(
        "mission_return_prealign_failed_fallback:status_6"
    )


def test_candidate_and_posture_failures_are_recoverable() -> None:
    assert not MODULE.mission_event_is_terminal_fault(
        "mission_approach_failed:status_6"
    )
    assert not MODULE.mission_event_is_terminal_fault(
        "mission_target_posture_failed:stand_down_rejected"
    )
    assert not MODULE.mission_event_is_terminal_fault(
        "mission_arm_grasp_failed:target_not_reached"
    )
    assert not MODULE.mission_event_is_terminal_fault(
        "mission_scan_motion_stalled_recovery:streak=4"
    )


def test_terminal_mission_events_keep_box_closed() -> None:
    for event in (
        "mission_aborted",
        "mission_return_failed:status_6",
        "mission_slam_lost_abort:pose_stale",
        "mission_motion_gate_closed:cmd_vel_safety_node",
        "mission_scan_motion_stalled",
        "mission_arm_manual_reset_required",
    ):
        assert MODULE.mission_event_is_terminal_fault(event), event


def _coordinator_stub():
    node = object.__new__(MODULE.ThreeTargetBoxMission)
    node._armed = True
    node._phase = "MISSION_RUNNING"
    node._detail = "mission active"
    node._last_event = ""
    node._return_confirmed = False
    node._publish_status = lambda: None

    def set_phase(phase: str, detail: str) -> None:
        node._phase = phase
        node._detail = detail

    node._set_phase = set_phase
    return node


def test_recoverable_event_does_not_disarm_running_coordinator() -> None:
    node = _coordinator_stub()
    MODULE.ThreeTargetBoxMission._on_task_event(
        node, String(data="mission_return_prealign_failed_fallback:status_6")
    )
    assert node._armed
    assert node._phase == "MISSION_RUNNING"
    assert not node._return_confirmed


def test_base_arrival_remains_actionable_after_recoverable_event() -> None:
    node = _coordinator_stub()
    MODULE.ThreeTargetBoxMission._on_task_event(
        node, String(data="mission_return_prealign_failed_fallback:status_6")
    )
    MODULE.ThreeTargetBoxMission._on_task_event(
        node, String(data="mission_arrived_base:nav2_succeeded")
    )
    assert node._armed
    assert node._return_confirmed
    assert node._phase == "WAITING_BASE_STABLE"


def test_terminal_event_disarms_running_coordinator() -> None:
    node = _coordinator_stub()
    MODULE.ThreeTargetBoxMission._on_task_event(
        node, String(data="mission_return_failed:status_6")
    )
    assert not node._armed
    assert node._phase == "INTERRUPTED_BOX_KEPT_CLOSED"
    assert "mission_return_failed" in node._detail


def test_operator_abort_is_reusable_not_coordinator_fault() -> None:
    node = _coordinator_stub()
    MODULE.ThreeTargetBoxMission._on_task_event(
        node, String(data="mission_aborted")
    )
    assert not node._armed
    assert node._phase == "ABORTED_BOX_KEPT_CLOSED"
    assert "operator stop" in node._detail
