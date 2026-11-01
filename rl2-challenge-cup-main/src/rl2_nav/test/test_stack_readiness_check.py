"""Regression tests for deterministic stack/mission readiness policy."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "stack_readiness_check.py"
SPEC = spec_from_file_location("stack_readiness_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def healthy_snapshots(*, gates_enabled: bool = False):
    return {
        "supervisor": {"state": "IDLE", "target_goal_count": 3},
        "coordinator": {"phase": "IDLE", "armed": False},
        "guard": {
            "state": "MONITORING",
            "monitoring_enabled": True,
            "pose_stream_lost": False,
        },
        "gate1": {"enabled": gates_enabled},
        "gate2": {"enabled": gates_enabled},
        "health": {"slam_pose_ok": True, "navigation_map_ok": True},
    }


def test_startup_requires_clean_idle_stack_with_closed_gates() -> None:
    assert MODULE.validate_snapshots(
        healthy_snapshots(), target_count=3, require_gates_enabled=False
    ) == []


def test_mission_start_requires_both_gates_enabled() -> None:
    failures = MODULE.validate_snapshots(
        healthy_snapshots(), target_count=3, require_gates_enabled=True
    )
    assert "Gate-1 must be enabled" in failures
    assert "Gate-2 must be enabled" in failures
    assert MODULE.validate_snapshots(
        healthy_snapshots(gates_enabled=True),
        target_count=3,
        require_gates_enabled=True,
    ) == []


def test_residual_coordinator_state_blocks_false_ready() -> None:
    snapshots = healthy_snapshots()
    snapshots["coordinator"] = {"phase": "MISSION_RUNNING", "armed": True}
    failures = MODULE.validate_snapshots(
        snapshots, target_count=3, require_gates_enabled=False
    )
    assert any("coordinator is not reusable" in failure for failure in failures)


def test_safe_terminal_coordinator_states_are_reusable() -> None:
    for phase in (
        "ABORTED_BOX_KEPT_CLOSED",
        "INTERRUPTED_BOX_KEPT_CLOSED",
        "ERROR_BOX_KEPT_CLOSED",
    ):
        snapshots = healthy_snapshots()
        snapshots["coordinator"] = {"phase": phase, "armed": False}
        assert MODULE.validate_snapshots(
            snapshots, target_count=3, require_gates_enabled=False
        ) == []


def test_slam_and_target_count_mismatch_are_reported() -> None:
    snapshots = healthy_snapshots()
    snapshots["supervisor"]["target_goal_count"] = 2
    snapshots["health"]["slam_pose_ok"] = False
    snapshots["guard"]["pose_stream_lost"] = True
    failures = MODULE.validate_snapshots(
        snapshots, target_count=3, require_gates_enabled=False
    )
    assert "target count mismatch: expected 3, got 2" in failures
    assert "SLAM pose is not healthy" in failures
    assert "SLAM pose stream is lost" in failures


def test_non_box_diagnostic_mode_does_not_require_coordinator() -> None:
    snapshots = healthy_snapshots()
    snapshots.pop("coordinator")
    snapshots["supervisor"]["target_goal_count"] = 1
    assert MODULE.validate_snapshots(
        snapshots, target_count=1, require_gates_enabled=False
    ) == []
