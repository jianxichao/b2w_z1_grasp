import pytest


try:
    import mujoco  # noqa: F401
except ImportError:
    mujoco = None


def _machine(target_position=None, physical_offset_x=0.0):
    if mujoco is None:
        pytest.skip("未安装 MuJoCo")
    from robot_arm.config import load_config
    from robot_arm.manipulation import Arx5Kinematics, GraspPlanner, MotionSettings, SafetyGuard
    from robot_arm.manipulation.state_machine import GraspStateMachine
    from robot_arm.simulation import FixedTargetProvider, MujocoBackend

    config = load_config()
    motion = config.motion
    settings = MotionSettings(
        home_joints=tuple(motion["home_joints"]),
        search_joints=tuple(motion["search_joints"]),
        open_gripper=motion["gripper"]["open_m"],
        closed_gripper=motion["gripper"]["closed_m"],
        pregrasp_offset_m=motion["planning"]["pregrasp_offset_m"],
        lift_offset_m=motion["planning"]["lift_offset_m"],
        transit_height_m=motion["planning"]["transit_height_m"],
        place_position=tuple(motion["planning"]["place_position_m"]),
        durations_s=motion["durations_s"],
    )
    kinematics = Arx5Kinematics()
    backend = MujocoBackend(config.root, config.simulation)
    if physical_offset_x:
        target_qpos = int(backend.model.joint("contact_target_freejoint").qposadr[0])
        original_connect = backend.connect

        def connect_with_misaligned_object():
            original_connect()
            backend.data.qpos[target_qpos] += physical_offset_x
            backend.mujoco.mj_forward(backend.model, backend.data)

        backend.connect = connect_with_misaligned_object
    machine = GraspStateMachine(
        backend,
        FixedTargetProvider(target_position or config.simulation["target_position_m"]),
        GraspPlanner(kinematics, settings),
        SafetyGuard(kinematics),
    )
    return machine, backend


def test_mujoco_state_machine_closed_loop():
    machine, backend = _machine()
    result = machine.run()
    assert result.success
    assert not backend.has_object()
    assert backend.target_in_storage()


def test_mujoco_misaligned_target_fails_real_contact_check():
    machine, backend = _machine(physical_offset_x=0.08)
    result = machine.run()
    assert not result.success
    assert "双侧接触" in result.error
    assert not backend.has_object()
