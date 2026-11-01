import types
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

from robot_arm.hardware.arx5_backend import Arx5Backend
from robot_arm.manipulation import MotionCommand


def config():
    return {
        "home_joints": [0, 0.645, 1.04, -0.77, 0, 0],
        "search_joints": [0, 0.645, 1.04, -0.77, 0, 0],
        "gripper_open": 0.088,
        "command_rate_hz": 50,
        "max_joint_speed_rad_s": 0.35,
        "joint_tolerance_rad": 0.06,
    }


def test_dry_run_never_loads_can_sdk(tmp_path):
    backend = Arx5Backend(tmp_path, config(), dry_run=True)
    backend.connect()
    assert backend.self_check() == (True, "")
    assert backend.latest_feedback_joints() == backend.home_joints
    assert backend.base_T_eef().shape == (4, 4)
    assert not backend.has_object()


def test_hardware_self_check_requires_advancing_timestamp(tmp_path, monkeypatch):
    backend = Arx5Backend(tmp_path, config())
    backend._last_feedback_stamp = 1.0
    monkeypatch.setattr(backend, "_feedback", lambda: backend._joints)
    assert backend.self_check() == (False, "机械臂反馈时间戳未更新")


def test_sdk_loader_rejects_external_module(tmp_path, monkeypatch):
    expected = tmp_path / "third_party/arx5_sdk/python/arx5_interface.fake.so"
    expected.parent.mkdir(parents=True)
    expected.touch()
    outside = tmp_path / "outside/arx5_interface.so"
    outside.parent.mkdir()
    outside.touch()
    fake = types.SimpleNamespace(__file__=str(outside))
    monkeypatch.setattr("importlib.import_module", lambda _: fake)
    backend = Arx5Backend(tmp_path, config())
    with pytest.raises(RuntimeError, match="项目外"):
        backend._load_sdk()


def test_sdk_loader_accepts_complete_official_package(tmp_path, monkeypatch):
    outside = tmp_path / "site-packages/arx5_interface/__init__.py"
    outside.parent.mkdir(parents=True)
    outside.touch()
    fake = types.SimpleNamespace(
        __file__=str(outside),
        RobotConfigFactory=object(),
        ControllerConfigFactory=object(),
        Arx5JointController=object(),
        JointState=object(),
        Gain=object(),
        LogLevel=object(),
    )
    monkeypatch.setattr("importlib.import_module", lambda _: fake)
    backend = Arx5Backend(tmp_path, config())
    assert backend._load_sdk() is fake


class JointState:
    def __init__(self, _dof=6):
        self.timestamp = 0.0
        self.gripper_pos = 0.088
        self.gripper_torque = 0.0
        self._pos = np.zeros(6)

    def pos(self):
        return self._pos


class Controller:
    def __init__(self):
        self.now = 1.0
        self.state = JointState()
        self.state.timestamp = self.now
        self.trajectories = []

    def get_timestamp(self):
        return self.now

    def get_joint_state(self):
        self.now += 0.01
        self.state.timestamp = self.now
        return self.state

    def get_joint_cmd(self):
        return self.state

    def set_joint_traj(self, trajectory):
        self.trajectories.append(trajectory)
        self.state._pos = trajectory[-1].pos().copy()
        self.state.gripper_pos = trajectory[-1].gripper_pos

    def set_joint_cmd(self, command):
        self.state._pos = command.pos().copy()
        self.state.gripper_pos = command.gripper_pos


class EefState:
    @staticmethod
    def pose_6d():
        return np.array([0.2, -0.1, 0.3, 0.0, 0.0, np.pi / 2])


def test_hardware_move_queues_one_sdk_trajectory(tmp_path, monkeypatch):
    backend = Arx5Backend(tmp_path, config())
    backend.arx5 = SimpleNamespace(JointState=JointState)
    backend.controller = Controller()
    backend._position_enabled = True
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    ticks = count(start=0.0, step=10.0)
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    result = backend._move((0.01, 0.645, 1.04, -0.77, 0, 0), 0.088, 0.1)
    assert result.reached
    assert len(backend.controller.trajectories) == 1
    stamps = [state.timestamp for state in backend.controller.trajectories[0]]
    assert stamps == sorted(stamps) and stamps[0] > backend.controller.now


def test_real_eye_in_hand_uses_fresh_sdk_eef_pose(tmp_path):
    backend = Arx5Backend(tmp_path, config())
    backend.controller = Controller()
    backend.controller.get_eef_state = lambda: EefState()
    transform = backend.base_T_eef()
    assert transform[:3, 3] == pytest.approx((0.2, -0.1, 0.3))
    assert np.allclose(
        transform[:3, :3],
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        atol=1e-12,
    )


def test_open_requires_measured_gripper_feedback(tmp_path, monkeypatch):
    backend = Arx5Backend(tmp_path, config())
    backend.arx5 = SimpleNamespace(JointState=JointState)
    backend.controller = Controller()
    backend.controller.state._pos = np.asarray(backend.home_joints)
    backend.controller.state.gripper_pos = 0.0
    backend._position_enabled = True
    backend.controller.set_joint_traj = lambda trajectory: None
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    ticks = count(start=0.0, step=10.0)
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    command = MotionCommand("open", backend.home_joints, 0.088, 0.1)
    assert not backend.execute(command).reached


def test_virtual_joint_drift_must_resync_or_fail(tmp_path, monkeypatch):
    backend = Arx5Backend(tmp_path, {**config(), "correction_attempts": 1})
    backend.arx5 = SimpleNamespace(JointState=JointState)
    backend.controller = Controller()
    backend.controller.state._pos = np.asarray(backend.home_joints)
    virtual = JointState()
    virtual._pos = np.asarray(backend.home_joints) + 0.3
    backend.controller.get_joint_cmd = lambda: virtual
    backend.controller.set_joint_traj = lambda trajectory: None
    backend._position_enabled = True
    ticks = iter(np.arange(0.0, 10.0, 0.01))
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="多次校正失败"):
        backend._move(backend.home_joints, 0.088, 0.01)


def test_virtual_joints_are_an_owned_copy(tmp_path):
    backend = Arx5Backend(tmp_path, config())
    backend.controller = Controller()
    copied = backend._virtual_joints()
    backend.controller.state._pos[0] = 9.0
    assert copied[0] == 0.0


def test_recovery_keeps_closed_gripper(tmp_path, monkeypatch):
    backend = Arx5Backend(tmp_path, config(), dry_run=True)
    backend._closed_commanded = True
    used = []
    monkeypatch.setattr(backend, "_move", lambda joints, gripper, duration: used.append(gripper))
    backend.recover(None)
    assert used == [0.0]
