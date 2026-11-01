"""读取机械臂启动时的关节反馈（与 teach_joint_pose 相同注意 pos().copy）。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_arm.hardware import Arx5Backend


def _validate_joints(joints: np.ndarray) -> tuple[float, ...]:
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise RuntimeError("关节反馈无效")
    return tuple(float(value) for value in joints)


def read_joints_from_backend(arm: Arx5Backend, *, samples: int = 3, interval_s: float = 0.02) -> tuple[float, ...]:
    """经 Arx5Backend 连读几次关节，取最后一次（预览阶段只读 CAN）。"""
    last: tuple[float, ...] | None = None
    for _ in range(max(1, samples)):
        last = arm.current_joints()
        time.sleep(interval_s)
    assert last is not None
    return last


def read_joints_from_controller(controller: object, *, samples: int = 3, interval_s: float = 0.02) -> tuple[float, ...]:
    """直接读 SDK JointState.pos()，必须 copy（见 teach_joint_pose._read_joint_state）。"""
    last: tuple[float, ...] | None = None
    for _ in range(max(1, samples)):
        state = controller.get_joint_state()
        joints = _validate_joints(np.array(state.pos(), dtype=float, copy=True))
        last = joints
        time.sleep(interval_s)
    assert last is not None
    return last


def format_joints(joints: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"j{i + 1}={float(v):+.4f}" for i, v in enumerate(joints))
