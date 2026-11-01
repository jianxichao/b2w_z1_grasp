"""ARX X5 的无相机 MuJoCo 后端和固定仿真目标。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..manipulation.types import ExecutionResult, MotionCommand, PickPlan, Pose, Target


class FixedTargetProvider:
    def __init__(self, position: Sequence[float], confidence: float = 1.0) -> None:
        self.target = Target("simulation_target", Pose.from_xyz(position), confidence)

    def lock(self) -> Target:
        return self.target

    def close(self) -> None:
        pass


class MujocoBackend:
    """只模拟机械臂；目标由场景提供，不创建任何相机。"""

    def __init__(self, root: Path, config: Mapping[str, Any], viewer: bool = False) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("缺少 MuJoCo，请安装可选依赖 simulation") from exc
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(root / config["model"]))
        self.data = mujoco.MjData(self.model)
        self.rate = float(config.get("command_rate_hz", 50.0))
        self.realtime = bool(config.get("realtime", viewer))
        self.viewer_requested = viewer
        self.viewer: Any | None = None
        self.joint_names = tuple(config.get("joint_names", [f"joint{i}" for i in range(1, 7)]))
        self.actuator_names = tuple(config.get("actuator_names", [f"arx_x5_joint{i}_position" for i in range(1, 7)]))
        self.gripper_name = str(config.get("gripper_actuator", "arx_x5_gripper_position"))
        self.qpos = tuple(int(self.model.joint(name).qposadr[0]) for name in self.joint_names)
        self.actuators = tuple(int(self.model.actuator(name).id) for name in self.actuator_names)
        self.gripper = int(self.model.actuator(self.gripper_name).id)
        self.home_joints = tuple(float(v) for v in config["home_joints"])
        self.search_joints = tuple(float(v) for v in config.get("search_joints", self.home_joints))
        self.opening = float(config.get("gripper_open", 0.088))
        verification = config["verification"]
        self.target_body = int(self.model.body(verification["target_body"]).id)
        self.target_geom = int(self.model.geom(verification["target_geom"]).id)
        self.left_geoms = frozenset(int(self.model.geom(name).id) for name in verification["left_gripper_geoms"])
        self.right_geoms = frozenset(int(self.model.geom(name).id) for name in verification["right_gripper_geoms"])
        self.storage_region = int(self.model.site(verification["storage_region_site"]).id)
        self.settle_steps = int(verification.get("open_settle_steps", 100))
        self.minimum_lift = float(verification.get("minimum_lift_m", 0.04))
        self._grasp_height: float | None = None
        self._initialized = False

    def connect(self) -> None:
        if self.viewer_requested and self.viewer is None:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
        self._set_joints(self.home_joints, self.opening)
        self._initialized = True

    def self_check(self) -> tuple[bool, str]:
        ok = self._initialized and len(self.qpos) == 6 and len(self.actuators) == 6
        return ok, "" if ok else "MuJoCo 模型接口不完整"

    def current_joints(self) -> tuple[float, ...]:
        return tuple(float(self.data.qpos[index]) for index in self.qpos)

    def _set_joints(self, joints: Sequence[float], gripper: float) -> None:
        for address, actuator, value in zip(self.qpos, self.actuators, joints):
            self.data.qpos[address] = self.data.ctrl[actuator] = float(value)
        self.data.ctrl[self.gripper] = float(gripper)
        self.mujoco.mj_forward(self.model, self.data)

    def home(self) -> None:
        self._set_joints(self.home_joints, self.opening)

    def search_pose(self) -> None:
        self._set_joints(self.search_joints, self.opening)

    def execute(self, command: MotionCommand) -> ExecutionResult:
        start = np.asarray(self.current_joints())
        goal = np.asarray(command.joints)
        timestep = float(self.model.opt.timestep)
        steps = max(2, int(command.duration_s / timestep))
        for index in range(1, steps + 1):
            ratio = 0.5 - 0.5 * np.cos(np.pi * index / steps)
            joints = start + ratio * (goal - start)
            for actuator, value in zip(self.actuators, joints):
                self.data.ctrl[actuator] = float(value)
            self.data.ctrl[self.gripper] = float(command.gripper)
            self._step(timestep)
        if command.name == "open":
            for _ in range(self.settle_steps):
                self._step(timestep)
        error = float(np.max(np.abs(np.asarray(self.current_joints()) - goal)))
        reason = f"最大关节误差 {error:.3f} rad"
        if error >= 0.08:
            return ExecutionResult(False, reason=reason)
        if command.name in {"close", "lift", "transit", "place"} and not self.has_object():
            return ExecutionResult(False, reason=f"{command.name} 未检测到目标与左右夹爪的双侧接触")
        if command.name == "close":
            self._grasp_height = self.target_position()[2]
        elif command.name == "lift":
            if self._grasp_height is None or self.target_position()[2] - self._grasp_height < self.minimum_lift:
                return ExecutionResult(False, reason="lift 目标未随夹爪抬升")
        elif command.name == "open":
            if self.has_object() or not self.target_in_storage():
                return ExecutionResult(False, reason="open 后目标未脱离夹爪并进入收纳区")
        return ExecutionResult(True, reason=reason)

    def _step(self, timestep: float) -> None:
        self.mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        if self.realtime:
            time.sleep(timestep)

    def has_object(self) -> bool:
        touched: set[int] = set()
        for contact in self.data.contact[: self.data.ncon]:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.target_geom:
                touched.add(geom2)
            elif geom2 == self.target_geom:
                touched.add(geom1)
        return bool(touched & self.left_geoms) and bool(touched & self.right_geoms)

    def target_position(self) -> np.ndarray:
        return self.data.xpos[self.target_body].copy()

    def target_in_storage(self) -> bool:
        rotation = self.data.site_xmat[self.storage_region].reshape(3, 3)
        offset = rotation.T @ (self.target_position() - self.data.site_xpos[self.storage_region])
        return bool(np.all(np.abs(offset) <= self.model.site_size[self.storage_region]))

    def recover(self, plan: PickPlan | None) -> None:
        if plan is not None:
            try:
                lift = plan.command("lift")
                if float(np.max(np.abs(np.asarray(lift.joints) - np.asarray(self.current_joints())))) <= 0.6:
                    self.execute(lift)
            except (LookupError, RuntimeError):
                pass

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None
