"""ARX X5 SDK 的精简、安全关节控制后端。"""

from __future__ import annotations

import importlib
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..manipulation.types import ExecutionResult, HardwareFault, MotionCommand, PickPlan
from ..manipulation.kinematics import Arx5Kinematics, rpy_matrix


class Arx5Backend:
    def __init__(
        self,
        root: Path,
        config: Mapping[str, Any],
        dry_run: bool = False,
        kinematics: Arx5Kinematics | None = None,
    ) -> None:
        self.root, self.config, self.dry_run = root, config, dry_run
        self.interface = str(config.get("can_interface", "can1"))
        self.home_joints = tuple(float(v) for v in config["home_joints"])
        self.search_joints = tuple(float(v) for v in config.get("search_joints", self.home_joints))
        self.opening = float(config.get("gripper_open", 0.08))
        self.rate = float(config.get("command_rate_hz", 50.0))
        self.max_speed = float(config.get("max_joint_speed_rad_s", 0.35))
        self.max_tracking_error = float(config.get("max_tracking_error_rad", 0.20))
        self.correction_attempts = int(config.get("correction_attempts", 2))
        if self.max_tracking_error <= 0.0:
            raise ValueError("跟踪误差上限必须为正数")
        if self.correction_attempts < 0:
            raise ValueError("修正次数不能为负数")
        self.feedback_timeout = float(config.get("feedback_timeout_s", 0.5))
        self.contact_torque = float(config.get("gripper_contact_torque_nm", 0.15))
        self.contact_position = float(config.get("gripper_contact_position_m", 0.003))
        self.gripper_tolerance = float(config.get("gripper_tolerance_m", 0.005))
        self.joint_tolerance = float(config.get("joint_tolerance_rad", 0.08))
        self.joint_lower = np.asarray(config.get("joint_lower_rad", [-1.581, 0.0, -1.57, -1.30, -1.57, -2.10]))
        self.joint_upper = np.asarray(config.get("joint_upper_rad", [1.581, 2.101, 1.57, 1.30, 1.57, 2.10]))
        self.arx5: Any | None = None
        self.controller: Any | None = None
        self._joints = self.home_joints
        self._joint_velocity = np.zeros(6, dtype=float)
        self._gripper_torque = 0.0
        self._gripper_position = self.opening
        self._joint_torque = np.zeros(6, dtype=float)
        # 可选的同步观测回调；在轨迹线程取得同一份反馈后调用，避免并发访问 SDK。
        self.motion_feedback_callback: Any | None = None
        self._closed_commanded = False
        self._position_enabled = False
        self._last_feedback_stamp: float | None = None
        self._last_feedback_time: float | None = None
        self.kinematics = kinematics or Arx5Kinematics()

    def _load_sdk(self) -> Any:
        sdk = self.root / "third_party" / "arx5_sdk"
        python_dir = sdk / "python"
        extension = next(python_dir.glob("arx5_interface*.so"), None)
        # The laptop keeps a source-built extension in the repository.  The
        # aarch64 dock instead installs the official arx5-interface wheel,
        # because an x86_64 extension can never be reused there.
        use_bundled = extension is not None
        if use_bundled and str(python_dir) not in sys.path:
            sys.path.insert(0, str(python_dir))
        try:
            module = importlib.import_module("arx5_interface")
        except ImportError as exc:
            raise RuntimeError(
                "ARX5 SDK 不可用：本机需构建工程 SDK，拓展坞需安装官方 "
                "arx5-interface ARM64 wheel"
            ) from exc
        loaded = Path(str(getattr(module, "__file__", ""))).resolve()
        if use_bundled and loaded.parent != python_dir.resolve():
            raise RuntimeError(f"加载了项目外的 ARX5 SDK：{loaded}")
        required = (
            "RobotConfigFactory",
            "ControllerConfigFactory",
            "Arx5JointController",
            "JointState",
            "Gain",
            "LogLevel",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(
                f"ARX5 SDK 接口不完整 ({loaded})：缺少 {', '.join(missing)}"
            )
        return module

    def connect(self) -> None:
        if self.dry_run:
            return
        try:
            socket.if_nametoindex(self.interface)
        except OSError as exc:
            raise HardwareFault(f"CAN 接口不存在：{self.interface}") from exc
        self.arx5 = self._load_sdk()
        robot = self.arx5.RobotConfigFactory.get_instance().get_config("X5")
        robot.urdf_path = str(self.root / "third_party/arx5_sdk/models/X5.urdf")
        gripper_open_readout = self.config.get("gripper_open_readout")
        if gripper_open_readout is not None:
            robot.gripper_open_readout = float(gripper_open_readout)
        if self.opening > float(robot.gripper_width):
            raise RuntimeError("配置夹爪开度超过 ARX5 SDK 上限")
        robot.joint_pos_min = np.maximum(np.asarray(robot.joint_pos_min), self.joint_lower)
        robot.joint_pos_max = np.minimum(np.asarray(robot.joint_pos_max), self.joint_upper)
        robot.joint_vel_max = np.minimum(np.asarray(robot.joint_vel_max), self.max_speed)
        cfg = self.arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", 6)
        cfg.background_send_recv = True
        cfg.gravity_compensation = True
        cfg.shutdown_to_passive = True
        self.controller = self.arx5.Arx5JointController(robot, cfg, self.interface)
        self.controller.set_log_level(self.arx5.LogLevel.WARNING)
        self._feedback()

    def _enable_position_control(self) -> None:
        """从实测位置开始平滑恢复 SDK 默认位置增益。"""
        if self.dry_run or self._position_enabled:
            return
        assert self.arx5 is not None and self.controller is not None
        current = self.controller.get_joint_state()
        command = self.arx5.JointState(6)
        command.pos()[:] = current.pos()
        command.gripper_pos = float(current.gripper_pos)
        self.controller.set_joint_cmd(command)
        gain = self.controller.get_controller_config()
        target = self.arx5.Gain(
            gain.default_kp,
            gain.default_kd,
            gain.default_gripper_kp,
            gain.default_gripper_kd,
        )
        for ratio in np.linspace(0.05, 1.0, 20):
            self.controller.set_gain(
                self.arx5.Gain(
                    target.kp() * ratio,
                    target.kd(),
                    target.gripper_kp * ratio,
                    target.gripper_kd,
                )
            )
            time.sleep(0.025)
        self._position_enabled = True

    def _feedback(self) -> tuple[float, ...]:
        if self.dry_run:
            return self._joints
        if self.controller is None:
            raise HardwareFault("机械臂尚未连接")
        state = self.controller.get_joint_state()
        joints = tuple(float(v) for v in state.pos())
        if len(joints) != 6 or not np.all(np.isfinite(joints)):
            raise HardwareFault("机械臂反馈无效")
        values = np.asarray(joints)
        # Feedback may sit a few encoder counts outside a software command
        # bound at startup. Accept only the configured convergence tolerance;
        # command validation remains clamped to the strict limits below.
        if np.any(values < self.joint_lower - self.joint_tolerance) or np.any(
            values > self.joint_upper + self.joint_tolerance
        ):
            raise HardwareFault(
                "机械臂反馈超过配置关节限位：feedback=%s lower=%s upper=%s"
                % (
                    np.round(values, 4).tolist(),
                    np.round(self.joint_lower, 4).tolist(),
                    np.round(self.joint_upper, 4).tolist(),
                )
            )
        self._joints = joints
        torque_reader = getattr(state, "torque", None)
        if callable(torque_reader):
            torques = np.asarray(torque_reader(), dtype=float)
            if torques.shape != (6,) or not np.all(np.isfinite(torques)):
                raise HardwareFault("机械臂关节扭矩反馈无效")
            self._joint_torque = torques.copy()
        vel_reader = getattr(state, "vel", None)
        if callable(vel_reader):
            velocities = np.asarray(vel_reader(), dtype=float)
            if velocities.shape == (6,) and np.all(np.isfinite(velocities)):
                self._joint_velocity = velocities.copy()
        self._gripper_torque = float(state.gripper_torque)
        self._gripper_position = float(state.gripper_pos)
        stamp = float(state.timestamp)
        if not np.all(np.isfinite((self._gripper_torque, self._gripper_position, stamp))):
            raise HardwareFault("夹爪或时间戳反馈无效")
        now = time.monotonic()
        feedback_age = float(self.controller.get_timestamp()) - stamp
        if feedback_age < -0.05 or feedback_age > self.feedback_timeout:
            raise HardwareFault("机械臂反馈过期")
        if self._last_feedback_stamp is None or stamp > self._last_feedback_stamp:
            self._last_feedback_stamp, self._last_feedback_time = stamp, now
        elif self._last_feedback_time is not None and now - self._last_feedback_time > self.feedback_timeout:
            raise HardwareFault("机械臂反馈时间戳停止更新")
        return joints

    def _move_reason(
        self,
        goal: np.ndarray,
        *,
        joint_error: float,
        gripper: float,
        gripper_error: float,
        virtual: np.ndarray | None = None,
    ) -> str:
        measured = np.asarray(self.current_joints(), dtype=float)
        parts = [
            f"最大关节误差 {joint_error:.3f} rad（容差 {self.joint_tolerance:.3f}）",
            f"夹爪误差 {gripper_error:.3f} m（容差 {self.gripper_tolerance:.3f}）",
        ]
        if virtual is not None and virtual.shape == (6,):
            tracking = np.abs(measured - virtual)
            worst = int(np.argmax(tracking))
            parts.append(
                f"实测/虚拟最大偏差 joint{worst + 1}={tracking[worst]:.3f} rad "
                f"(meas={measured[worst]:+.3f}, virt={virtual[worst]:+.3f})"
            )
        per_joint = []
        for index in range(6):
            err = float(abs(measured[index] - goal[index]))
            if err >= self.joint_tolerance * 0.5:
                per_joint.append(
                    f"j{index + 1} meas={measured[index]:+.3f} goal={goal[index]:+.3f} err={err:.3f}"
                )
        if per_joint:
            parts.append("关节明细: " + " | ".join(per_joint))
        return "；".join(parts)

    def self_check(self) -> tuple[bool, str]:
        try:
            self._feedback()
            first_stamp = self._last_feedback_stamp
            time.sleep(0.02)
            self._feedback()
            ok = self.dry_run or (
                first_stamp is not None
                and self._last_feedback_stamp is not None
                and self._last_feedback_stamp > first_stamp
            )
            return ok, "" if ok else "机械臂反馈时间戳未更新"
        except RuntimeError as exc:
            return False, str(exc)

    def current_joints(self) -> tuple[float, ...]:
        return self._feedback()

    def latest_feedback_joints(self) -> tuple[float, ...]:
        """返回轨迹线程最近一次实测反馈，不再次访问 SDK。"""
        return tuple(self._joints)

    def latest_feedback_velocity(self) -> tuple[float, ...]:
        """返回轨迹线程最近一次实测关节速度（rad/s）。"""
        if self.dry_run:
            return tuple(float(value) for value in self._joint_velocity)
        self._feedback()
        return tuple(float(value) for value in self._joint_velocity)

    def current_joint_torques(self) -> tuple[float, ...]:
        """返回最近一次 SDK 反馈的六个关节扭矩（Nm）。"""
        if self.dry_run:
            return tuple(float(value) for value in self._joint_torque)
        self._feedback()
        return tuple(float(value) for value in self._joint_torque)

    def base_T_eef(self) -> np.ndarray:
        """真机使用 SDK 末端反馈，与手眼标定时的位姿来源一致。"""
        if self.dry_run:
            return self.kinematics.forward_matrix(self.current_joints())
        if self.controller is None:
            raise HardwareFault("机械臂尚未连接")
        self._feedback()
        state = self.controller.get_eef_state()
        pose = np.asarray(state.pose_6d(), dtype=float)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise HardwareFault("机械臂末端反馈无效")
        transform = np.eye(4)
        transform[:3, :3] = rpy_matrix(pose[3:])
        transform[:3, 3] = pose[:3]
        return transform

    def _reset_virtual_start(self, feedback: np.ndarray, gripper: float) -> None:
        """Reset the SDK interpolator to the measured state before replanning."""
        command = self.arx5.JointState(6)
        command.pos()[:] = feedback
        command.gripper_pos = float(gripper)
        command.timestamp = float(self.controller.get_timestamp())
        self.controller.set_joint_cmd(command)

    def _virtual_joints(self) -> np.ndarray:
        """Copy SDK command data while its owning C++ state is still alive."""
        state = self.controller.get_joint_cmd()
        joints = np.array(state.pos(), dtype=float, copy=True)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise HardwareFault("机械臂虚拟轨迹无效")
        return joints

    def _move(self, joints: Sequence[float], gripper: float, duration: float) -> ExecutionResult:
        self._enable_position_control()
        goal = np.asarray(joints, dtype=float)
        if goal.shape != (6,) or np.any(goal < self.joint_lower) or np.any(goal > self.joint_upper):
            raise RuntimeError("目标关节超过配置限位")
        if not 0.0 <= gripper <= self.opening:
            raise RuntimeError("夹爪命令超过配置限位")
        last_virtual = None
        if self.dry_run:
            self._joints = tuple(float(v) for v in goal)
            self._gripper_position = float(gripper)
            error = 0.0
        else:
            for correction in range(self.correction_attempts + 1):
                start = np.asarray(self.current_joints())
                required = float(np.max(np.abs(goal - start))) / self.max_speed
                command_duration = float(duration) if correction == 0 else 0.5
                move_duration = max(command_duration, required, 0.02)
                steps = max(2, int(move_duration * self.rate))
                controller_time = float(self.controller.get_timestamp())
                trajectory = []
                for index in range(1, steps + 1):
                    ratio = 0.5 - 0.5 * np.cos(np.pi * index / steps)
                    command = self.arx5.JointState(6)
                    command.pos()[:] = start + ratio * (goal - start)
                    command.gripper_pos = float(gripper)
                    command.timestamp = controller_time + move_duration * index / steps
                    trajectory.append(command)
                self.controller.set_joint_traj(trajectory)
                deadline = time.monotonic() + move_duration
                lost_sync = False
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    time.sleep(min(1.0 / self.rate, max(0.0, deadline - now)))
                    feedback = np.asarray(self._feedback())
                    virtual = self._virtual_joints()
                    last_virtual = virtual.copy()
                    tracking = np.abs(feedback - virtual)
                    worst_joint = int(np.argmax(tracking))
                    tracking_error = float(tracking[worst_joint])
                    callback = self.motion_feedback_callback
                    if callback is not None:
                        callback(feedback.copy(), self._joint_torque.copy(), goal.copy(), virtual.copy())
                    if tracking_error > self.max_tracking_error:
                        self._reset_virtual_start(feedback, self._gripper_position)
                        lost_sync = True
                        break
                if not lost_sync:
                    break
                if correction >= self.correction_attempts:
                    raise HardwareFault(
                        f"实测与虚拟轨迹多次校正失败："
                        f"joint{worst_joint + 1} 误差 {tracking_error:.3f} rad，"
                        f"实测 {feedback[worst_joint]:.3f}，虚拟 {virtual[worst_joint]:.3f}"
                    )

            settle_deadline = time.monotonic() + max(0.5, move_duration * 0.35)
            while time.monotonic() < settle_deadline:
                feedback = np.asarray(self._feedback())
                virtual = self._virtual_joints()
                last_virtual = virtual.copy()
                error = float(np.max(np.abs(feedback - goal)))
                gripper_error = abs(self._gripper_position - gripper)
                if error <= self.joint_tolerance and (
                    gripper < self.opening * 0.5 or gripper_error <= self.gripper_tolerance
                ):
                    break
                time.sleep(1.0 / self.rate)
            error = float(np.max(np.abs(np.asarray(self.current_joints()) - goal)))

        gripper_error = abs(self._gripper_position - gripper)
        joint_ok = error <= self.joint_tolerance
        gripper_ok = gripper < self.opening * 0.5 or gripper_error <= self.gripper_tolerance
        reached = joint_ok and gripper_ok
        virtual = last_virtual
        return ExecutionResult(
            reached,
            reason=self._move_reason(
                goal,
                joint_error=error,
                gripper=gripper,
                gripper_error=gripper_error,
                virtual=virtual,
            ),
        )

    def home(self) -> ExecutionResult:
        return self._move(self.home_joints, self.opening, 2.0)

    def search_pose(self) -> ExecutionResult:
        return self._move(self.search_joints, self.opening, 2.0)

    def execute(self, command: MotionCommand) -> ExecutionResult:
        result = self._move(command.joints, command.gripper, command.duration_s)
        self._closed_commanded = command.gripper < self.opening * 0.5
        if self.dry_run:
            self._gripper_position = float(command.gripper)
        else:
            # 重新读取夹爪位置/力矩，供后续抓取与放置验证。
            self._feedback()
        return result

    def has_object(self) -> bool:
        if self.dry_run:
            return self._gripper_position < self.opening * 0.5
        commanded_closed = self._closed_commanded
        self._feedback()
        return commanded_closed and (
            abs(self._gripper_torque) >= self.contact_torque
            # 闭合命令为 0；若实测开度仍大于阈值，夹爪被物体阻挡。
            or self._gripper_position >= self.contact_position
        )

    def recover(self, plan: PickPlan | None) -> None:
        current = np.asarray(self.current_joints())
        home = np.asarray(self.home_joints)
        # 目标/轨迹未知时继续保持当前夹持状态，避免误放物体；只有在已知
        # 抓取计划中确认没有持物，才在安全撤退前张开夹爪。
        gripper = 0.0 if self._closed_commanded else self.opening
        # 先沿已规划的抬升点退出；目标未知时保持当前位置，避免盲目扫回零位。
        if plan is not None:
            try:
                if self._closed_commanded and not self.has_object():
                    gripper = self.opening
                lift = plan.command("lift")
                if float(np.max(np.abs(np.asarray(lift.joints) - current))) <= 0.6:
                    self._move(lift.joints, gripper, 2.0)
                    current = np.asarray(lift.joints)
                    # 已经到达本次规划的安全抬升点时，继续沿原先通过安全
                    # 检查的中转轨迹退出，避免因与零位相差过大而悬停。
                    transit = plan.command("transit")
                    self._move(transit.joints, gripper, max(2.0, transit.duration_s))
                    current = np.asarray(transit.joints)
                    retreat = plan.command("retreat")
                    self._move(retreat.joints, gripper, max(2.0, retreat.duration_s))
                    return
            except HardwareFault:
                raise
            except (LookupError, RuntimeError):
                pass
        if float(np.max(np.abs(home - current))) <= 0.8:
            self._move(self.home_joints, gripper, 2.0)

    def hold_enabled(self, *, block: bool = True) -> None:
        """Keep the current pose enabled, optionally without blocking the caller.

        The historic default (``block=True``) remains an operator hold loop
        that exits on Ctrl+C.  Continuous task programs can use
        ``block=False`` to retain the enabled pose and immediately continue to
        their own command prompt.
        """
        if self.dry_run:
            return
        if not block:
            self._feedback()
            return
        print("机械臂保持使能；按 Ctrl+C 结束并进入阻尼。")
        try:
            while True:
                self._feedback()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n收到停止指令，机械臂进入阻尼。")

    def close(self) -> None:
        controller = self.controller
        try:
            if controller is not None:
                controller.set_to_damping()
        finally:
            self.controller = self.arx5 = None
            self._position_enabled = self._closed_commanded = False
