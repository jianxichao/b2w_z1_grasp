"""Common command guards used before simulation or hardware execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np

from .kinematics import Arx5Kinematics
from .types import MotionCommand, Pose, Target, Vector6


class SafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyLimits:
    joint_lower: Vector6 = (-1.581, 0.0, -1.57, -1.30, -1.57, -2.10)
    joint_upper: Vector6 = (1.581, 2.101, 1.57, 1.30, 1.57, 2.10)
    minimum_radius_m: float = 0.10
    maximum_radius_m: float = 0.62
    minimum_z_m: float = 0.08
    maximum_z_m: float = 0.62
    minimum_x_m: float = 0.10
    maximum_abs_y_m: float = 0.45
    maximum_abs_heading_rad: float = math.pi / 2.0
    trajectory_step_rad: float = 0.02
    target_max_age_s: float = 0.50
    minimum_confidence: float = 0.25
    gripper_min: float = 0.0
    gripper_max: float = 0.088
    frame_id: str = "arm_base"

    def __post_init__(self) -> None:
        if len(self.joint_lower) != 6 or len(self.joint_upper) != 6:
            raise ValueError("joint limits must contain six values")
        if self.trajectory_step_rad <= 0.0 or self.target_max_age_s <= 0.0:
            raise ValueError("safety steps and timeouts must be positive")


class SafetyGuard:
    def __init__(self, kinematics: Arx5Kinematics, limits: SafetyLimits | None = None) -> None:
        self.kinematics = kinematics
        self.limits = limits or SafetyLimits()

    def validate_joints(self, joints: Sequence[float], label: str = "joints") -> None:
        values = np.asarray(joints, dtype=float)
        lower, upper = np.asarray(self.limits.joint_lower), np.asarray(self.limits.joint_upper)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise SafetyError(f"{label} must contain six finite values")
        if np.any(values < lower) or np.any(values > upper):
            bad = (values < lower) | (values > upper)
            index = int(np.flatnonzero(bad)[0])
            parts = []
            for joint_index in np.flatnonzero(bad):
                value = float(values[joint_index])
                lo, hi = float(lower[joint_index]), float(upper[joint_index])
                parts.append(f"joint{joint_index + 1}={value:+.4f} not in [{lo:+.4f}, {hi:+.4f}]")
            detail = "; ".join(parts)
            raise SafetyError(f"{label}: {detail}")

    def validate_pose(self, pose: Pose, label: str = "pose") -> None:
        if pose.frame_id != self.limits.frame_id:
            raise SafetyError(f"{label} frame must be {self.limits.frame_id}")
        x, y, z = pose.position
        radius, heading = math.hypot(x, y), abs(math.atan2(y, x))
        limits = self.limits
        if x < limits.minimum_x_m or abs(y) > limits.maximum_abs_y_m:
            raise SafetyError(f"{label} is outside the front workspace")
        if heading > limits.maximum_abs_heading_rad:
            raise SafetyError(f"{label} heading is outside the workspace")
        if not limits.minimum_radius_m <= radius <= limits.maximum_radius_m:
            raise SafetyError(f"{label} radius is outside the workspace")
        if not limits.minimum_z_m <= z <= limits.maximum_z_m:
            raise SafetyError(f"{label} height is outside the workspace")

    def validate_target(self, target: Target, now_s: float | None = None) -> None:
        if target.confidence < self.limits.minimum_confidence:
            raise SafetyError("target confidence is too low")
        if target.stamp_s is not None:
            age = (time.time() if now_s is None else float(now_s)) - target.stamp_s
            if age < -0.05 or age > self.limits.target_max_age_s:
                raise SafetyError("target is stale or has an invalid timestamp")
        self.validate_pose(target.pose, "target")

    def validate_command(self, command: MotionCommand) -> None:
        if command.duration_s <= 0.0:
            raise SafetyError("command duration must be positive")
        if not self.limits.gripper_min <= command.gripper <= self.limits.gripper_max:
            raise SafetyError("gripper command is outside its limit")
        self.validate_joints(command.joints, command.name)
        if command.target_pose is not None:
            self.validate_pose(command.target_pose, command.name)
        tcp = self.kinematics.forward_position(command.joints)
        rotation = self.kinematics.forward_matrix(command.joints)[:3, :3]
        self.validate_pose(Pose(tuple(tcp), tuple(map(tuple, rotation))), f"{command.name} FK")  # type: ignore[arg-type]

    def _validate_segment(self, start_joints: Sequence[float], command: MotionCommand) -> None:
        self.validate_joints(start_joints, "path start")
        self.validate_command(command)
        start, target = np.asarray(start_joints, dtype=float), np.asarray(command.joints)
        count = max(1, math.ceil(float(np.max(np.abs(target - start))) / self.limits.trajectory_step_rad))
        for index in range(1, count + 1):
            joints = start + (target - start) * (index / count)
            self.validate_joints(joints, f"{command.name} path")
            tcp = self.kinematics.forward_position(joints)
            rotation = self.kinematics.forward_matrix(joints)[:3, :3]
            self.validate_pose(Pose(tuple(tcp), tuple(map(tuple, rotation))), f"{command.name} path")  # type: ignore[arg-type]

    def validate_path(
        self,
        start_or_commands: Sequence[float] | Sequence[MotionCommand],
        command: MotionCommand | None = None,
    ) -> None:
        """Validate one segment or every consecutive segment in a plan."""

        if command is not None:
            self._validate_segment(start_or_commands, command)  # type: ignore[arg-type]
            return
        commands = tuple(start_or_commands)
        if not commands or not all(isinstance(item, MotionCommand) for item in commands):
            raise SafetyError("plan path must contain motion commands")
        self.validate_command(commands[0])  # type: ignore[arg-type]
        for previous, following in zip(commands, commands[1:]):
            self._validate_segment(previous.joints, following)  # type: ignore[union-attr]
