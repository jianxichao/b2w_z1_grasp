"""Convert one target into a short, named pick-and-place plan."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from .kinematics import Arx5Kinematics, rpy_matrix
from .types import MotionCommand, PickPlan, Pose, Target, Vector6


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class PosePolicy:
    """RL2 policy: face the target and progressively pitch down near it."""

    base_roll_rad: float = 0.0
    base_pitch_rad: float = 0.375
    dive_start_z_m: float = 0.25
    minimum_z_m: float = 0.08
    dive_gain_rad: float = 1.0
    pitch_min_rad: float = 0.20
    pitch_max_rad: float = 1.55

    def __post_init__(self) -> None:
        if self.minimum_z_m >= self.dive_start_z_m:
            raise ValueError("minimum_z_m must be below dive_start_z_m")
        if self.pitch_min_rad > self.pitch_max_rad:
            raise ValueError("pitch limits are reversed")

    def rotation(self, position: Sequence[float]) -> np.ndarray:
        x, y, z = (float(value) for value in position)
        heading = math.atan2(y, x)
        span = max(1e-9, self.dive_start_z_m - self.minimum_z_m)
        ratio = float(np.clip((self.dive_start_z_m - z) / span, 0.0, 1.0))
        pitch = float(np.clip(
            self.base_pitch_rad + self.dive_gain_rad * ratio,
            self.pitch_min_rad,
            self.pitch_max_rad,
        ))
        return rpy_matrix((self.base_roll_rad, pitch, heading))


def _durations() -> dict[str, float]:
    return {
        "home": 2.0,
        "search": 2.0,
        "pregrasp": 2.0,
        "grasp": 1.5,
        "close": 0.65,
        "lift": 1.5,
        "transit": 2.0,
        "place": 2.5,
        "open": 0.65,
        "retreat": 2.0,
    }


@dataclass(frozen=True)
class MotionSettings:
    home_joints: Vector6 = (0.0, 0.645, 1.040, -0.770, 0.0, 0.0)
    search_joints: Vector6 = (0.0, 0.645, 1.040, -0.770, 0.0, 0.0)
    open_gripper: float = 0.088
    closed_gripper: float = 0.0
    pregrasp_offset_m: float = 0.08
    lift_offset_m: float = 0.08
    transit_height_m: float = 0.40
    place_position: tuple[float, float, float] = (0.25, 0.30, 0.305)
    place_clearance_m: float = 0.05
    durations_s: Mapping[str, float] = field(default_factory=_durations)
    pose_policy: PosePolicy = field(default_factory=PosePolicy)
    frame_id: str = "arm_base"

    def __post_init__(self) -> None:
        if len(self.home_joints) != 6 or len(self.search_joints) != 6:
            raise ValueError("home and search poses must contain six joints")
        if min(self.pregrasp_offset_m, self.lift_offset_m, self.place_clearance_m) <= 0.0:
            raise ValueError("grasp offsets must be positive")


class GraspPlanner:
    def __init__(self, kinematics: Arx5Kinematics, settings: MotionSettings | None = None) -> None:
        self.kinematics = kinematics
        self.settings = settings or MotionSettings()

    def _pose(self, position: Sequence[float]) -> Pose:
        rotation = self.settings.pose_policy.rotation(position)
        return Pose(
            tuple(float(value) for value in position),  # type: ignore[arg-type]
            tuple(tuple(float(value) for value in row) for row in rotation),  # type: ignore[arg-type]
            self.settings.frame_id,
        )

    def _seeds(self, position: Sequence[float], current: Sequence[float]) -> tuple[Vector6, ...]:
        heading = float(np.clip(math.atan2(position[1], position[0]), self.kinematics.lower[0], self.kinematics.upper[0]))
        home = self.settings.home_joints
        candidates = (
            current,
            self.settings.search_joints,
            home,
            (heading, *home[1:]),
            (heading, 0.35, 0.80, -0.55, 0.0, 0.0),
            (heading, 0.75, 1.05, -0.75, 0.0, 0.0),
            tuple((self.kinematics.lower + self.kinematics.upper) / 2.0),
        )
        unique: list[Vector6] = []
        for candidate in candidates:
            values = tuple(float(value) for value in candidate)
            if values not in unique:
                unique.append(values)  # type: ignore[arg-type]
        return tuple(unique)

    def _solve(self, name: str, pose: Pose, current: Vector6, gripper: float) -> MotionCommand:
        result = self.kinematics.solve(pose.position, pose.rotation, self._seeds(pose.position, current))
        if result is None:
            xyz = pose.position
            radius = math.hypot(float(xyz[0]), float(xyz[1]))
            current_tcp = self.kinematics.forward_position(current)
            search_tcp = self.kinematics.forward_position(self.settings.search_joints)
            limit_detail = getattr(self.kinematics, "last_ik_diagnostic", "") or ""
            limit_suffix = f" 触限/诊断: {limit_detail}" if limit_detail else ""
            raise PlanningError(
                f"{name} IK has no solution for ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}); "
                f"horizontal radius={radius:.3f} m. "
                f"Current TCP≈({current_tcp[0]:.3f}, {current_tcp[1]:.3f}, {current_tcp[2]:.3f}), "
                f"search TCP≈({search_tcp[0]:.3f}, {search_tcp[1]:.3f}, {search_tcp[2]:.3f}). "
                "At table height this search pose typically reaches x≈0.20–0.35 m in arm_base; "
                "targets beyond ~0.40 m usually mean the stone is too far, hand-eye calibration is off, "
                f"or search_joints should be an inverted/teach pose for downward grasp.{limit_suffix}"
            )
        try:
            duration = float(self.settings.durations_s[name])
        except KeyError as exc:
            raise PlanningError(f"missing duration for {name}") from exc
        return MotionCommand(name, result.joints, gripper, duration, pose)

    def _direct(self, name: str, joints: Vector6, gripper: float) -> MotionCommand:
        try:
            duration = float(self.settings.durations_s[name])
        except KeyError as exc:
            raise PlanningError(f"missing duration for {name}") from exc
        return MotionCommand(name, joints, gripper, duration)

    def plan(self, target: Target, current_joints: Sequence[float]) -> PickPlan:
        if target.pose.frame_id != self.settings.frame_id:
            raise PlanningError(f"target frame must be {self.settings.frame_id}")
        current = tuple(float(value) for value in current_joints)
        if len(current) != 6:
            raise PlanningError("current_joints must contain six values")
        xyz = np.asarray(target.pose.position, dtype=float)
        pre_z = min(
            float(xyz[2] + self.settings.pregrasp_offset_m),
            self.settings.transit_height_m,
        )
        poses = {
            "pregrasp": self._pose((xyz[0], xyz[1], pre_z)),
            "grasp": self._pose(xyz),
            "lift": self._pose(xyz + (0.0, 0.0, self.settings.lift_offset_m)),
            "transit": self._pose((
                self.settings.place_position[0],
                self.settings.place_position[1],
                min(
                    self.settings.transit_height_m,
                    self.settings.place_position[2] + self.settings.place_clearance_m,
                ),
            )),
            "place": self._pose(self.settings.place_position),
        }
        commands: list[MotionCommand] = [
            self._direct("home", self.settings.home_joints, self.settings.open_gripper),
            self._direct("search", self.settings.search_joints, self.settings.open_gripper),
        ]
        previous = current
        for name in ("pregrasp", "grasp"):
            command = self._solve(name, poses[name], previous, self.settings.open_gripper)
            commands.append(command)
            previous = command.joints
        commands.append(self._direct("close", previous, self.settings.closed_gripper))
        lift = self._solve("lift", poses["lift"], previous, self.settings.closed_gripper)
        transit = self._solve("transit", poses["transit"], lift.joints, self.settings.closed_gripper)
        place = self._solve("place", poses["place"], transit.joints, self.settings.closed_gripper)
        commands.extend((lift, transit, place, self._direct("open", place.joints, self.settings.open_gripper)))
        commands.append(self._direct("retreat", self.settings.home_joints, self.settings.open_gripper))
        return PickPlan(target, tuple(commands))

    def plan_pick_from_current(self, target: Target, current_joints: Sequence[float]) -> PickPlan:
        """从当前关节直接规划抓取四步：pregrasp → grasp → close → lift（不含 home/search/place）。"""
        if target.pose.frame_id != self.settings.frame_id:
            raise PlanningError(f"target frame must be {self.settings.frame_id}")
        current = tuple(float(value) for value in current_joints)
        if len(current) != 6:
            raise PlanningError("current_joints must contain six values")
        xyz = np.asarray(target.pose.position, dtype=float)
        pre_z = min(
            float(xyz[2] + self.settings.pregrasp_offset_m),
            self.settings.transit_height_m,
        )
        poses = {
            "pregrasp": self._pose((xyz[0], xyz[1], pre_z)),
            "grasp": self._pose(xyz),
            "lift": self._pose(xyz + (0.0, 0.0, self.settings.lift_offset_m)),
        }
        commands: list[MotionCommand] = []
        previous = current
        for name in ("pregrasp", "grasp"):
            command = self._solve(name, poses[name], previous, self.settings.open_gripper)
            commands.append(command)
            previous = command.joints
        commands.append(self._direct("close", previous, self.settings.closed_gripper))
        lift = self._solve("lift", poses["lift"], previous, self.settings.closed_gripper)
        commands.append(lift)
        return PickPlan(target, tuple(commands))
