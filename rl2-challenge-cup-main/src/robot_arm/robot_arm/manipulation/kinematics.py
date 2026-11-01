"""ARX X5 kinematics: SDK ``Arx5Solver`` (KDL) for FK/IK, numerical fallback for CI/sim."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .types import Matrix3, Vector3, Vector6


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def matrix_to_rpy(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Rotation matrix -> roll/pitch/yaw, same convention as SDK ``pose_6d``."""
    matrix = np.asarray(rotation, dtype=float)
    pitch = math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return np.array((roll, pitch, yaw), dtype=float)


def pose6_from_target(position: Sequence[float], rotation: Sequence[Sequence[float]]) -> np.ndarray:
    xyz = np.asarray(position, dtype=float)
    return np.concatenate((xyz, matrix_to_rpy(rotation)))


def _load_arx5_sdk(root: Path) -> Any:
    sdk_python = root / "third_party" / "arx5_sdk" / "python"
    extension = next(sdk_python.glob("arx5_interface*.so"), None)
    if extension is None:
        raise RuntimeError("ARX5 SDK 未构建，请先 cmake --build third_party/arx5_sdk/build")
    if str(sdk_python) not in sys.path:
        sys.path.insert(0, str(sdk_python))
    module = importlib.import_module("arx5_interface")
    loaded = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded.parent != sdk_python.resolve():
        raise RuntimeError(f"加载了项目外的 ARX5 SDK：{loaded}")
    return module


def _translation(xyz: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _rotation(rpy: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    return transform


def _axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    value = np.asarray(axis, dtype=float)
    value /= np.linalg.norm(value)
    x, y, z = value
    c, s, v = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    transform = np.eye(4)
    transform[:3, :3] = (
        (x * x * v + c, x * y * v - z * s, x * z * v + y * s),
        (y * x * v + z * s, y * y * v + c, y * z * v - x * s),
        (z * x * v - y * s, z * y * v + x * s, z * z * v + c),
    )
    return transform


@dataclass(frozen=True)
class JointSpec:
    name: str
    axis: Vector3
    origin_xyz: Vector3
    origin_rpy: Vector3
    lower: float
    upper: float


@dataclass(frozen=True)
class RobotModel:
    joints: tuple[JointSpec, ...]
    tcp_offset_xyz: Vector3
    base_correction_xyz: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.joints) != 6:
            raise ValueError("ARX X5 model must contain six joints")


def _vector3(value: Any, name: str) -> Vector3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def robot_model_from_config(config: Mapping[str, Any]) -> RobotModel:
    """Build the ARX X5 model directly from robot.yaml data."""

    try:
        dimensions = config["kinematics"]
        joint_values = config["joints"]
        joints = []
        for index, item in enumerate(joint_values):
            limits = np.asarray(item["limits_rad"], dtype=float)
            if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
                raise ValueError(f"joint{index + 1} limits are invalid")
            joints.append(JointSpec(
                str(item["name"]),
                _vector3(item["axis"], f"joint{index + 1} axis"),
                _vector3(item["origin_xyz_m"], f"joint{index + 1} origin_xyz_m"),
                _vector3(item["origin_rpy_rad"], f"joint{index + 1} origin_rpy_rad"),
                float(limits[0]),
                float(limits[1]),
            ))
        return RobotModel(
            tuple(joints),
            _vector3(dimensions["tcp_offset_xyz_m"], "tcp_offset_xyz_m"),
            _vector3(dimensions.get("base_correction_xyz_m", (0.0, 0.0, 0.0)), "base_correction_xyz_m"),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("robot.yaml 运动学配置不完整") from exc


@lru_cache(maxsize=1)
def default_arx5_model() -> RobotModel:
    """Load the default model from the packaged robot.yaml."""

    from ..config import load_robot_config

    return robot_model_from_config(load_robot_config())


@dataclass(frozen=True)
class IkResult:
    joints: Vector6
    position_error_m: float
    orientation_error_rad: float


def format_joint_limit_hits(
    joints: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    edge_tol: float = 0.02,
) -> str:
    """列出触及或超出限位的关节（1-based j1..j6）。"""
    hits: list[str] = []
    # ``zip(strict=...)`` is Python 3.10-only; lengths are validated by the
    # callers and the dock runs Python 3.8.
    for index, (value, lo, hi) in enumerate(zip(joints, lower, upper)):
        joint = index + 1
        q, lo_f, hi_f = float(value), float(lo), float(hi)
        if q < lo_f - 1e-6:
            hits.append(f"j{joint}={q:+.4f}<{lo_f:+.4f}(低于下限)")
        elif q > hi_f + 1e-6:
            hits.append(f"j{joint}={q:+.4f}>{hi_f:+.4f}(高于上限)")
        elif abs(q - lo_f) <= edge_tol:
            hits.append(f"j{joint}={q:+.4f}≈下限{lo_f:+.4f}")
        elif abs(q - hi_f) <= edge_tol:
            hits.append(f"j{joint}={q:+.4f}≈上限{hi_f:+.4f}")
    return ", ".join(hits)


class Arx5Kinematics:
    def __init__(self, model: RobotModel | None = None) -> None:
        self.model = model or default_arx5_model()
        self.lower = np.asarray([joint.lower for joint in self.model.joints])
        self.upper = np.asarray([joint.upper for joint in self.model.joints])
        self.last_ik_diagnostic: str = ""

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Arx5Kinematics":
        return cls(robot_model_from_config(config))

    def _chain(self, joints: Sequence[float]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("joints must contain six finite values")
        transform = _translation(self.model.base_correction_xyz)
        origins: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        for spec, angle in zip(self.model.joints, q):
            transform = transform @ _translation(spec.origin_xyz) @ _rotation(spec.origin_rpy)
            axis = np.asarray(spec.axis, dtype=float)
            if not np.any(axis):
                raise ValueError(f"{spec.name} axis cannot be zero")
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ (axis / np.linalg.norm(axis)))
            transform = transform @ _axis_angle(axis, float(angle))
        return transform, origins, axes

    def forward_matrix(self, joints: Sequence[float]) -> np.ndarray:
        link6, _, _ = self._chain(joints)
        return link6 @ _translation(self.model.tcp_offset_xyz)

    def forward_position(self, joints: Sequence[float]) -> np.ndarray:
        return self.forward_matrix(joints)[:3, 3].copy()

    def jacobian(self, joints: Sequence[float]) -> np.ndarray:
        link6, origins, axes = self._chain(joints)
        tcp = (link6 @ _translation(self.model.tcp_offset_xyz))[:3, 3]
        return np.vstack((
            np.column_stack([np.cross(axis, tcp - origin) for origin, axis in zip(origins, axes)]),
            np.column_stack(axes),
        ))

    @staticmethod
    def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        relative = target @ current.T
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        skew = np.array((relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]))
        if angle < 1e-7:
            return 0.5 * skew
        if math.pi - angle < 1e-5:
            values, vectors = np.linalg.eig(relative)
            axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
            return axis / np.linalg.norm(axis) * angle
        return skew * angle / (2.0 * math.sin(angle))

    def solve(
        self,
        position: Sequence[float],
        rotation: Sequence[Sequence[float]],
        seeds: Iterable[Sequence[float]],
        *,
        position_tolerance_m: float = 0.004,
        orientation_tolerance_rad: float = 0.10,
        max_iterations: int = 350,
        damping: float = 0.035,
        max_step_rad: float = 0.10,
        orientation_weight: float = 0.35,
    ) -> IkResult | None:
        target_p = np.asarray(position, dtype=float)
        target_r = np.asarray(rotation, dtype=float)
        if (
            target_p.shape != (3,)
            or target_r.shape != (3, 3)
            or not np.all(np.isfinite(target_p))
            or not np.all(np.isfinite(target_r))
            or not np.allclose(target_r.T @ target_r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(target_r), 1.0, atol=1e-6)
        ):
            raise ValueError("IK target must contain a 3D position and 3x3 rotation")
        if (
            position_tolerance_m <= 0.0
            or orientation_tolerance_rad <= 0.0
            or max_iterations <= 0
            or damping <= 0.0
            or max_step_rad <= 0.0
            or orientation_weight <= 0.0
        ):
            raise ValueError("IK settings must be positive")
        best: IkResult | None = None
        self.last_ik_diagnostic = ""
        attempt_notes: list[str] = []
        for values in seeds:
            q = np.asarray(values, dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                continue
            q = np.clip(q, self.lower, self.upper)
            seed = q.copy()
            for _ in range(max_iterations):
                transform = self.forward_matrix(q)
                linear = target_p - transform[:3, 3]
                angular = self._orientation_error(transform[:3, :3], target_r)
                if np.linalg.norm(linear) <= position_tolerance_m and np.linalg.norm(angular) <= orientation_tolerance_rad:
                    break
                jacobian = self.jacobian(q)
                jacobian[3:] *= orientation_weight
                error = np.concatenate((linear, angular * orientation_weight))
                step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), error)
                step += (np.eye(6) - np.linalg.pinv(jacobian) @ jacobian) @ (seed - q) * 0.008
                q = np.clip(q + np.clip(step, -max_step_rad, max_step_rad), self.lower, self.upper)
            transform = self.forward_matrix(q)
            result = IkResult(
                tuple(float(value) for value in q),  # type: ignore[arg-type]
                float(np.linalg.norm(target_p - transform[:3, 3])),
                float(np.linalg.norm(self._orientation_error(transform[:3, :3], target_r))),
            )
            if best is None or result.position_error_m + orientation_weight * result.orientation_error_rad < best.position_error_m + orientation_weight * best.orientation_error_rad:
                best = result
            if result.position_error_m <= position_tolerance_m and result.orientation_error_rad <= orientation_tolerance_rad:
                return result
            limit_hits = format_joint_limit_hits(q, self.lower, self.upper)
            if limit_hits:
                attempt_notes.append(
                    f"err=({result.position_error_m:.4f}m,{result.orientation_error_rad:.3f}rad) [{limit_hits}]"
                )
        if best is not None:
            limit_hits = format_joint_limit_hits(best.joints, self.lower, self.upper)
            if limit_hits:
                self.last_ik_diagnostic = f"最近解未达标且触限: [{limit_hits}]"
            else:
                self.last_ik_diagnostic = (
                    f"最近解未达标 pos_err={best.position_error_m:.4f}m "
                    f"ori_err={best.orientation_error_rad:.3f}rad"
                )
        elif attempt_notes:
            self.last_ik_diagnostic = "; ".join(attempt_notes[:4])
        else:
            self.last_ik_diagnostic = "无有效 IK 种子"
        return None


class Arx5SdkKinematics(Arx5Kinematics):
    """FK/IK via ARX5 SDK ``Arx5Solver`` (KDL LMA), matching ``get_eef_state`` / hand-eye."""

    def __init__(
        self,
        solver: Any,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        model: RobotModel | None = None,
    ) -> None:
        super().__init__(model)
        self._solver = solver
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    @classmethod
    def from_project(
        cls,
        root: Path,
        robot: Mapping[str, Any],
        hardware: Mapping[str, Any],
    ) -> "Arx5SdkKinematics":
        module = _load_arx5_sdk(root)
        robot_cfg = module.RobotConfigFactory.get_instance().get_config("X5")
        robot_cfg.urdf_path = str(root / "third_party" / "arx5_sdk" / "models" / "X5.urdf")
        lower = np.maximum(
            np.asarray(robot_cfg.joint_pos_min, dtype=float),
            np.asarray(hardware["joint_lower_rad"], dtype=float),
        )
        upper = np.minimum(
            np.asarray(robot_cfg.joint_pos_max, dtype=float),
            np.asarray(hardware["joint_upper_rad"], dtype=float),
        )
        solver = module.Arx5Solver(
            robot_cfg.urdf_path,
            6,
            lower,
            upper,
            robot_cfg.base_link_name,
            robot_cfg.eef_link_name,
            robot_cfg.gravity_vector,
        )
        return cls(solver, lower, upper, model=robot_model_from_config(robot))

    def _pose6_to_matrix(self, pose6: Sequence[float]) -> np.ndarray:
        values = np.asarray(pose6, dtype=float)
        transform = np.eye(4)
        transform[:3, :3] = rpy_matrix(values[3:])
        transform[:3, 3] = values[:3]
        return transform

    def forward_matrix(self, joints: Sequence[float]) -> np.ndarray:
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("joints must contain six finite values")
        pose6 = np.asarray(self._solver.forward_kinematics(q), dtype=float)
        return self._pose6_to_matrix(pose6)

    def forward_position(self, joints: Sequence[float]) -> np.ndarray:
        return self.forward_matrix(joints)[:3, 3].copy()

    def solve(
        self,
        position: Sequence[float],
        rotation: Sequence[Sequence[float]],
        seeds: Iterable[Sequence[float]],
        *,
        position_tolerance_m: float = 0.004,
        orientation_tolerance_rad: float = 0.10,
        max_iterations: int = 350,
        damping: float = 0.035,
        max_step_rad: float = 0.10,
        orientation_weight: float = 0.35,
    ) -> IkResult | None:
        del max_iterations, damping, max_step_rad  # SDK solver owns iteration strategy.
        target_p = np.asarray(position, dtype=float)
        target_r = np.asarray(rotation, dtype=float)
        if (
            target_p.shape != (3,)
            or target_r.shape != (3, 3)
            or not np.all(np.isfinite(target_p))
            or not np.all(np.isfinite(target_r))
        ):
            raise ValueError("IK target must contain a 3D position and 3x3 rotation")
        pose6 = pose6_from_target(target_p, target_r)
        best: IkResult | None = None
        self.last_ik_diagnostic = ""
        attempt_notes: list[str] = []
        for values in seeds:
            seed = np.asarray(values, dtype=float)
            if seed.shape != (6,) or not np.all(np.isfinite(seed)):
                continue
            ik_status, joints = self._solver.multi_trial_ik(pose6, seed, 5)
            q = np.asarray(joints, dtype=float)
            limit_hits = format_joint_limit_hits(q, self.lower, self.upper)
            if ik_status != 0:
                status_name = self._solver.get_ik_status_name(ik_status)
                note = f"status={status_name}({ik_status})"
                if limit_hits:
                    note += f" [{limit_hits}]"
                attempt_notes.append(note)
                continue
            transform = self.forward_matrix(q)
            result = IkResult(
                tuple(float(value) for value in q),  # type: ignore[arg-type]
                float(np.linalg.norm(target_p - transform[:3, 3])),
                float(np.linalg.norm(self._orientation_error(transform[:3, :3], target_r))),
            )
            metric = result.position_error_m + orientation_weight * result.orientation_error_rad
            if best is None or metric < best.position_error_m + orientation_weight * best.orientation_error_rad:
                best = result
            if (
                result.position_error_m <= position_tolerance_m
                and result.orientation_error_rad <= orientation_tolerance_rad
            ):
                return result
            if limit_hits:
                attempt_notes.append(
                    f"err=({result.position_error_m:.4f}m,{result.orientation_error_rad:.3f}rad) [{limit_hits}]"
                )
        if best is not None:
            limit_hits = format_joint_limit_hits(best.joints, self.lower, self.upper)
            if limit_hits:
                self.last_ik_diagnostic = f"最近解未达标且触限: [{limit_hits}]"
            else:
                self.last_ik_diagnostic = (
                    f"最近解未达标 pos_err={best.position_error_m:.4f}m "
                    f"ori_err={best.orientation_error_rad:.3f}rad"
                )
        elif attempt_notes:
            self.last_ik_diagnostic = "; ".join(attempt_notes[:4])
        else:
            self.last_ik_diagnostic = "无有效 IK 种子"
        return None


def create_kinematics(
    root: Path,
    robot: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> Arx5Kinematics:
    """Prefer SDK ``Arx5Solver``; fall back to numerical IK when SDK is unavailable."""
    try:
        return Arx5SdkKinematics.from_project(root, robot, hardware)
    except RuntimeError:
        return Arx5Kinematics.from_config(robot)
