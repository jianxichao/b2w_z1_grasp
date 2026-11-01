"""Small data contracts shared by planning, simulation, and hardware."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Tuple, runtime_checkable

try:
    from typing import TypeAlias
except ImportError:  # Python 3.8 on the Unitree dock
    TypeAlias = Any  # type: ignore[misc,assignment]

import numpy as np

Vector3: TypeAlias = Tuple[float, float, float]
Vector6: TypeAlias = Tuple[float, float, float, float, float, float]
Matrix3: TypeAlias = Tuple[Vector3, Vector3, Vector3]


class HardwareFault(RuntimeError):
    """A feedback or controller fault that must not be retried automatically."""


def _vector(values: object, size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(value) for value in array)


def _rotation(values: object) -> Matrix3:
    matrix = np.asarray(values, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6)
    ):
        raise ValueError("rotation must be a proper 3x3 matrix")
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]


@dataclass(frozen=True)
class Pose:
    position: Vector3
    rotation: Matrix3
    frame_id: str = "arm_base"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        object.__setattr__(self, "rotation", _rotation(self.rotation))
        if not self.frame_id:
            raise ValueError("frame_id cannot be empty")

    @classmethod
    def from_xyz(cls, position: object, frame_id: str = "arm_base") -> "Pose":
        return cls(
            _vector(position, 3, "position"),  # type: ignore[arg-type]
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            frame_id,
        )


@dataclass(frozen=True)
class Target:
    target_id: str
    pose: Pose
    confidence: float = 1.0
    stamp_s: float | None = None

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id cannot be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.stamp_s is not None and not math.isfinite(self.stamp_s):
            raise ValueError("stamp_s must be finite")


@dataclass(frozen=True)
class MotionCommand:
    name: str
    joints: Vector6
    gripper: float
    duration_s: float
    target_pose: Pose | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("command name cannot be empty")
        object.__setattr__(self, "joints", _vector(self.joints, 6, "joints"))
        if not math.isfinite(self.gripper) or not math.isfinite(self.duration_s):
            raise ValueError("gripper and duration_s must be finite")


@dataclass(frozen=True)
class PickPlan:
    target: Target
    commands: tuple[MotionCommand, ...]

    def command(self, name: str) -> MotionCommand:
        matches = [command for command in self.commands if command.name == name]
        if len(matches) != 1:
            raise LookupError(f"plan has {len(matches)} commands named {name!r}")
        return matches[0]


@dataclass(frozen=True)
class ExecutionResult:
    reached: bool
    reason: str = ""
    cancelled: bool = False


@runtime_checkable
class ArmBackend(Protocol):
    """The only operations that the grasp workflow may request from an arm."""

    def connect(self) -> None: ...

    def self_check(self) -> tuple[bool, str]: ...

    def current_joints(self) -> Vector6: ...

    def home(self) -> ExecutionResult | None: ...

    def search_pose(self) -> ExecutionResult | None: ...

    def execute(self, command: MotionCommand) -> ExecutionResult: ...

    def has_object(self) -> bool: ...

    def recover(self, plan: PickPlan | None) -> ExecutionResult | None: ...

    def hold_enabled(self, *, block: bool = True) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class TargetProvider(Protocol):
    """Return one stable target already expressed in the planning frame."""

    def lock(self) -> Target: ...

    def close(self) -> None: ...
