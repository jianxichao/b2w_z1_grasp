"""集中读取项目配置，业务模块不直接访问文件系统。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

import numpy as np


ConfigMap = Mapping[str, Any]


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    robot: ConfigMap
    motion: ConfigMap
    safety: ConfigMap
    simulation: ConfigMap
    hardware: ConfigMap
    camera: ConfigMap
    vision: ConfigMap
    calibration: ConfigMap

    def resolve(self, relative_path: str) -> Path:
        """将配置中的相对路径转换为项目内路径。"""

        return (self.root / relative_path).resolve()


def _default_root() -> Path:
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "config").is_dir():
        return source_root
    candidates = [
        source_root / "share" / "robot_arm",
        Path(sys.prefix) / "share" / "robot_arm",
    ]
    scripts = Path(sys.executable).resolve().parent
    candidates.append(scripts.parent / "share" / "robot_arm")
    for entry in sys.path:
        if entry:
            path = Path(entry).resolve()
            candidates.extend(
                (path / "share" / "robot_arm", path.parent / "share" / "robot_arm")
            )
    for candidate in candidates:
        if (candidate / "config").is_dir():
            return candidate
    try:
        from ament_index_python.packages import get_package_share_directory

        ament_root = Path(get_package_share_directory("robot_arm"))
        if (ament_root / "config").is_dir():
            return ament_root
    except (ImportError, LookupError):
        pass
    return source_root


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取配置：{path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"配置缺少 schema_version: 1：{path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取配置：{path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"配置缺少 schema_version: 1：{path}")
    return value


def load_robot_config(root: str | Path | None = None) -> dict[str, Any]:
    """读取机械臂运动学配置。"""

    project_root = Path(root).resolve() if root else _default_root()
    robot = _load_yaml(project_root / "config/robot.yaml")
    if len(robot.get("joints", [])) != 6:
        raise ValueError("robot.yaml 必须定义 6 个关节")
    return robot


def load_config(root: str | Path | None = None) -> ProjectConfig:
    """一次性读取全部配置，并检查最关键的结构。"""

    project_root = Path(root).resolve() if root else _default_root()
    config_dir = project_root / "config"
    robot = load_robot_config(project_root)
    motion = _load_yaml(config_dir / "motion.yaml")
    safety = _load_yaml(config_dir / "safety.yaml")
    calibration = _load_json(config_dir / "eye_in_hand_calibration.json")

    if len(motion.get("home_joints", [])) != 6:
        raise ValueError("motion.yaml 的 home_joints 必须包含 6 个数值")
    if calibration.get("transform_direction") != "camera_to_eef":
        raise ValueError("手眼标定矩阵必须明确为 camera_to_eef")
    hand_eye = calibration.get("camera_to_eef", {})
    rotation = np.asarray(hand_eye.get("rotation_matrix"), dtype=float)
    translation = np.asarray(hand_eye.get("translation"), dtype=float)
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("手眼标定矩阵不是有效的刚体变换")

    hardware = _load_yaml(config_dir / "hardware.yaml")
    for key in ("joint_lower_rad", "joint_upper_rad"):
        if hardware.get(key) != safety.get(key):
            raise ValueError(f"hardware.yaml 的 {key} 必须与 safety.yaml 一致")

    return ProjectConfig(
        root=project_root,
        robot=robot,
        motion=motion,
        safety=safety,
        simulation=_load_yaml(config_dir / "simulation.yaml"),
        hardware=hardware,
        camera=_load_yaml(config_dir / "camera.yaml"),
        vision=_load_yaml(config_dir / "target_classes.yaml"),
        calibration=calibration,
    )
