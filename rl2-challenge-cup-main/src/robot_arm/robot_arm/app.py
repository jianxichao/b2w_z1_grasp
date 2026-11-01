"""仿真、真机空运行和真机抓取的统一入口。"""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np

from .config import load_config
from .hardware import Arx5Backend
from .manipulation import (
    GraspPlanner,
    MotionSettings,
    PosePolicy,
    SafetyGuard,
    SafetyLimits,
    create_kinematics,
)
from .manipulation.state_machine import GraspStateMachine
from .simulation import FixedTargetProvider, MujocoBackend
from .vision import OrbbecCamera, TargetProvider, YoloTracker, load_eye_in_hand


T = TypeVar("T")


def _dataclass(cls: type[T], values: Mapping[str, Any]) -> T:
    names = {item.name for item in fields(cls)}
    return cls(**{name: value for name, value in values.items() if name in names})


def _demo_class_map(mode: str, mapping: Mapping[str, Any]) -> dict[str, str]:
    enabled = bool(mapping.get("enabled", False))
    if mode == "real" and enabled:
        raise ValueError("真机模式禁止 demo_mapping，请使用专用权重或关闭映射")
    return dict(mapping.get("class_map", {})) if mode == "dry-run" and enabled else {}


def build_core(config: Any) -> tuple[GraspPlanner, SafetyGuard]:
    kinematics = create_kinematics(config.root, config.robot, config.hardware)
    motion = config.motion
    policy = _dataclass(PosePolicy, motion["pose_policy"])
    settings = MotionSettings(
        home_joints=tuple(motion["home_joints"]),
        search_joints=tuple(motion["search_joints"]),
        open_gripper=float(motion["gripper"]["open_m"]),
        closed_gripper=float(motion["gripper"]["closed_m"]),
        pregrasp_offset_m=float(motion["planning"]["pregrasp_offset_m"]),
        lift_offset_m=float(motion["planning"]["lift_offset_m"]),
        transit_height_m=float(motion["planning"]["transit_height_m"]),
        place_position=tuple(motion["planning"]["place_position_m"]),
        place_clearance_m=float(motion["planning"]["place_clearance_m"]),
        durations_s=motion["durations_s"],
        pose_policy=policy,
    )
    safety = config.safety
    workspace = safety["workspace"]
    limits = SafetyLimits(
        joint_lower=tuple(safety["joint_lower_rad"]),
        joint_upper=tuple(safety["joint_upper_rad"]),
        minimum_radius_m=float(workspace["minimum_radius_m"]),
        maximum_radius_m=float(workspace["maximum_radius_m"]),
        minimum_z_m=float(workspace["minimum_z_m"]),
        maximum_z_m=float(workspace["maximum_z_m"]),
        minimum_x_m=float(workspace["minimum_x_m"]),
        maximum_abs_y_m=float(workspace["maximum_abs_y_m"]),
        maximum_abs_heading_rad=float(workspace["maximum_abs_heading_rad"]),
        trajectory_step_rad=float(safety["trajectory_step_rad"]),
        target_max_age_s=float(safety["target_max_age_s"]),
        minimum_confidence=float(safety["minimum_confidence"]),
        gripper_min=float(safety["gripper_min_m"]),
        gripper_max=float(safety["gripper_max_m"]),
    )
    return GraspPlanner(kinematics, settings), SafetyGuard(kinematics, limits)


# 保留旧名称，避免已有验收脚本失效。
_core = build_core


def build_runtime(args: argparse.Namespace) -> GraspStateMachine:
    config = load_config(args.root)
    planner, safety = build_core(config)
    if args.mode == "sim":
        arm = MujocoBackend(config.root, config.simulation, args.viewer)
        targets = FixedTargetProvider(config.simulation["target_position_m"])
    else:
        arm = Arx5Backend(
            config.root,
            config.hardware,
            dry_run=args.mode == "dry-run",
            kinematics=planner.kinematics,
        )
        camera_cfg = config.camera["camera"]
        camera = OrbbecCamera(
            width=int(camera_cfg["width"]),
            height=int(camera_cfg["height"]),
            fps=int(camera_cfg["fps"]),
            depth_width=int(camera_cfg.get("depth_width", camera_cfg["width"])),
            depth_height=int(camera_cfg.get("depth_height", 800)),
            min_depth_m=float(camera_cfg["min_depth_m"]),
            max_depth_m=float(camera_cfg["max_depth_m"]),
            calibrated_camera_matrix=np.asarray(config.calibration["camera_matrix"]),
        )
        vision = config.vision
        model = vision["model"]
        mapping = vision["demo_mapping"]
        class_map = _demo_class_map(args.mode, mapping)
        configured_weights = str(model["weights"])
        local_weights = config.resolve(configured_weights)
        tracker = YoloTracker(
            weights=str(local_weights) if local_weights.is_file() else configured_weights,
            confidence=float(model["confidence"]),
            device=str(model["device"]),
            tracker=str(model["tracker"]),
            image_size=int(model.get("image_size", 416)),
            class_map=class_map,
            target_classes=set(vision["target_classes"]),
        )
        targets = TargetProvider(
            camera,
            tracker,
            load_eye_in_hand(config.resolve("config/eye_in_hand_calibration.json")),
            arm.base_T_eef,
            target_class=str(vision["event_policy"]["target_class"]),
            acquire_confidence=float(vision["event_policy"]["acquire_confidence"]),
            stable_frames=int(vision["event_policy"]["seen_confirm_frames"]),
            max_spread_m=float(vision["event_policy"]["max_position_spread_m"]),
            lock_timeout_s=float(vision["event_policy"]["lock_timeout_s"]),
            show=bool(config.camera["display"]["enabled"]),
            show_depth=bool(config.camera["display"]["show_depth"]),
            distortion=config.calibration["distortion_coefficients"],
            continuous_tracking=bool(config.camera["display"]["continuous_tracking"]),
        )
    # 真机普通目标失败时保持同一 CAN 控制器，回搜索位继续找目标。
    retries = None if args.mode == "real" else 2
    return GraspStateMachine(
        arm,
        targets,
        planner,
        safety,
        lambda state: print(state.value),
        max_retries=retries,
        hold_after_done=args.mode == "real",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "dry-run", "real"), default="sim")
    parser.add_argument("--root", default=None, help="项目根目录，默认自动定位")
    parser.add_argument("--viewer", action="store_true", help="仿真时打开 MuJoCo 窗口")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_runtime(args).run()
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"启动失败：{exc}")
        return 2
    print("抓取完成" if result.success else f"抓取失败：{result.error}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
