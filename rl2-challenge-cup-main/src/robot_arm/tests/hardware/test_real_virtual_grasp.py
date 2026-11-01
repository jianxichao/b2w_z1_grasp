"""真机虚拟目标验收。

该文件不会被普通 pytest 自动执行；请按 docs/x5_real_virtual_test.md 显式启动。
"""

from __future__ import annotations

__test__ = False

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_arm.app import _core  # noqa: E402
from robot_arm.config import load_config  # noqa: E402
from robot_arm.hardware import Arx5Backend  # noqa: E402
from robot_arm.manipulation import Pose, Target  # noqa: E402
from robot_arm.manipulation.state_machine import GraspStateMachine  # noqa: E402
from robot_arm.vision import camera_point_to_base, load_eye_in_hand  # noqa: E402


# 由 SDK FK 反算；搜索姿态下约为 arm_base (0.30, 0.15, 0.15) m。
CAMERA_POINT_M = (-0.1443, 0.2279, 0.1743)


class VirtualCameraTarget:
    """用固定相机坐标生成一次目标，不打开相机或 YOLO。"""

    def __init__(self, arm: Any, eef_T_camera: np.ndarray) -> None:
        self.arm = arm
        self.eef_T_camera = eef_T_camera

    def lock(self) -> Target:
        base_xyz = camera_point_to_base(
            CAMERA_POINT_M, self.arm.base_T_eef(), self.eef_T_camera
        )
        print(
            "虚拟相机坐标:",
            " ".join(f"{value:+.4f}" for value in CAMERA_POINT_M),
            "-> arm_base:",
            " ".join(f"{value:+.4f}" for value in base_xyz),
        )
        return Target("virtual_camera_target", Pose.from_xyz(base_xyz), 1.0, time.time())

    def close(self) -> None:
        pass


class VirtualObjectArm:
    """仅在显式 --virtual-object 时跳过真实夹爪接触反馈。"""

    def __init__(self, backend: Arx5Backend) -> None:
        self.backend = backend
        self.holding = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def execute(self, command: Any) -> Any:
        result = self.backend.execute(command)
        if result.reached and command.name == "close":
            self.holding = True
        elif result.reached and command.name == "open":
            self.holding = False
        return result

    def has_object(self) -> bool:
        return self.holding


def run(root: Path, real: bool, virtual_object: bool) -> int:
    config = load_config(root)
    planner, safety = _core(config)
    backend = Arx5Backend(
        config.root,
        config.hardware,
        dry_run=not real,
        kinematics=planner.kinematics,
    )
    arm: Any = VirtualObjectArm(backend) if virtual_object else backend
    provider = VirtualCameraTarget(
        arm, load_eye_in_hand(config.resolve("config/eye_in_hand_calibration.json"))
    )
    machine = GraspStateMachine(
        arm,
        provider,
        planner,
        safety,
        lambda state: print(state.value),
        hold_after_done=real,
    )
    result = machine.run()
    print("状态历史:", " -> ".join(result.history))
    print("验收通过" if result.success else f"验收失败：{result.error}")
    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="项目根目录")
    parser.add_argument("--confirm", action="store_true", help="确认允许真机运动")
    parser.add_argument(
        "--check-only", action="store_true", help="只用同一状态机做空运行检查"
    )
    parser.add_argument(
        "--virtual-object",
        action="store_true",
        help="把闭合夹爪视为持物；只验证运动链，不验证真实接触力",
    )
    args = parser.parse_args(argv)
    if args.confirm == args.check_only:
        parser.error("二选一：--check-only 或 --confirm")
    if args.confirm:
        print("警告：即将连接 CAN 并执行 HOME/SEARCH/抓取/放置/撤离运动。")
        if not args.virtual_object:
            print("未使用 --virtual-object：请把轻小目标放在打印出的 arm_base 坐标附近。")
    return run(Path(args.root).resolve(), real=args.confirm, virtual_object=args.virtual_object)


if __name__ == "__main__":
    raise SystemExit(main())
