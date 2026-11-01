"""关节示教与复刻：手动控制机械臂、记录六关节角、回放轨迹。

直接调用 ARX5 SDK（``Arx5JointController``），不经过 ``Arx5Backend``。

示教::
  python3 tests/hardware/teach_joint_pose.py --teach
  python3 tests/hardware/teach_joint_pose.py --teach --output my_pose.json

回放::
  python3 tests/hardware/teach_joint_pose.py --replay
  python3 tests/hardware/teach_joint_pose.py --replay --input my_pose.json --waypoint-hold 0.5
"""

from __future__ import annotations

__test__ = False

import argparse
import importlib
import json
import socket
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = Path("tests/hardware/outputs/teach_joint_pose.json")
JOINT_LABELS = ("j1", "j2", "j3", "j4", "j5", "j6")


def _load_hardware(root: Path) -> dict[str, Any]:
    with (root / "config" / "hardware.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_sdk(root: Path) -> Any:
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


def _record_joints(values: Any) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise RuntimeError("关节反馈无效")
    return {
        "rad": [float(v) for v in array],
        "deg": [float(np.degrees(v)) for v in array],
    }


def _make_joint_controller(arx5: Any, root: Path, hardware: dict[str, Any], interface: str) -> Any:
    robot = arx5.RobotConfigFactory.get_instance().get_config("X5")
    robot.urdf_path = str(root / "third_party" / "arx5_sdk" / "models" / "X5.urdf")
    lower = np.asarray(hardware["joint_lower_rad"], dtype=float)
    upper = np.asarray(hardware["joint_upper_rad"], dtype=float)
    robot.joint_pos_min = np.maximum(np.asarray(robot.joint_pos_min), lower)
    robot.joint_pos_max = np.minimum(np.asarray(robot.joint_pos_max), upper)
    cfg = arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", 6)
    cfg.background_send_recv = True
    cfg.gravity_compensation = True
    cfg.shutdown_to_passive = True
    controller = arx5.Arx5JointController(robot, cfg, interface)
    controller.set_log_level(arx5.LogLevel.WARNING)
    return controller


def _read_joint_state(controller: Any) -> tuple[np.ndarray, float]:
    state = controller.get_joint_state()
    # SDK 返回的 pos() 指向 C++ 缓冲区，必须 copy，否则等待按键期间会被后台 CAN 覆盖。
    joints = np.array(state.pos(), dtype=float, copy=True)
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise RuntimeError("关节反馈无效")
    return joints, float(state.gripper_pos)


def _wait_controller_ready(controller: Any, min_timestamp: float = 0.25) -> None:
    deadline = time.monotonic() + 3.0
    while float(controller.get_timestamp()) < min_timestamp:
        if time.monotonic() > deadline:
            raise RuntimeError("控制器 timestamp 未就绪，请检查 CAN 反馈")
        time.sleep(0.05)


def _print_joints(joints: np.ndarray, prefix: str = "") -> None:
    deg = np.degrees(joints)
    rad_line = " ".join(f"{JOINT_LABELS[i]}={joints[i]:+.4f}" for i in range(6))
    deg_line = " ".join(f"{JOINT_LABELS[i]}={deg[i]:+.1f}°" for i in range(6))
    print(f"{prefix}rad : {rad_line}")
    print(f"{prefix}deg : {deg_line}")


def _hold_current(controller: Any, arx5: Any, joints: np.ndarray, gripper: float) -> None:
    cmd = arx5.JointState(6)
    cmd.pos()[:] = joints
    cmd.gripper_pos = float(gripper)
    cmd.timestamp = float(controller.get_timestamp()) + 0.05
    controller.set_joint_cmd(cmd)


def _enter_damping(controller: Any, arx5: Any, factor: float = 0.08) -> None:
    gain = controller.get_gain()
    damped = arx5.Gain(gain.kp() * 0.0, gain.kd() * factor, gain.gripper_kp * 0.0, gain.gripper_kd * factor)
    controller.set_gain(damped)
    controller.set_to_damping()


def _enter_position_hold(controller: Any, arx5: Any, joints: np.ndarray, gripper: float) -> None:
    _hold_current(controller, arx5, joints, gripper)
    cfg = controller.get_controller_config()
    controller.set_gain(
        arx5.Gain(
            cfg.default_kp,
            cfg.default_kd,
            cfg.default_gripper_kp,
            cfg.default_gripper_kd,
        )
    )


def _move_to_waypoint(
    controller: Any,
    arx5: Any,
    target: np.ndarray,
    gripper: float,
    move_duration_s: float,
) -> None:
    """单点关节命令（对齐 SDK joint_waypoint_scheduling 示例）。"""
    move_duration_s = max(float(move_duration_s), 0.05)
    cmd = arx5.JointState(6)
    cmd.pos()[:] = target
    cmd.gripper_pos = float(gripper)
    cmd.timestamp = float(controller.get_timestamp()) + move_duration_s
    controller.set_joint_cmd(cmd)
    time.sleep(move_duration_s)


def _replay_waypoints(
    controller: Any,
    arx5: Any,
    waypoints: list[np.ndarray],
    gripper: float,
    move_duration_s: float,
    waypoint_hold_s: float,
) -> None:
    if len(waypoints) < 1:
        raise RuntimeError("回放至少需要 1 个路点")
    _wait_controller_ready(controller)
    current, current_gripper = _read_joint_state(controller)
    _enter_position_hold(controller, arx5, current, current_gripper)
    time.sleep(0.15)

    for index, target in enumerate(waypoints):
        target = np.asarray(target, dtype=float)
        print(f"回放路点 {index + 1}/{len(waypoints)}")
        _print_joints(target, prefix="  目标 ")
        _move_to_waypoint(controller, arx5, target, gripper, move_duration_s)
        if waypoint_hold_s > 0.0:
            time.sleep(waypoint_hold_s)
        actual, _ = _read_joint_state(controller)
        _print_joints(actual, prefix="  实测 ")


def _save_session(path: Path, hardware: dict[str, Any], waypoints: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "can_interface": str(hardware.get("can_interface", "can1")),
        "gripper_open": float(hardware.get("gripper_open", 0.088)),
        "waypoint_count": len(waypoints),
        "waypoints": waypoints,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_session(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("waypoints", [])
    if not points:
        raise RuntimeError(f"{path} 中没有路点")
    arrays: list[np.ndarray] = []
    for index, item in enumerate(points):
        joints = np.asarray(item["joints_rad"], dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"路点 #{index} 关节数据无效")
        if np.any(np.abs(joints) > 10.0):
            raise RuntimeError(
                f"路点 #{index} 关节角异常（可能为旧版示教 bug 损坏），请重新 --teach："
                f" {joints.tolist()}"
            )
        arrays.append(joints.copy())
    return arrays


def _help_text() -> str:
    return """
示教命令（单键，无需回车）:
  s  保存当前 6 关节为一路点
  p  打印当前关节
  d  阻尼模式（可手掰示教）
  h  位置保持当前关节
  [  撤销最后一个路点
  l  列出已保存路点
  w  写入 JSON 文件
  q  保存并退出

  1-6        选中关节
  + / =      选中关节 +step
  - / _      选中关节 -step
  ?          显示帮助
"""


def _read_key() -> str:
    if not sys.stdin.isatty():
        raise RuntimeError("示教模式需要 TTY 终端")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_teach(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    hardware = _load_hardware(root)
    interface = str(hardware.get("can_interface", "can1"))
    gripper_open = float(hardware.get("gripper_open", 0.088))
    step = float(args.step_rad)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    try:
        socket.if_nametoindex(interface)
    except OSError as exc:
        raise RuntimeError(f"CAN 接口不存在：{interface}") from exc

    arx5 = _load_sdk(root)
    controller = _make_joint_controller(arx5, root, hardware, interface)
    waypoints: list[dict[str, Any]] = []
    selected_joint = 0
    mode = "hold"

    try:
        joints, gripper = _read_joint_state(controller)
        _enter_position_hold(controller, arx5, joints, gripper)
        print("关节示教已连接 CAN。")
        print(f"输出文件: {output}")
        print(_help_text())

        while True:
            joints, gripper = _read_joint_state(controller)
            print(
                f"\r[{mode}] 路点={len(waypoints)} 选中={JOINT_LABELS[selected_joint]}  ",
                " ".join(f"{JOINT_LABELS[i]}{joints[i]:+.3f}" for i in range(6)),
                end="",
                flush=True,
            )
            key = _read_key()

            if key in "123456":
                selected_joint = int(key) - 1
                print(f"\n选中 {JOINT_LABELS[selected_joint]}")
                continue
            if key in "+=":
                joints, gripper = _read_joint_state(controller)
                joints = joints.copy()
                joints[selected_joint] += step
                _enter_position_hold(controller, arx5, joints, gripper)
                mode = "hold"
                print(f"\n{JOINT_LABELS[selected_joint]} += {step:.4f} rad")
                _print_joints(joints, prefix="  ")
                continue
            if key in "-_":
                joints, gripper = _read_joint_state(controller)
                joints = joints.copy()
                joints[selected_joint] -= step
                _enter_position_hold(controller, arx5, joints, gripper)
                mode = "hold"
                print(f"\n{JOINT_LABELS[selected_joint]} -= {step:.4f} rad")
                _print_joints(joints, prefix="  ")
                continue
            if key == "s":
                joints, gripper = _read_joint_state(controller)
                entry = {
                    "index": len(waypoints),
                    "label": f"wp{len(waypoints)}",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "joints_rad": _record_joints(joints)["rad"],
                    "joints_deg": _record_joints(joints)["deg"],
                }
                waypoints.append(entry)
                print(f"\n已保存路点 wp{len(waypoints) - 1}:")
                _print_joints(joints, prefix="  ")
                continue
            if key == "p":
                joints, gripper = _read_joint_state(controller)
                print("\n当前关节:")
                _print_joints(joints, prefix="  ")
                continue
            if key == "d":
                _enter_damping(controller, arx5)
                mode = "damp"
                print("\n已进入阻尼模式，可手掰机械臂；按 h 恢复位置保持")
                continue
            if key == "h":
                joints, gripper = _read_joint_state(controller)
                _enter_position_hold(controller, arx5, joints, gripper)
                mode = "hold"
                print("\n已恢复位置保持")
                _print_joints(joints, prefix="  ")
                continue
            if key == "[":
                if waypoints:
                    removed = waypoints.pop()
                    print(f"\n已撤销 {removed['label']}")
                else:
                    print("\n没有可撤销的路点")
                continue
            if key == "l":
                print(f"\n已保存 {len(waypoints)} 个路点:")
                for item in waypoints:
                    arr = np.asarray(item["joints_rad"], dtype=float)
                    print(f"  [{item['index']}] ", end="")
                    _print_joints(arr, prefix="    ")
                continue
            if key == "w":
                _save_session(output, hardware, waypoints)
                print(f"\n已写入 {output}")
                continue
            if key == "?":
                print(_help_text())
                continue
            if key == "q":
                _save_session(output, hardware, waypoints)
                print(f"\n已保存 {len(waypoints)} 个路点到 {output}")
                break
            if key == "\x03":
                print("\nCtrl-C，退出前写入文件")
                _save_session(output, hardware, waypoints)
                break
    finally:
        controller.set_to_damping()
    return 0


def run_replay(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    hardware = _load_hardware(root)
    interface = str(hardware.get("can_interface", "can1"))
    gripper_open = float(hardware.get("gripper_open", 0.088))
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    try:
        socket.if_nametoindex(interface)
    except OSError as exc:
        raise RuntimeError(f"CAN 接口不存在：{interface}") from exc

    waypoints = _load_session(input_path)
    print(f"从 {input_path} 读取 {len(waypoints)} 个路点")
    for index, joints in enumerate(waypoints):
        print(f"  [{index}]")
        _print_joints(joints, prefix="    ")

    arx5 = _load_sdk(root)
    controller = _make_joint_controller(arx5, root, hardware, interface)
    try:
        _replay_waypoints(
            controller,
            arx5,
            waypoints,
            gripper_open,
            float(args.segment_duration),
            float(args.waypoint_hold),
        )
        final, _ = _read_joint_state(controller)
        print("回放完成，最终关节:")
        _print_joints(final, prefix="  ")
        if args.hold:
            print("保持使能；按 Ctrl+C 结束。")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n收到停止指令。")
    finally:
        controller.set_to_damping()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="robot_arm 工程根目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="示教 JSON 输出路径")
    parser.add_argument("--input", default=str(DEFAULT_OUTPUT), help="回放 JSON 输入路径")
    parser.add_argument("--step-rad", type=float, default=0.05, help="示教微调步长 (rad)")
    parser.add_argument("--segment-duration", type=float, default=2.5, help="回放移动到每个路点的时长 (秒)")
    parser.add_argument("--waypoint-hold", type=float, default=0.5, help="到达每个路点后停留 (秒)")
    parser.add_argument("--hold", action="store_true", help="回放结束后保持使能")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--teach", action="store_true", help="示教：手动控制并记录路点")
    mode.add_argument("--replay", action="store_true", help="回放：复刻已记录路点")
    args = parser.parse_args(argv)
    if args.teach:
        return run_teach(args)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
