#!/usr/bin/env python3
"""使用独立滑块面板操作 ARX X5，并记录候选关节限位和工作空间。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "mjcf" / "scenes" / "arx_x5_standalone.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "joint_limit_workspace_record.yaml"
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
DEFAULT_POSE = (0.0, 0.645, 1.040, -0.770, 0.0, 0.0)
EXPLORATION_LIMIT_RAD = 2.0 * math.pi


@dataclass
class WorkspaceBounds:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def empty(cls) -> "WorkspaceBounds":
        return cls(np.full(3, np.inf), np.full(3, -np.inf))

    def add(self, point: Sequence[float]) -> None:
        value = np.asarray(point, dtype=float)
        self.minimum = np.minimum(self.minimum, value)
        self.maximum = np.maximum(self.maximum, value)

    @property
    def has_samples(self) -> bool:
        return bool(
            np.all(np.isfinite(self.minimum))
            and np.all(np.isfinite(self.maximum))
        )


class JointLimitTuner:
    """在内存中保存面板目标、MuJoCo 状态和标定记录。"""

    def __init__(
        self,
        mujoco: Any,
        model: Any,
        data: Any,
        output: Path,
        step_rad: float = 0.01,
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.output = output
        self.step_rad = float(step_rad)
        self.selected = 0
        self._lock = threading.RLock()
        self._targets = list(DEFAULT_POSE)
        self._gripper_open = True
        self._message = "拖动滑块即可调节关节"

        self.joint_ids = tuple(int(model.joint(name).id) for name in JOINT_NAMES)
        self.qpos_addresses = tuple(
            int(model.joint(name).qposadr[0]) for name in JOINT_NAMES
        )
        self.actuator_ids = tuple(
            self._id(
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"arx_x5_joint{index}_position",
            )
            for index in range(1, 7)
        )
        self.gripper_actuator_id = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "arx_x5_gripper_position"
        )
        self.gripper_qpos_addresses = tuple(
            int(model.joint(name).qposadr[0])
            for name in ("gripper_opening", "gripper_opening_mirror")
        )
        self.tcp_site_id = self._id(
            mujoco.mjtObj.mjOBJ_SITE, "gripper_grasp_center"
        )
        self.base_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "arx_x5_base"
        )
        self.recorded_lower: list[float | None] = [None] * 6
        self.recorded_upper: list[float | None] = [None] * 6
        self.workspace_samples: list[dict[str, object]] = []
        self.workspace_bounds = WorkspaceBounds.empty()

        # 只在该调试进程的内存模型中解除旧限位。
        for joint_id in self.joint_ids:
            model.jnt_limited[joint_id] = 0
        for actuator_id in self.actuator_ids:
            model.actuator_ctrllimited[actuator_id] = 0
        self.apply_targets()

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    def _id(self, object_type: Any, name: str) -> int:
        value = int(self.mujoco.mj_name2id(self.model, object_type, name))
        if value < 0:
            raise RuntimeError(f"MuJoCo 模型缺少 {name!r}")
        return value

    def targets(self) -> tuple[float, ...]:
        with self._lock:
            return tuple(self._targets)

    def joints(self) -> tuple[float, ...]:
        with self._lock:
            return tuple(
                float(self.data.qpos[address]) for address in self.qpos_addresses
            )

    def set_targets(
        self,
        joints: Sequence[float],
        gripper_open: bool | None = None,
    ) -> None:
        if len(joints) != 6:
            raise ValueError("关节位姿必须包含 6 个数值")
        values = [
            float(np.clip(value, -EXPLORATION_LIMIT_RAD, EXPLORATION_LIMIT_RAD))
            for value in joints
        ]
        with self._lock:
            self._targets = values
            if gripper_open is not None:
                self._gripper_open = bool(gripper_open)

    def set_pose(self, joints: Sequence[float]) -> None:
        """保留便于无界面测试的直接设置入口。"""

        self.set_targets(joints)
        self.apply_targets()

    def apply_targets(self) -> None:
        """像 demoPPO joint 模式一样直接写 qpos 并刷新模型。"""

        with self._lock:
            for address, actuator_id, value in zip(
                self.qpos_addresses, self.actuator_ids, self._targets
            ):
                self.data.qpos[address] = value
                self.data.ctrl[actuator_id] = value
            opening = 0.09 if self._gripper_open else 0.0
            for address in self.gripper_qpos_addresses:
                self.data.qpos[address] = opening / 2.0
            self.data.ctrl[self.gripper_actuator_id] = opening
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)

    def tcp_arm_base(self) -> tuple[float, float, float]:
        with self._lock:
            base_position = np.asarray(
                self.data.xpos[self.base_body_id], dtype=float
            )
            base_rotation = np.asarray(
                self.data.xmat[self.base_body_id], dtype=float
            ).reshape(3, 3)
            tcp_world = np.asarray(
                self.data.site_xpos[self.tcp_site_id], dtype=float
            )
            point = base_rotation.T @ (tcp_world - base_position)
            return tuple(float(value) for value in point)  # type: ignore[return-value]

    def adjust(self, direction: float) -> None:
        values = list(self.targets())
        values[self.selected] = float(
            np.clip(
                values[self.selected] + direction * self.step_rad,
                -EXPLORATION_LIMIT_RAD,
                EXPLORATION_LIMIT_RAD,
            )
        )
        self.set_pose(values)
        with self._lock:
            self._message = (
                f"{JOINT_NAMES[self.selected]} = {values[self.selected]:.5f} rad "
                f"({math.degrees(values[self.selected]):.2f} deg)"
            )

    def select(self, index: int) -> None:
        with self._lock:
            self.selected = index
            self._message = f"已选择 {JOINT_NAMES[index]}"

    def record_limit(self, index: int | None = None, *, lower: bool) -> None:
        with self._lock:
            target_index = self.selected if index is None else index
            value = self._targets[target_index]
            records = self.recorded_lower if lower else self.recorded_upper
            records[target_index] = value
            label = "下限" if lower else "上限"
            self._message = (
                f"已记录 {JOINT_NAMES[target_index]} {label}: "
                f"{value:.5f} rad ({math.degrees(value):.2f} deg)"
            )
        self.print_status()

    def record_workspace(self) -> None:
        self.apply_targets()
        point = self.tcp_arm_base()
        with self._lock:
            sample = {
                "joint_pos_rad": [round(value, 8) for value in self._targets],
                "tcp_position_arm_base_m": [round(value, 8) for value in point],
            }
            self.workspace_samples.append(sample)
            self.workspace_bounds.add(point)
            self._message = (
                f"已记录工作空间样本 {len(self.workspace_samples)}: "
                f"({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}) m"
            )
        self.print_status()

    def reset(self) -> None:
        self.set_pose(DEFAULT_POSE)
        with self._lock:
            self._gripper_open = True
            self._message = "已恢复用户确认的启动初始姿态"

    def save(self) -> None:
        self.apply_targets()
        tcp = self.tcp_arm_base()
        with self._lock:
            limits: dict[str, dict[str, float | None]] = {}
            for index, name in enumerate(JOINT_NAMES):
                lower = self.recorded_lower[index]
                upper = self.recorded_upper[index]
                limits[name] = {
                    "lower_rad": None if lower is None else round(lower, 8),
                    "upper_rad": None if upper is None else round(upper, 8),
                    "lower_deg": (
                        None if lower is None else round(math.degrees(lower), 4)
                    ),
                    "upper_deg": (
                        None if upper is None else round(math.degrees(upper), 4)
                    ),
                }
            workspace: dict[str, object] = {
                "sample_count": len(self.workspace_samples),
                "samples": list(self.workspace_samples),
            }
            if self.workspace_bounds.has_samples:
                radii = [
                    math.hypot(
                        float(sample["tcp_position_arm_base_m"][0]),  # type: ignore[index]
                        float(sample["tcp_position_arm_base_m"][1]),  # type: ignore[index]
                    )
                    for sample in self.workspace_samples
                ]
                workspace.update(
                    {
                        "minimum_xyz_m": [
                            round(float(value), 8)
                            for value in self.workspace_bounds.minimum
                        ],
                        "maximum_xyz_m": [
                            round(float(value), 8)
                            for value in self.workspace_bounds.maximum
                        ],
                        "minimum_radius_m": round(min(radii), 8),
                        "maximum_radius_m": round(max(radii), 8),
                    }
                )
            document = {
                "schema_version": 1,
                "warning": "候选标定记录；尚未写入机器人正式配置",
                "control_mode": "direct_mujoco_qpos",
                "exploration_hard_limit_rad": EXPLORATION_LIMIT_RAD,
                "slider_resolution_rad": self.step_rad,
                "joint_limits": limits,
                "workspace_arm_base": workspace,
                "last_pose": {
                    "joint_pos_rad": [
                        round(value, 8) for value in self._targets
                    ],
                    "tcp_position_arm_base_m": [
                        round(value, 8) for value in tcp
                    ],
                    "gripper_open": self._gripper_open,
                },
            }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        with self._lock:
            self._message = f"已保存: {self.output}"
        print(f"\n{self.message}")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "targets": tuple(self._targets),
                "actual": tuple(
                    float(self.data.qpos[address])
                    for address in self.qpos_addresses
                ),
                "tcp": self.tcp_arm_base(),
                "lower": tuple(self.recorded_lower),
                "upper": tuple(self.recorded_upper),
                "sample_count": len(self.workspace_samples),
                "message": self._message,
                "gripper_open": self._gripper_open,
            }

    def key_callback(self, keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 128 else ""
        if key in "123456":
            self.select(int(key) - 1)
        elif key in {"=", "+"}:
            self.adjust(1.0)
        elif key in {"-", "_"}:
            self.adjust(-1.0)
        elif key == "L":
            self.record_limit(lower=True)
        elif key == "U":
            self.record_limit(lower=False)
        elif key == "W":
            self.record_workspace()
        elif key == "R":
            self.reset()
        elif key == "S":
            self.save()

    def overlay(self) -> list[tuple[int, int, str, str]]:
        snapshot = self.snapshot()
        values = snapshot["actual"]
        tcp = snapshot["tcp"]
        lines = [
            f"J{index + 1}: {value:+.4f} rad ({math.degrees(value):+.1f} deg)"
            for index, value in enumerate(values)  # type: ignore[arg-type]
        ]
        text = (
            "\n".join(lines)
            + f"\nTCP arm_base: ({tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f}) m"  # type: ignore[index]
            + f"\n工作空间样本: {snapshot['sample_count']}"
        )
        return [(1, 0, "ARX X5 手动关节调试", text)]

    def print_status(self) -> None:
        snapshot = self.snapshot()
        print(f"\n{snapshot['message']}")
        print(
            "当前关节角: "
            f"{[round(value, 6) for value in snapshot['targets']]}"  # type: ignore[union-attr]
        )
        print(
            "当前 TCP arm_base: "
            f"{[round(value, 6) for value in snapshot['tcp']]}"  # type: ignore[union-attr]
        )


class TkJointCalibrationPanel:
    """参考 demoPPO joint 模式的六关节滑块面板。"""

    def __init__(
        self,
        tuner: JointLimitTuner,
        stop_event: threading.Event,
        resolution_rad: float,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.tuner = tuner
        self.stop_event = stop_event
        self.root = tk.Tk()
        self.root.title("ARX X5 Joint Calibration")
        self.root.geometry("680x780")
        self.root.minsize(620, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.sliders: list[Any] = []
        self.value_labels: list[Any] = []
        self.gripper_var = tk.BooleanVar(value=True)
        self._building = True

        title = tk.Label(
            self.root,
            text="ARX X5 关节手动标定",
            font=("Sans", 16, "bold"),
        )
        title.pack(pady=(12, 4))
        warning = tk.Label(
            self.root,
            text="滑块直接写入 MuJoCo qpos；当前为候选限位探索，不会改动正式配置。",
            fg="#9a3d00",
        )
        warning.pack(pady=(0, 8))

        initial = tuner.targets()
        for index, value in enumerate(initial):
            frame = tk.LabelFrame(self.root, text=f"Joint {index + 1}")
            frame.pack(fill="x", padx=12, pady=4)
            value_label = tk.Label(frame, width=36, anchor="w")
            value_label.grid(row=0, column=0, padx=6, sticky="w")
            self.value_labels.append(value_label)
            button_frame = tk.Frame(frame)
            button_frame.grid(row=0, column=1, padx=4, sticky="e")
            tk.Button(
                button_frame,
                text="记下限",
                width=8,
                command=lambda i=index: self.record_limit(i, lower=True),
            ).pack(side="left", padx=2)
            tk.Button(
                button_frame,
                text="记上限",
                width=8,
                command=lambda i=index: self.record_limit(i, lower=False),
            ).pack(side="left", padx=2)
            slider = tk.Scale(
                frame,
                from_=-EXPLORATION_LIMIT_RAD,
                to=EXPLORATION_LIMIT_RAD,
                resolution=resolution_rad,
                orient="horizontal",
                length=620,
                showvalue=False,
                command=lambda _value, i=index: self.slider_changed(i),
            )
            slider.set(value)
            slider.grid(row=1, column=0, columnspan=2, padx=6, sticky="ew")
            frame.columnconfigure(0, weight=1)
            self.sliders.append(slider)
            self.update_joint_label(index, value, None, None)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=8)
        tk.Checkbutton(
            controls,
            text="夹爪打开",
            variable=self.gripper_var,
            command=self.push_slider_values,
        ).pack(side="left", padx=4)
        tk.Button(
            controls, text="记录 TCP 样本", command=self.record_workspace
        ).pack(side="left", padx=4)
        tk.Button(controls, text="恢复启动初始姿态", command=self.reset).pack(
            side="left", padx=4
        )
        tk.Button(controls, text="保存 YAML", command=self.save).pack(
            side="left", padx=4
        )

        self.tcp_label = tk.Label(
            self.root, text="TCP arm_base: --", font=("Monospace", 11)
        )
        self.tcp_label.pack(anchor="w", padx=18, pady=(2, 2))
        self.sample_label = tk.Label(self.root, text="工作空间样本: 0")
        self.sample_label.pack(anchor="w", padx=18)
        self.status_label = tk.Label(
            self.root,
            text=tuner.message,
            anchor="w",
            justify="left",
            fg="#164a7b",
        )
        self.status_label.pack(fill="x", padx=18, pady=(4, 8))
        self._building = False
        self.push_slider_values()
        self.root.after(50, self.refresh_status)

    def slider_changed(self, index: int) -> None:
        if self._building:
            return
        value = float(self.sliders[index].get())
        self.update_joint_label(
            index,
            value,
            self.tuner.recorded_lower[index],
            self.tuner.recorded_upper[index],
        )
        self.push_slider_values()

    def push_slider_values(self) -> None:
        if self._building:
            return
        values = [float(slider.get()) for slider in self.sliders]
        self.tuner.set_targets(values, bool(self.gripper_var.get()))

    def update_joint_label(
        self,
        index: int,
        value: float,
        lower: float | None,
        upper: float | None,
    ) -> None:
        lower_text = "--" if lower is None else f"{lower:+.4f}"
        upper_text = "--" if upper is None else f"{upper:+.4f}"
        self.value_labels[index].configure(
            text=(
                f"{value:+.4f} rad / {math.degrees(value):+.2f}°   "
                f"记录 [{lower_text}, {upper_text}]"
            )
        )

    def record_limit(self, index: int, *, lower: bool) -> None:
        self.push_slider_values()
        self.tuner.record_limit(index, lower=lower)
        snapshot = self.tuner.snapshot()
        self.update_joint_label(
            index,
            float(self.sliders[index].get()),
            snapshot["lower"][index],  # type: ignore[index]
            snapshot["upper"][index],  # type: ignore[index]
        )

    def record_workspace(self) -> None:
        self.push_slider_values()
        self.tuner.record_workspace()

    def reset(self) -> None:
        self._building = True
        for slider, value in zip(self.sliders, DEFAULT_POSE):
            slider.set(value)
        self.gripper_var.set(True)
        self._building = False
        self.tuner.reset()
        self.push_slider_values()

    def save(self) -> None:
        self.push_slider_values()
        self.tuner.save()
        self.messagebox.showinfo("保存完成", f"记录已保存到：\n{self.tuner.output}")

    def refresh_status(self) -> None:
        if self.stop_event.is_set():
            self.close()
            return
        snapshot = self.tuner.snapshot()
        tcp = snapshot["tcp"]
        self.tcp_label.configure(
            text=(
                "TCP arm_base: "
                f"X={tcp[0]:+.5f}  Y={tcp[1]:+.5f}  Z={tcp[2]:+.5f} m"  # type: ignore[index]
            )
        )
        self.sample_label.configure(
            text=f"工作空间样本: {snapshot['sample_count']}"
        )
        self.status_label.configure(text=str(snapshot["message"]))
        self.root.after(50, self.refresh_status)

    def close(self) -> None:
        if not self.stop_event.is_set():
            self.push_slider_values()
            self.tuner.save()
            self.stop_event.set()
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def run(self) -> None:
        self.root.mainloop()


def run_mujoco_viewer(
    tuner: JointLimitTuner,
    stop_event: threading.Event,
    ready_event: threading.Event,
    errors: list[BaseException],
    show_ui: bool = False,
) -> None:
    viewer = None
    try:
        import mujoco.viewer

        # MuJoCo 3.x 的 passive viewer 不提供窗口句柄。设置 GLFW 的
        # MAXIMIZED hint 必须在 launch_passive 创建窗口前完成，Wayland 和
        # X11 都会据此以最大化窗口启动。若当前桌面环境不允许设置 hint，
        # 仍继续启动，不影响鼠标相机控制。
        try:
            import glfw

            if glfw.init():
                glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
        except Exception:
            pass

        viewer = mujoco.viewer.launch_passive(
            tuner.model,
            tuner.data,
            key_callback=tuner.key_callback,
            # 左右面板会占用大量视口并遮住模型；关闭它们后，整个窗口
            # 都是可用的 3D 视图，左键旋转、右键/滚轮缩放、中键平移。
            show_left_ui=show_ui,
            show_right_ui=show_ui,
        )
        viewer.cam.lookat[:] = (0.0, 0.0, 0.25)
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0
        ready_event.set()
        while viewer.is_running() and not stop_event.is_set():
            tuner.apply_targets()
            viewer.set_texts(tuner.overlay())
            viewer.sync()
            time.sleep(0.02)
    except BaseException as exc:
        errors.append(exc)
        ready_event.set()
    finally:
        stop_event.set()
        if viewer is not None:
            viewer.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sim"], default="sim")
    parser.add_argument("--control-mode", choices=["joint"], default="joint")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resolution-rad",
        type=float,
        default=0.001,
        help="滑块最小分辨率，单位为弧度",
    )
    parser.add_argument(
        "--viewer-ui",
        action="store_true",
        help="显示 MuJoCo 左右控制面板；默认隐藏以扩大鼠标 3D 视图",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0001 <= args.resolution_rad <= 0.1:
        raise SystemExit("--resolution-rad 必须位于 0.0001 到 0.1 之间")
    try:
        import mujoco
        import tkinter  # noqa: F401
    except ImportError as exc:
        raise SystemExit("需要 MuJoCo 和 Tkinter 才能启动滑块面板") from exc

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    tuner = JointLimitTuner(
        mujoco,
        model,
        data,
        args.output.resolve(),
        args.resolution_rad,
    )
    stop_event = threading.Event()
    ready_event = threading.Event()
    errors: list[BaseException] = []
    viewer_thread = threading.Thread(
        target=run_mujoco_viewer,
        args=(tuner, stop_event, ready_event, errors, args.viewer_ui),
        name="mujoco-joint-tuner",
        daemon=True,
    )
    viewer_thread.start()
    ready_event.wait(timeout=10.0)
    if errors:
        raise SystemExit(f"MuJoCo viewer 启动失败: {errors[0]}")
    if not ready_event.is_set():
        stop_event.set()
        raise SystemExit("MuJoCo viewer 启动超时")

    print("ARX X5 滑块关节标定工具已启动。")
    print(f"记录文件: {tuner.output}")
    panel = TkJointCalibrationPanel(tuner, stop_event, args.resolution_rad)
    try:
        panel.run()
    finally:
        stop_event.set()
        viewer_thread.join(timeout=2.0)
        tuner.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
