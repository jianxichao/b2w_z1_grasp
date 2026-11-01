"""Gemini 2 最小打开测试（不连 CAN、不连机械臂）。

检查 USB 枚举、pyorbbecsdk、RGB 与 RGB-D 对齐流能否各采一帧。
可选 ``--preview`` 弹出预览窗口（需 DISPLAY 与 opencv-python GUI）。

显式启动::

  python3 tests/hardware/test_gemini2_open.py
  python3 tests/hardware/test_gemini2_open.py --preview
  python3 tests/hardware/test_gemini2_open.py --rgb-only
"""

from __future__ import annotations

__test__ = False

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_arm.config import load_config  # noqa: E402
from robot_arm.vision import OrbbecCamera  # noqa: E402


def _require_orbbec_sdk() -> None:
    if importlib.util.find_spec("pyorbbecsdk") is None:
        raise ImportError(
            "缺少 pyorbbecsdk。请执行：\n"
            '  pip install "pyorbbecsdk2==2.0.15"\n'
            "  python3 -c \"import pyorbbecsdk; print(pyorbbecsdk.__file__)\""
        )


def _usb_hint() -> None:
    try:
        out = subprocess.check_output(["lsusb", "-d", "2bc5:"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = ""
    if out.strip():
        print("lsusb:", out.strip())
    else:
        print("lsusb: 未找到 Orbbec (2bc5:*)，请检查 USB 连接。")


def _sdk_device_count() -> int:
    import pyorbbecsdk as sdk

    ctx = sdk.Context()
    devs = ctx.query_devices()
    return int(devs.get_count())


def _print_frame_summary(label: str, frame: object) -> None:
    color = frame.color_bgr
    print(f"[{label}] RGB {color.shape[1]}x{color.shape[0]} mean_bgr={color.mean(axis=(0, 1))}")
    matrix = frame.camera_matrix
    print(
        f"[{label}] intrinsics fx={matrix[0, 0]:.2f} fy={matrix[1, 1]:.2f} "
        f"cx={matrix[0, 2]:.2f} cy={matrix[1, 2]:.2f}"
    )
    depth = frame.depth_m
    if depth.size == 0:
        print(f"[{label}] depth: (disabled, color_only)")
        return
    valid = depth[(depth > 0.0) & np.isfinite(depth)]
    if valid.size == 0:
        print(f"[{label}] depth: 1280x720 aligned but no valid pixels")
        return
    print(
        f"[{label}] depth: valid={valid.size} median={float(np.median(valid)):.3f} m "
        f"scale={frame.depth_scale_m}"
    )


def _opencv_gui_available() -> bool:
    if not os.environ.get("DISPLAY"):
        return False
    try:
        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imshow("__opencv_gui_probe__", probe)
        cv2.waitKey(1)
        cv2.destroyWindow("__opencv_gui_probe__")
        return True
    except cv2.error:
        return False


def _save_snapshot(frame: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame.color_bgr):
        raise RuntimeError(f"无法写入快照：{path}")
    return path


def _preview_loop(camera: OrbbecCamera, *, seconds: float) -> None:
    if not _opencv_gui_available():
        raise RuntimeError(
            "当前 OpenCV 无 GUI 支持（常见为 opencv-python-headless）。\n"
            "  预览可改用：python3 tests/hardware/test_gemini2_open.py --save-snapshot\n"
            "  或安装窗口版：pip uninstall -y opencv-python-headless && pip install opencv-python"
        )
    window = "Gemini 2 Test (Q quit)"
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            frame = camera.read()
            view = frame.color_bgr.copy()
            cv2.putText(
                view,
                "Gemini 2 OK | Q quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def run(args: argparse.Namespace) -> int:
    _require_orbbec_sdk()
    _usb_hint()

    count = _sdk_device_count()
    print(f"pyorbbecsdk query_devices: {count}")
    if count == 0:
        print(
            "SDK 看不到相机。常见原因：\n"
            "  1) USB 权限：/dev/bus/usb 节点为 root:root → 安装 udev 规则后拔插相机\n"
            "     sudo cp udev/99-orbbec.rules /etc/udev/rules.d/\n"
            "     sudo udevadm control --reload-rules && sudo udevadm trigger\n"
            "  2) 上次程序被 kill -9，SDK 未释放 → 拔插 Gemini 2\n"
            "  3) 其他进程占用相机 → 关闭 robot-arm / 其他 Orbbec 程序"
        )
        return 2

    config = load_config(args.root)
    camera_cfg = config.camera["camera"]
    calib = np.asarray(config.calibration["camera_matrix"]) if not args.skip_calib_check else None

    modes: list[tuple[str, bool]] = [("RGB", True)]
    if not args.rgb_only:
        modes.append(("RGB-D", False))

    for label, color_only in modes:
        print(f"\n--- {label} ---")
        camera = OrbbecCamera(
            width=int(camera_cfg["width"]),
            height=int(camera_cfg["height"]),
            fps=int(camera_cfg["fps"]),
            depth_width=int(camera_cfg.get("depth_width", camera_cfg["width"])),
            depth_height=int(camera_cfg.get("depth_height", 800)),
            min_depth_m=float(camera_cfg["min_depth_m"]),
            max_depth_m=float(camera_cfg["max_depth_m"]),
            calibrated_camera_matrix=calib,
            color_only=color_only,
        )
        t0 = time.time()
        try:
            camera.start()
            frame = camera.read()
        except Exception as exc:
            print(f"FAIL ({label}): {exc}")
            return 1
        finally:
            camera.close()
        print(f"open+read OK in {time.time() - t0:.2f}s")
        _print_frame_summary(label, frame)

    if args.preview or args.save_snapshot:
        camera = OrbbecCamera(
            width=int(camera_cfg["width"]),
            height=int(camera_cfg["height"]),
            fps=int(camera_cfg["fps"]),
            depth_width=int(camera_cfg.get("depth_width", camera_cfg["width"])),
            depth_height=int(camera_cfg.get("depth_height", 800)),
            min_depth_m=float(camera_cfg["min_depth_m"]),
            max_depth_m=float(camera_cfg["max_depth_m"]),
            color_only=True,
        )
        try:
            camera.start()
            if args.save_snapshot:
                out = Path(args.save_snapshot)
                if not out.is_absolute():
                    out = Path(args.root) / out
                path = _save_snapshot(camera.read(), out)
                print(f"\n已保存快照：{path}")
            if args.preview:
                print(f"\n预览 {args.preview_seconds:.0f}s（按 Q 提前退出）…")
                try:
                    _preview_loop(camera, seconds=float(args.preview_seconds))
                except RuntimeError as exc:
                    print(f"预览跳过：{exc}")
                    if not args.save_snapshot:
                        fallback = Path(args.root) / "tests/hardware/outputs/gemini2_preview.jpg"
                        path = _save_snapshot(camera.read(), fallback)
                        print(f"已改存快照：{path}")
        finally:
            camera.close()

    print("\nPASS: Gemini 2 打开并采帧成功。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="robot_arm 根目录")
    parser.add_argument("--rgb-only", action="store_true", help="只测 RGB，跳过深度对齐")
    parser.add_argument(
        "--skip-calib-check",
        action="store_true",
        help="不校验内参与 eye_in_hand 标定矩阵是否一致",
    )
    parser.add_argument("--preview", action="store_true", help="采帧后弹出 RGB 预览窗口")
    parser.add_argument(
        "--save-snapshot",
        nargs="?",
        const="tests/hardware/outputs/gemini2_preview.jpg",
        metavar="PATH",
        help="保存一帧 RGB 到文件（无 GUI 时可用；默认 tests/hardware/outputs/gemini2_preview.jpg）",
    )
    parser.add_argument("--preview-seconds", type=float, default=30.0, help="预览最长秒数")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
