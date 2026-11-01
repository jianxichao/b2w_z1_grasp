"""用 Gemini 2 对齐深度验证桌面在 arm_base 中的高度。

棋盘只用于在 RGB 图像中找到同一组像素。深度点来自同一帧的
RGB 对齐深度图，然后按“彩色内参 → eef_T_camera → base_T_eef”转换。
本程序只保持机械臂阻尼，不发送运动轨迹。
"""

from __future__ import annotations

__test__ = False

import argparse
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
import time
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_arm.config import load_config  # noqa: E402
from robot_arm.hardware import Arx5Backend  # noqa: E402
from robot_arm.vision import (  # noqa: E402
    OrbbecCamera,
    camera_point_to_base,
    deproject_pixel,
    load_eye_in_hand,
)
from test_eye_in_hand_calibration import (  # noqa: E402
    PATTERN_SIZE,
    SQUARE_SIZE_M,
    detect_board,
)


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _sample_depth(
    depth_m: np.ndarray, pixel: Sequence[float], radius: int
) -> tuple[float, int, tuple[int, int, int, int]] | None:
    """在一个 RGB 像素的小邻域取深度中值，避免刚好落在深度空洞上。"""
    u, v = (int(round(float(value))) for value in pixel)
    height, width = depth_m.shape
    left, right = max(0, u - radius), min(width, u + radius + 1)
    top, bottom = max(0, v - radius), min(height, v + radius + 1)
    if left >= right or top >= bottom:
        return None
    values = depth_m[top:bottom, left:right]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.median(values)), int(values.size), (left, top, right, bottom)


def _depth_board_measurement(
    frame: object,
    corners: np.ndarray,
    base_T_eef: np.ndarray,
    eef_T_camera: np.ndarray,
    radius: int,
) -> dict[str, object] | None:
    """把棋盘所有内角点的对齐深度转换到 arm_base。"""
    depth_m = frame.depth_m
    distortion = frame.distortion_coefficients
    if depth_m.ndim != 2 or depth_m.size == 0:
        return None
    camera_points: list[np.ndarray] = []
    base_points: list[np.ndarray] = []
    depths: list[float] = []
    valid_pixels = []
    rois = []
    for corner in corners.reshape(-1, 2):
        sampled = _sample_depth(depth_m, corner, radius)
        if sampled is None:
            continue
        depth, pixel_count, roi = sampled
        camera = deproject_pixel(corner, depth, frame.camera_matrix, distortion)
        base = camera_point_to_base(camera, base_T_eef, eef_T_camera)
        depths.append(depth)
        valid_pixels.append(pixel_count)
        rois.append(roi)
        camera_points.append(camera)
        base_points.append(base)
    if not base_points:
        return None
    camera_values = np.stack(camera_points)
    base_values = np.stack(base_points)
    # 第一个内角点仍是标定约定的参考点；其余角点只用于检验整块平面。
    origin = base_values[0]
    return {
        "valid_corner_count": int(len(base_values)),
        "origin_depth_m": float(depths[0]),
        "origin_camera_xyz_m": camera_values[0].tolist(),
        "origin_arm_base_xyz_m": origin.tolist(),
        "plane_depth_median_m": float(np.median(depths)),
        "plane_arm_base_z_m": float(np.median(base_values[:, 2])),
        "plane_arm_base_z_spread_m": float(np.max(base_values[:, 2]) - np.min(base_values[:, 2])),
        "valid_depth_pixels_per_corner": valid_pixels,
        "first_depth_roi_xyxy": rois[0],
    }


def _draw(
    image: np.ndarray,
    observation: object | None,
    depth_measurement: dict[str, object] | None,
    count: int,
    target: int,
) -> np.ndarray:
    view = image.copy()
    if observation is not None:
        cv2.drawChessboardCorners(view, PATTERN_SIZE, observation.corners, True)
        origin = tuple(np.rint(observation.corners[0, 0]).astype(int))
        cv2.circle(view, origin, 9, (0, 0, 255), -1)
        cv2.putText(view, "DEPTH ORIGIN", (origin[0] + 12, origin[1]), 0, 0.65, (0, 0, 255), 2)
    lines = [
        f"Samples: {count}/{target}",
        "Board: FOUND" if observation is not None else "Board: NOT FOUND",
        "Depth: READY" if depth_measurement is not None else "Depth: --",
        "SPACE save | BACKSPACE undo | ENTER finish | Q quit",
    ]
    if depth_measurement is not None:
        lines.append(f"depth={float(depth_measurement['origin_depth_m']):.3f}m")
        lines.append(f"base_z={float(depth_measurement['origin_arm_base_xyz_m'][2]):+.3f}m")
    for index, text in enumerate(lines):
        cv2.putText(view, text, (15, 30 + 28 * index), 0, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
    return view


def run(args: argparse.Namespace) -> int:
    config = load_config(args.root)
    if importlib.util.find_spec("pyorbbecsdk") is None:
        raise RuntimeError(
            "当前 Python 没有 pyorbbecsdk。请在 piper 环境运行，不要使用系统 Python。"
        )
    interface = str(config.hardware["can_interface"])
    try:
        socket.if_nametoindex(interface)
    except OSError as exc:
        raise RuntimeError(f"CAN 接口不存在或未启动：{interface}") from exc

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
    arm = Arx5Backend(config.root, config.hardware)
    eef_T_camera = load_eye_in_hand(config.resolve("config/eye_in_hand_calibration.json"))
    samples: list[dict[str, object]] = []
    output = Path(args.output)
    if not output.is_absolute():
        output = config.root / output

    try:
        print(f"连接 {interface}：只保持阻尼/重力补偿，不发送运动轨迹。")
        arm.connect()
        ok, reason = arm.self_check()
        if not ok:
            raise RuntimeError(reason or "机械臂自检失败")
        camera.start()
        print("棋盘底面平放在桌面；机械臂停稳后按空格。现在读取 RGB-D 对齐深度。")
        print(f"目标 {args.samples} 组；至少 {args.min_samples} 组后按 Enter 计算。")
        while len(samples) < args.samples:
            frame = camera.read()
            observation = detect_board(
                frame.color_bgr,
                frame.camera_matrix,
                frame.distortion_coefficients,
            ) if frame.distortion_coefficients is not None else None
            current = arm.base_T_eef()
            measurement = (
                _depth_board_measurement(frame, observation.corners, current, eef_T_camera, args.radius)
                if observation is not None
                else None
            )
            cv2.imshow("X5 Gemini2 RGB-D Table Verification", _draw(frame.color_bgr, observation, measurement, len(samples), args.samples))
            key = cv2.waitKeyEx(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return 1
            if key in (8, 127):
                if samples:
                    samples.pop()
                    print(f"已删除上一组，当前 {len(samples)}/{args.samples}")
                continue
            if key == 13:
                if len(samples) < args.min_samples:
                    print(f"当前 {len(samples)} 组，至少 {args.min_samples} 组才能计算")
                    continue
                break
            if key != 32:
                continue
            if observation is None or measurement is None:
                print("棋盘或对齐深度无效，未采样")
                continue
            if observation.reprojection_error_px > args.max_reprojection_error:
                print(f"PnP 重投影误差 {observation.reprojection_error_px:.3f}px 过大，未采样")
                continue
            before = arm.base_T_eef()
            time.sleep(0.12)
            settled = arm.base_T_eef()
            moved_m = float(np.linalg.norm(settled[:3, 3] - before[:3, 3]))
            moved_deg = math.degrees(_rotation_angle(before[:3, :3].T @ settled[:3, :3]))
            if moved_m > 0.0015 or moved_deg > 0.5:
                print("机械臂仍在移动，请停稳后重新采样")
                continue
            # 重新读取一帧，使图像、深度和停稳后的末端反馈尽量对应同一时刻。
            frame = camera.read()
            observation = detect_board(frame.color_bgr, frame.camera_matrix, frame.distortion_coefficients)
            if observation is None or observation.reprojection_error_px > args.max_reprojection_error:
                print("重新取样时棋盘无效，未采样")
                continue
            measurement = _depth_board_measurement(frame, observation.corners, settled, eef_T_camera, args.radius)
            if measurement is None or int(measurement["valid_corner_count"]) < args.min_valid_corners:
                print("重新取样时有效深度角点不足，未采样")
                continue
            pnp_origin = camera_point_to_base(observation.camera_T_board[:3, 3], settled, eef_T_camera)
            measurement["pnp_origin_arm_base_xyz_m"] = pnp_origin.tolist()
            measurement["pnp_reprojection_error_px"] = float(observation.reprojection_error_px)
            measurement["image_shape"] = list(frame.color_bgr.shape[:2])
            measurement["depth_shape"] = list(frame.depth_m.shape)
            measurement["depth_scale_m"] = float(frame.depth_scale_m)
            measurement["depth_raw_origin"] = int(round(float(measurement["origin_depth_m"]) / frame.depth_scale_m))
            samples.append(measurement)
            print(
                f"采样成功 {len(samples)}/{args.samples}："
                f"raw≈{measurement['depth_raw_origin']}，depth={float(measurement['origin_depth_m']):.3f}m，"
                f"C={np.round(measurement['origin_camera_xyz_m'], 4)}，"
                f"B={np.round(measurement['origin_arm_base_xyz_m'], 4)}"
            )
        if len(samples) < args.min_samples:
            raise RuntimeError(f"有效样本不足：{len(samples)}，至少需要 {args.min_samples} 组")
        depth_origins = np.asarray([item["origin_arm_base_xyz_m"] for item in samples], dtype=float)
        plane_z = np.asarray([item["plane_arm_base_z_m"] for item in samples], dtype=float)
        pnp_origins = np.asarray([item["pnp_origin_arm_base_xyz_m"] for item in samples], dtype=float)
        table_z = float(np.mean(depth_origins[:, 2]) - args.board_thickness_m)
        payload = {
            "schema_version": 2,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "frame_id": "arm_base",
            "reference": "棋盘第一个内角点；深度来自与 RGB 对齐的同一帧",
            "board": {"squares": [10, 7], "inner_corners": list(PATTERN_SIZE), "square_size_m": SQUARE_SIZE_M},
            "board_thickness_m": args.board_thickness_m,
            "table_z_m": table_z,
            "depth_plane_z_m": float(np.mean(plane_z) - args.board_thickness_m),
            "pnp_table_z_m_for_comparison": float(np.mean(pnp_origins[:, 2]) - args.board_thickness_m),
            "depth_origin_rms_m": float(np.sqrt(np.mean(np.sum((depth_origins - depth_origins.mean(0)) ** 2, axis=1)))),
            "depth_origin_xyz_m": depth_origins.mean(0).tolist(),
            "sample_count": len(samples),
            "samples": samples,
            "interpretation": {
                "expected_10cm_object_center_z_m": table_z + 0.10,
                "depth_and_pnp_z_difference_m": float(np.mean(depth_origins[:, 2] - pnp_origins[:, 2])),
                "hardware_verification_required": True,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"测量完成：{output}")
        print(f"深度桌面 z = {table_z:.4f} m；深度平面 z = {payload['depth_plane_z_m']:.4f} m")
        print(f"PnP 对照 z = {payload['pnp_table_z_m_for_comparison']:.4f} m")
        print(f"深度与 PnP 的平均 z 差 = {payload['interpretation']['depth_and_pnp_z_difference_m']:+.4f} m")
        print(f"若目标中心高出桌面 10 cm，预期 arm_base.z ≈ {table_z + 0.10:.4f} m")
        return 0
    finally:
        cv2.destroyAllWindows()
        camera.close()
        arm.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--confirm", action="store_true", help="确认连接 can1；不会发送运动轨迹")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-valid-corners", type=int, default=20)
    parser.add_argument("--radius", type=int, default=2, help="每个棋盘角点取深度的邻域半径")
    parser.add_argument("--max-reprojection-error", type=float, default=1.5)
    parser.add_argument("--board-thickness-m", type=float, default=0.0)
    parser.add_argument("--output", default="tests/hardware/outputs/table_height_result.json")
    args = parser.parse_args(argv)
    if not args.confirm:
        parser.error("本程序会连接 can1；确认安全后添加 --confirm")
    if args.samples < args.min_samples or args.min_samples < 1 or args.min_valid_corners < 1:
        parser.error("samples、min-samples 和 min-valid-corners 参数无效")
    if args.radius < 0 or args.max_reprojection_error <= 0.0 or args.board_thickness_m < 0.0:
        parser.error("radius、误差阈值和棋盘厚度参数无效")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
