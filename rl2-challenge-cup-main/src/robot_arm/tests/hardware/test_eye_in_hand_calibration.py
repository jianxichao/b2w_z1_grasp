"""ARX X5 与 Orbbec Gemini 2 多姿态眼在手上标定。

普通 pytest 不会连接硬件；请按 docs/x5_eye_in_hand_calibration.md 显式启动。
"""

from __future__ import annotations

__test__ = False

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_arm.config import load_config  # noqa: E402
from robot_arm.hardware import Arx5Backend  # noqa: E402
from robot_arm.vision import OrbbecCamera  # noqa: E402


PATTERN_SIZE = (13, 9)
SQUARE_SIZE_M = 0.03
TARGET_SAMPLES = 50
MIN_SAMPLES = 15


@dataclass(frozen=True)
class Sample:
    base_T_eef: np.ndarray
    camera_T_board: np.ndarray
    joints_rad: tuple[float, ...]
    reprojection_error_px: float


@dataclass(frozen=True)
class BoardObservation:
    camera_T_board: np.ndarray
    corners: np.ndarray
    reprojection_error_px: float


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _matrix_to_rpy(rotation: np.ndarray) -> list[float]:
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return [roll, pitch, yaw]


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def _valid_rotation(rotation: np.ndarray) -> bool:
    return bool(
        rotation.shape == (3, 3)
        and np.all(np.isfinite(rotation))
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    )


def solve_hand_eye(samples: Sequence[Sample]) -> tuple[str, np.ndarray, dict[str, Any]]:
    """求解 eef_T_camera，并用固定棋盘在基座下的一致性评分。"""
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"至少需要 {MIN_SAMPLES} 组样本")
    methods = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    gripper_r = [sample.base_T_eef[:3, :3] for sample in samples]
    gripper_t = [sample.base_T_eef[:3, 3] for sample in samples]
    target_r = [sample.camera_T_board[:3, :3] for sample in samples]
    target_t = [sample.camera_T_board[:3, 3] for sample in samples]
    results: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                gripper_r, gripper_t, target_r, target_t, method=method
            )
        except cv2.error:
            continue
        rotation = np.asarray(rotation, dtype=float)
        if not _valid_rotation(rotation) or not np.all(np.isfinite(translation)):
            continue
        eef_T_camera = _transform(rotation, translation)
        base_T_boards = [
            sample.base_T_eef @ eef_T_camera @ sample.camera_T_board
            for sample in samples
        ]
        mean_t = np.mean([value[:3, 3] for value in base_T_boards], axis=0)
        mean_r = _mean_rotation([value[:3, :3] for value in base_T_boards])
        translation_errors = np.array(
            [np.linalg.norm(value[:3, 3] - mean_t) for value in base_T_boards]
        )
        rotation_errors = np.array(
            [_rotation_angle(mean_r.T @ value[:3, :3]) for value in base_T_boards]
        )
        translation_rms = float(np.sqrt(np.mean(translation_errors**2)))
        rotation_rms = float(np.sqrt(np.mean(rotation_errors**2)))
        metrics = {
            "board_translation_rms_m": translation_rms,
            "board_translation_max_m": float(np.max(translation_errors)),
            "board_rotation_rms_deg": math.degrees(rotation_rms),
            "board_rotation_max_deg": math.degrees(float(np.max(rotation_errors))),
            "score": translation_rms + 0.05 * rotation_rms,
            "translation": eef_T_camera[:3, 3].tolist(),
            "rotation_matrix": eef_T_camera[:3, :3].tolist(),
        }
        results[name] = eef_T_camera, metrics

    if not results:
        raise RuntimeError("所有手眼标定方法都求解失败，请增加姿态变化后重试")
    selected = min(results, key=lambda name: results[name][1]["score"])
    selected_transform = results[selected][0]
    for transform, metrics in results.values():
        metrics["translation_to_selected_m"] = float(
            np.linalg.norm(transform[:3, 3] - selected_transform[:3, 3])
        )
        metrics["rotation_to_selected_deg"] = math.degrees(
            _rotation_angle(selected_transform[:3, :3].T @ transform[:3, :3])
        )
    return selected, selected_transform, {
        name: metrics for name, (_, metrics) in results.items()
    }


def detect_board(
    image: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray
) -> BoardObservation | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    object_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    object_points[:, :2] = (
        np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
        * SQUARE_SIZE_M
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=2))))
    rotation, _ = cv2.Rodrigues(rvec)
    return BoardObservation(_transform(rotation, tvec), corners, error)


def _is_duplicate(
    transform: np.ndarray,
    samples: Sequence[Sample],
    min_translation_m: float,
    min_rotation_deg: float,
) -> bool:
    for sample in samples:
        translation = np.linalg.norm(transform[:3, 3] - sample.base_T_eef[:3, 3])
        rotation = math.degrees(
            _rotation_angle(sample.base_T_eef[:3, :3].T @ transform[:3, :3])
        )
        if translation < min_translation_m and rotation < min_rotation_deg:
            return True
    return False


def _draw(
    image: np.ndarray,
    observation: BoardObservation | None,
    base_T_eef: np.ndarray,
    sample_count: int,
    target_count: int,
) -> np.ndarray:
    view = image.copy()
    if observation is not None:
        cv2.drawChessboardCorners(view, PATTERN_SIZE, observation.corners, True)
        origin = tuple(np.rint(observation.corners[0, 0]).astype(int))
        cv2.circle(view, origin, 9, (0, 0, 255), -1)
        cv2.putText(view, "ORIGIN", (origin[0] + 12, origin[1]), 0, 0.7, (0, 0, 255), 2)
    lines = [
        f"Samples: {sample_count}/{target_count}",
        "Board: FOUND" if observation else "Board: NOT FOUND",
        (
            f"PnP error: {observation.reprojection_error_px:.3f} px"
            if observation
            else "PnP error: --"
        ),
        "SPACE save | BACKSPACE undo | ENTER solve | Q quit",
        "Keep the red ORIGIN on the same marked physical corner",
        "EEF xyz: " + " ".join(f"{value:+.3f}" for value in base_T_eef[:3, 3]),
    ]
    for index, text in enumerate(lines):
        y = 30 + 30 * index
        cv2.putText(view, text, (15, y), 0, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return view


def _result_payload(
    samples: Sequence[Sample],
    method: str,
    eef_T_camera: np.ndarray,
    method_metrics: dict[str, Any],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[str, Any]:
    pnp_errors = [sample.reprojection_error_px for sample in samples]
    selected_quality = method_metrics[method]
    return {
        "schema_version": 1,
        "model": "X5",
        "calibration_type": "eye-in-hand",
        "camera": "Orbbec Gemini 2",
        "transform_direction": "camera_to_eef",
        "transform_note": "矩阵表示 eef_T_camera，用于把相机坐标转换到末端坐标。",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": method,
        "camera_to_eef": {
            "translation": eef_T_camera[:3, 3].tolist(),
            "rotation_matrix": eef_T_camera[:3, :3].tolist(),
            "pose_6d": eef_T_camera[:3, 3].tolist()
            + _matrix_to_rpy(eef_T_camera[:3, :3]),
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "num_samples": len(samples),
        "board": {
            "squares": [10, 7],
            "inner_corners": list(PATTERN_SIZE),
            "square_size_m": SQUARE_SIZE_M,
            "origin": "OpenCV 检测的第一个内角点（红点），不是棋盘外框角点。",
        },
        "quality": {
            "selected_method": method,
            "pnp_reprojection_mean_px": float(np.mean(pnp_errors)),
            "pnp_reprojection_max_px": float(np.max(pnp_errors)),
            "recommended": bool(
                selected_quality["board_translation_rms_m"] <= 0.015
                and selected_quality["board_rotation_rms_deg"] <= 3.0
            ),
            "methods": method_metrics,
        },
        "samples": [
            {
                "joints_rad": list(sample.joints_rad),
                "base_T_eef": sample.base_T_eef.tolist(),
                "camera_T_board": sample.camera_T_board.tolist(),
                "reprojection_error_px": sample.reprojection_error_px,
            }
            for sample in samples
        ],
        "hardware_verification_required": True,
    }


def run(args: argparse.Namespace) -> int:
    config = load_config(args.root)
    camera_cfg = config.camera["camera"]
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
    arm = Arx5Backend(config.root, config.hardware)
    samples: list[Sample] = []
    output = Path(args.output)
    if not output.is_absolute():
        output = config.root / output

    try:
        print("连接 can1：SDK 只保持阻尼/重力补偿，本程序不发送目标姿态或轨迹。")
        arm.connect()
        ok, reason = arm.self_check()
        if not ok:
            raise RuntimeError(reason)
        camera.start()
        print("棋盘保持不动；手动调整机械臂，完全停稳后按空格。目标 50 组。")
        camera_matrix = distortion = None
        while True:
            frame = camera.read()
            if frame.distortion_coefficients is None:
                raise RuntimeError("当前 pyorbbecsdk 未返回彩色相机畸变参数")
            camera_matrix = frame.camera_matrix
            distortion = frame.distortion_coefficients
            observation = detect_board(frame.color_bgr, camera_matrix, distortion)
            base_T_eef = arm.base_T_eef()
            joints = arm.current_joints()
            cv2.imshow(
                "X5 Gemini2 Eye-in-Hand Calibration",
                _draw(frame.color_bgr, observation, base_T_eef, len(samples), args.samples),
            )
            key = cv2.waitKeyEx(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return 1
            if key in (8, 127):
                if samples:
                    samples.pop()
                    print(f"已删除上一组，当前 {len(samples)}/{args.samples}")
                continue
            if key in (10, 13):
                if len(samples) < MIN_SAMPLES:
                    print(f"当前 {len(samples)} 组，至少 {MIN_SAMPLES} 组才能计算")
                    continue
                break
            if key != 32:
                continue
            if observation is None:
                print("未检测到完整 9x6 内角点，未采样")
                continue
            if observation.reprojection_error_px > args.max_reprojection_error:
                print(f"PnP 重投影误差 {observation.reprojection_error_px:.3f}px 过大，未采样")
                continue
            time.sleep(0.12)
            settled = arm.base_T_eef()
            moved_m = float(np.linalg.norm(settled[:3, 3] - base_T_eef[:3, 3]))
            moved_deg = math.degrees(
                _rotation_angle(base_T_eef[:3, :3].T @ settled[:3, :3])
            )
            if moved_m > 0.0015 or moved_deg > 0.5:
                print("机械臂仍在移动，请停稳后重新采样")
                continue
            if _is_duplicate(
                base_T_eef,
                samples,
                args.min_translation,
                args.min_rotation,
            ):
                print("姿态与已有样本太接近，请改变位置或朝向")
                continue
            samples.append(
                Sample(base_T_eef.copy(), observation.camera_T_board.copy(), tuple(joints), observation.reprojection_error_px)
            )
            print(f"采样成功：{len(samples)}/{args.samples}")
            if len(samples) >= args.samples:
                break

        assert camera_matrix is not None and distortion is not None
        method, eef_T_camera, metrics = solve_hand_eye(samples)
        payload = _result_payload(
            samples, method, eef_T_camera, metrics, camera_matrix, distortion
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        quality = payload["quality"]
        selected = quality["methods"][method]
        print(f"标定完成：{output}")
        print("eef_T_camera:\n", np.array2string(eef_T_camera, precision=8))
        print(
            f"方法 {method}，棋盘一致性 RMS: "
            f"{selected['board_translation_rms_m'] * 1000:.2f} mm, "
            f"{selected['board_rotation_rms_deg']:.2f} deg，"
            f"建议使用：{'YES' if quality['recommended'] else 'NO'}"
        )
        return 0
    finally:
        cv2.destroyAllWindows()
        camera.close()
        arm.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="项目根目录")
    parser.add_argument("--confirm", action="store_true", help="确认连接 CAN 并使用阻尼/重力补偿")
    parser.add_argument("--samples", type=int, default=TARGET_SAMPLES, help="目标采样数，默认 50")
    parser.add_argument(
        "--output",
        default="tests/hardware/outputs/eye_in_hand_calibration_result.json",
        help="结果 JSON（相对于项目根目录）",
    )
    parser.add_argument("--max-reprojection-error", type=float, default=1.0, help="单组最大 PnP 误差/px")
    parser.add_argument("--min-translation", type=float, default=0.015, help="重复姿态平移阈值/m")
    parser.add_argument("--min-rotation", type=float, default=8.0, help="重复姿态旋转阈值/deg")
    args = parser.parse_args(argv)
    if not args.confirm:
        parser.error("本程序会连接 can1；确认机械臂周围安全后添加 --confirm")
    if args.samples < MIN_SAMPLES:
        parser.error(f"--samples 不能小于 {MIN_SAMPLES}")
    if args.max_reprojection_error <= 0 or args.min_translation < 0 or args.min_rotation < 0:
        parser.error("误差和姿态阈值必须为正数")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"标定失败：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
