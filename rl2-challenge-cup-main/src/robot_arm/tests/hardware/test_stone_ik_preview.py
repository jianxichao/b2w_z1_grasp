"""石头 YOLO + 实时 IK 可达性预览（启动时读实测关节，不发送轨迹）。

用 Gemini 2 检测石头，按手眼标定转到 ``arm_base``，并对 pregrasp/grasp 做与
``test_grasp.py`` 相同的 IK。启动时连接 CAN **只读**初始关节（见 ``joint_feedback``），
用于 FK（视觉坐标）与 IK 种子；不预设 search 位。

显式启动::

  python3 tests/hardware/test_stone_ik_preview.py --confirm
  python3 tests/hardware/test_stone_ik_preview.py --confirm --preview
"""

from __future__ import annotations

__test__ = False

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT = ROOT.parent
YOLO_DIR = ROOT / "models/yolo"
DEFAULT_WEIGHTS = YOLO_DIR / "best.pt"
DEFAULT_TARGET_CLASS = "rock_obstacle"
DEFAULT_SNAPSHOT = ROOT / "tests/hardware/outputs/stone_ik_preview.jpg"
INVERTED_SEARCH_JOINTS = (0.0, 1.2500, 4.2800, 0.8300, 0.0, 0.0)
WINDOW = "Stone IK Preview (Q quit)"


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_arm.app import build_core  # noqa: E402
from robot_arm.config import load_config  # noqa: E402
from robot_arm.hardware import Arx5Backend  # noqa: E402
from robot_arm.manipulation.kinematics import Arx5Kinematics  # noqa: E402
from robot_arm.manipulation.planner import GraspPlanner, PlanningError  # noqa: E402
from robot_arm.vision import (  # noqa: E402
    OrbbecCamera,
    YoloTracker,
    camera_point_to_base,
    deproject_pixel,
    load_eye_in_hand,
    median_bbox_depth,
)
from joint_feedback import format_joints, read_joints_from_backend  # noqa: E402


@dataclass(frozen=True)
class IkStageResult:
    ok: bool
    position_error_m: float | None
    orientation_error_rad: float | None
    limit_detail: str = ""


@dataclass(frozen=True)
class StoneIkReport:
    track_id: int
    confidence: float
    base_xyz: np.ndarray
    camera_xyz: np.ndarray
    depth_m: float
    radius_m: float
    bbox_xyxy: tuple[float, float, float, float]
    pregrasp: IkStageResult
    grasp: IkStageResult
    full_plan_ok: bool
    full_plan_error: str

    @property
    def reachable(self) -> bool:
        return self.pregrasp.ok and self.grasp.ok

    def summary_line(self) -> str:
        status = "REACHABLE" if self.reachable else "NO_IK"
        pre = "OK" if self.pregrasp.ok else "FAIL"
        gr = "OK" if self.grasp.ok else "FAIL"
        xyz = self.base_xyz
        return (
            f"#{self.track_id} conf={self.confidence:.2f} {status} "
            f"B=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) r={self.radius_m:.3f}m "
            f"pregrasp={pre} grasp={gr} plan={'OK' if self.full_plan_ok else 'FAIL'}"
            + self._limit_suffix()
        )

    def _limit_suffix(self) -> str:
        parts: list[str] = []
        for stage_name, stage in (("pregrasp", self.pregrasp), ("grasp", self.grasp)):
            if stage.ok or not stage.limit_detail:
                continue
            parts.append(f"{stage_name}[{stage.limit_detail}]")
        if self.full_plan_error and not self.full_plan_ok:
            parts.append(f"plan[{self.full_plan_error}]")
        return f" | {'; '.join(parts)}" if parts else ""


def _require_orbbec_sdk() -> None:
    if importlib.util.find_spec("pyorbbecsdk") is None:
        raise ImportError(
            "缺少 pyorbbecsdk。请执行： pip install \"pyorbbecsdk2==2.0.15\""
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


def _load_target_class(names_path: Path, override: str | None) -> str:
    if override:
        return override
    if names_path.is_file():
        names = json.loads(names_path.read_text(encoding="utf-8"))
        if names:
            return str(names[0])
    return DEFAULT_TARGET_CLASS


def _resolve_calibration(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else root / explicit
        if not path.is_file():
            raise FileNotFoundError(f"手眼标定文件不存在：{path}")
        return path
    primary = root / "config" / "eye_in_hand_calibration.json"
    if primary.is_file():
        return primary
    raise FileNotFoundError("未找到 eye_in_hand_calibration.json")


def _load_joints_file(path: Path) -> tuple[float, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "joints_rad" in data:
        values = data["joints_rad"]
    elif "joints" in data:
        values = data["joints"]
    else:
        raise ValueError(f"关节文件缺少 joints_rad：{path}")
    joints = tuple(float(v) for v in values)
    if len(joints) != 6:
        raise ValueError(f"关节文件必须包含 6 个值：{path}")
    return joints


def _resolve_fk_joints(
    args: argparse.Namespace,
    measured: tuple[float, ...] | None,
) -> tuple[float, ...]:
    if args.joints_file:
        return _load_joints_file(Path(args.joints_file))
    if args.inverted:
        return INVERTED_SEARCH_JOINTS
    if args.fk_joints:
        return tuple(float(v) for v in args.fk_joints)
    if measured is not None:
        return measured
    raise RuntimeError("需要 --confirm 读取初始关节，或指定 --joints-file / --fk-joints")


def _resolve_ik_seed(
    args: argparse.Namespace,
    fk_joints: tuple[float, ...],
) -> tuple[float, ...]:
    if args.ik_seed_file:
        return _load_joints_file(Path(args.ik_seed_file))
    if args.inverted:
        return INVERTED_SEARCH_JOINTS
    if args.ik_seed:
        return tuple(float(v) for v in args.ik_seed)
    return fk_joints


def _stage_result(
    solve_result: Any | None,
    *,
    tol_m: float,
    tol_rad: float,
    kinematics: Any | None = None,
) -> IkStageResult:
    if solve_result is None:
        detail = getattr(kinematics, "last_ik_diagnostic", "") if kinematics is not None else ""
        return IkStageResult(False, None, None, limit_detail=detail)
    ok = (
        float(solve_result.position_error_m) <= tol_m
        and float(solve_result.orientation_error_rad) <= tol_rad
    )
    return IkStageResult(
        ok,
        float(solve_result.position_error_m),
        float(solve_result.orientation_error_rad),
    )


def _evaluate_stone(
    planner: GraspPlanner,
    *,
    base_xyz: Sequence[float],
    ik_seed: Sequence[float],
    track_id: int,
    confidence: float,
    camera_xyz: Sequence[float],
    depth_m: float,
    bbox_xyxy: Sequence[float],
) -> StoneIkReport:
    kinematics = planner.kinematics
    settings = planner.settings
    xyz = np.asarray(base_xyz, dtype=float)
    pre_z = min(float(xyz[2] + settings.pregrasp_offset_m), settings.transit_height_m)
    stages: dict[str, IkStageResult] = {}
    for name, position in (
        ("pregrasp", (float(xyz[0]), float(xyz[1]), pre_z)),
        ("grasp", (float(xyz[0]), float(xyz[1]), float(xyz[2]))),
    ):
        pose = planner._pose(position)
        solved = kinematics.solve(
            pose.position,
            pose.rotation,
            planner._seeds(pose.position, ik_seed),
        )
        stages[name] = _stage_result(
            solved,
            tol_m=0.004,
            tol_rad=0.10,
            kinematics=kinematics,
        )

    full_plan_ok = False
    full_plan_error = ""
    try:
        from robot_arm.manipulation import Target

        grasp_pose = planner._pose((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        target = Target(
            target_id=f"stone#{track_id}",
            pose=grasp_pose,
            confidence=float(confidence),
            stamp_s=time.time(),
        )
        planner.plan(target, ik_seed)
        full_plan_ok = True
    except PlanningError as exc:
        full_plan_error = str(exc)

    bbox = tuple(float(v) for v in bbox_xyxy)
    return StoneIkReport(
        track_id=track_id,
        confidence=float(confidence),
        base_xyz=xyz,
        camera_xyz=np.asarray(camera_xyz, dtype=float),
        depth_m=float(depth_m),
        radius_m=float(math.hypot(xyz[0], xyz[1])),
        bbox_xyxy=bbox,
        pregrasp=stages["pregrasp"],
        grasp=stages["grasp"],
        full_plan_ok=full_plan_ok,
        full_plan_error=full_plan_error,
    )


def _stones_from_frame(
    frame: Any,
    detections: list[Any],
    *,
    target_class: str,
    min_confidence: float,
    base_T_eef: np.ndarray,
    eef_T_camera: np.ndarray,
    distortion: Sequence[float] | None,
) -> list[tuple[Any, np.ndarray, np.ndarray, float]]:
    stones: list[tuple[Any, np.ndarray, np.ndarray, float]] = []
    for detection in detections:
        if detection.class_name != target_class:
            continue
        if detection.confidence < min_confidence:
            continue
        depth = median_bbox_depth(frame.depth_m, detection.bbox_xyxy)
        if depth is None:
            continue
        camera_xyz = deproject_pixel(
            detection.center,
            depth,
            frame.camera_matrix,
            distortion,
        )
        base_xyz = camera_point_to_base(camera_xyz, base_T_eef, eef_T_camera)
        stones.append((detection, base_xyz, camera_xyz, float(depth)))
    return stones


def _draw_reports(
    image: np.ndarray,
    reports: Sequence[StoneIkReport],
    *,
    hud: str,
) -> np.ndarray:
    view = image.copy()
    for report in reports:
        x1, y1, x2, y2 = (int(v) for v in report.bbox_xyxy)
        if report.reachable:
            color = (40, 220, 80)
            label = "IK OK"
        elif report.pregrasp.ok or report.grasp.ok:
            color = (0, 200, 255)
            label = "PARTIAL"
        else:
            color = (40, 40, 240)
            label = "NO IK"
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        xyz = report.base_xyz
        lines = [
            f"#{report.track_id} {label} conf={report.confidence:.2f}",
            f"B=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) r={report.radius_m:.3f}",
            f"pre={'OK' if report.pregrasp.ok else 'X'} grasp={'OK' if report.grasp.ok else 'X'}",
        ]
        for index, text in enumerate(lines):
            cv2.putText(
                view,
                text,
                (x1, max(18, y1 - 8 - 18 * (len(lines) - 1 - index))),
                0,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    for index, line in enumerate(hud.split("\n")):
        cv2.putText(view, line, (12, 28 + 24 * index), 0, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, 28 + 24 * index), 0, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def _evaluate_frame(
    planner: GraspPlanner,
    frame: Any,
    detections: list[Any],
    *,
    target_class: str,
    min_confidence: float,
    base_T_eef: np.ndarray,
    eef_T_camera: np.ndarray,
    distortion: Sequence[float] | None,
    ik_seed: Sequence[float],
) -> list[StoneIkReport]:
    raw = _stones_from_frame(
        frame,
        detections,
        target_class=target_class,
        min_confidence=min_confidence,
        base_T_eef=base_T_eef,
        eef_T_camera=eef_T_camera,
        distortion=distortion,
    )
    reports: list[StoneIkReport] = []
    for detection, base_xyz, camera_xyz, depth_m in raw:
        report = _evaluate_stone(
            planner,
            base_xyz=base_xyz,
            ik_seed=ik_seed,
            track_id=int(detection.track_id),
            confidence=float(detection.confidence),
            camera_xyz=camera_xyz,
            depth_m=depth_m,
            bbox_xyxy=detection.bbox_xyxy,
        )
        reports.append(report)
    reports.sort(key=lambda item: item.confidence, reverse=True)
    return reports


def _print_context(
    *,
    fk_joints: Sequence[float],
    ik_seed: Sequence[float],
    kinematics: Arx5Kinematics,
) -> None:
    fk_tcp = kinematics.forward_position(fk_joints)
    seed_tcp = kinematics.forward_position(ik_seed)
    print("启动实测 FK 关节(rad):", format_joints(tuple(fk_joints)))
    print("IK 种子关节(rad):", format_joints(tuple(ik_seed)))
    print(
        "FK TCP (arm_base, m): "
        f"({fk_tcp[0]:+.3f},{fk_tcp[1]:+.3f},{fk_tcp[2]:+.3f}) "
        f"radius={math.hypot(fk_tcp[0], fk_tcp[1]):.3f}"
    )
    print(
        "IK seed TCP (arm_base, m): "
        f"({seed_tcp[0]:+.3f},{seed_tcp[1]:+.3f},{seed_tcp[2]:+.3f}) "
        f"radius={math.hypot(seed_tcp[0], seed_tcp[1]):.3f}"
    )
    print("手眼/FK 与 IK 均基于启动时读到的初始关节（不预设 search）。")


def run(args: argparse.Namespace) -> int:
    _require_orbbec_sdk()
    if not args.confirm:
        raise RuntimeError("必须加 --confirm：连接 CAN 只读启动时的实测初始关节")
    root = Path(args.root).resolve()
    config = load_config(root)
    planner, _safety = build_core(config)
    kinematics = planner.kinematics

    arm = Arx5Backend(root, config.hardware, dry_run=False, kinematics=kinematics)
    measured: tuple[float, ...] | None = None
    try:
        print("连接 CAN；只读关节反馈，不发送轨迹。")
        arm.connect()
        ok, reason = arm.self_check()
        if not ok:
            raise RuntimeError(reason)
        measured = read_joints_from_backend(arm)
        print(f"启动实测初始关节(rad): {format_joints(measured)}")
    except Exception:
        arm.close()
        raise

    fk_joints = _resolve_fk_joints(args, measured)
    ik_seed = _resolve_ik_seed(args, fk_joints)
    base_T_eef = kinematics.forward_matrix(fk_joints)

    calib_path = _resolve_calibration(root, Path(args.calibration) if args.calibration else None)
    eef_T_camera = load_eye_in_hand(calib_path)
    target_class = _load_target_class(Path(args.names), args.target_class)

    weights = Path(args.weights)
    if not weights.is_file() and not weights.is_absolute():
        alt = root / weights
        if alt.is_file():
            weights = alt
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在：{weights}")

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
    tracker = YoloTracker(
        weights=str(weights),
        confidence=float(args.confidence),
        device=str(args.device),
        tracker="bytetrack.yaml",
        image_size=int(args.imgsz),
        target_classes={target_class},
    )
    distortion = config.calibration["distortion_coefficients"]
    use_gui = bool(args.preview) and _opencv_gui_available()
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.is_absolute():
        snapshot_path = root / snapshot_path

    print(f"YOLO: {weights}")
    print(f"手眼: {calib_path}")
    print(f"目标类别: {target_class}")
    _print_context(fk_joints=fk_joints, ik_seed=ik_seed, kinematics=kinematics)
    if args.preview and not use_gui:
        print("OpenCV 无 GUI，使用终端输出 + 快照文件。")
    print("按 Q 退出。")

    last_lines: tuple[str, ...] = ()
    last_write = 0.0
    try:
        camera.start()
        if use_gui:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            frame = camera.read()
            detections = tracker.track(frame.color_bgr)
            reports = _evaluate_frame(
                planner,
                frame,
                [replace(d, detail="") for d in detections],
                target_class=target_class,
                min_confidence=float(args.confidence),
                base_T_eef=base_T_eef,
                eef_T_camera=eef_T_camera,
                distortion=distortion,
                ik_seed=ik_seed,
            )
            lines = tuple(report.summary_line() for report in reports)
            if not lines:
                lines = ("stones=0 | 无有效深度/类别匹配",)
            hud = "\n".join(
                (
                    f"stones={len(reports)} | class={target_class} | conf>={args.confidence:.2f}",
                    "绿=pre+grasp IK OK | 黄=部分 | 红=无解",
                    *lines[:3],
                )
            )
            now = time.time()
            if lines != last_lines or now - last_write >= float(args.print_interval):
                for line in lines:
                    print(line, flush=True)
                last_lines = lines
                last_write = now
                display = _draw_reports(frame.color_bgr, reports, hud=hud)
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snapshot_path), display)

            if use_gui:
                display = _draw_reports(frame.color_bgr, reports, hud=hud)
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        camera.close()
        tracker.close()
        arm.close()
        if use_gui:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
    print(f"快照: {snapshot_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="robot_arm 根目录")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO 权重")
    parser.add_argument("--names", default=str(YOLO_DIR / "best.names.json"), help="类别 JSON")
    parser.add_argument("--target-class", default="", help="覆盖检测类别")
    parser.add_argument("--calibration", default="", help="手眼 JSON 路径")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO 置信度阈值")
    parser.add_argument("--device", default="auto", help="YOLO 设备")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--confirm", action="store_true", help="连接 CAN 并读取启动时实测关节")
    parser.add_argument(
        "--inverted",
        action="store_true",
        help="覆盖：FK/IK 使用倒置候选关节 (0, 1.25, 4.28, 0.83, 0, 0)",
    )
    parser.add_argument(
        "--joints-file",
        default="",
        help="覆盖：教点 JSON（joints_rad）作为 FK/IK 假设",
    )
    parser.add_argument(
        "--ik-seed-file",
        default="",
        help="覆盖：单独指定 IK 种子 JSON",
    )
    parser.add_argument(
        "--fk-joints",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="6 个关节角(rad) 作为 FK 假设姿态",
    )
    parser.add_argument(
        "--ik-seed",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="6 个关节角(rad) 作为 IK 迭代种子",
    )
    parser.add_argument("--preview", action="store_true", help="OpenCV 预览窗口")
    parser.add_argument(
        "--snapshot",
        default=str(DEFAULT_SNAPSHOT.relative_to(ROOT)),
        help="带标注快照路径",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="终端重复打印最小间隔(秒)",
    )
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
