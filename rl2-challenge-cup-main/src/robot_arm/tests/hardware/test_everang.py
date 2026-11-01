"""石头 YOLO 识别 + 三方位扫描抓取 + 示教放置（Gemini 2 + ARX X5）。

每轮流程：回零 → J1 以 0°、+90°、-90° 为中心，在各中心的 ±20° 范围内
每 10° 停一次；每个 J1 扫描点下，J2 依次俯视 0°、7°、14°、21°、28°、35°，
每次 J2 小步运动 0.1 秒；
只要检测到完整 IK/安全规划可执行的石头，立即抓取。三个方向均无有效石头时回零。
每轮结束后，输入 ``a`` 开始下一轮，输入 ``q`` 退出。

示教文件默认 ``tests/hardware/outputs/teach_joint_pose.json``，可用 ``--teach-file`` 指定。

显式启动（不会由 pytest 自动运行）::

  python3 tests/hardware/test_everang.py --confirm
"""

from __future__ import annotations

__test__ = False

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT = ROOT.parent
YOLO_DIR = ROOT / "models/yolo"
DEFAULT_WEIGHTS = YOLO_DIR / "best.pt"
DEFAULT_CALIB = ROOT / "config/eye_in_hand_calibration.json"
DEFAULT_TARGET_CLASS = "rock_obstacle"
WINDOW = "Stone Grasp (auto)"
PREVIEW_SNAPSHOT = ROOT / "tests/hardware/outputs/grasp_preview.jpg"
DEFAULT_TEACH = ROOT / "tests/hardware/outputs/teach_joint_pose.json"
J2_LOOK_DOWN_DELTA_RAD = math.radians(10.0)
SCAN_DWELL_S = 0.1
SCAN_LOOK_DOWN_MAX_DEG = 35.0
SCAN_J1_CENTERS_DEG = (0.0, 90.0, -90.0)
SCAN_J1_OFFSETS_DEG = (-20.0, -10.0, 0.0, 10.0, 20.0)
SCAN_J2_ANGLES_DEG = (0.0, 7.0, 14.0, 21.0, 28.0, 35.0)
SCAN_J2_STEP_DURATION_S = 0.5
# Kept only for the legacy helper below.
J1_INITIAL_YAW_DELTA_RAD = 0.0
PUT_PAUSE_S = 3.0
GRASP_PITCH_CANDIDATES_DEG = (90.0, 75.0, 60.0, 45.0, 30.0, 15.0, 0.0)


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_feedback import format_joints, read_joints_from_backend  # noqa: E402

from robot_arm.app import build_core  # noqa: E402
from robot_arm.config import load_config  # noqa: E402
from robot_arm.hardware import Arx5Backend  # noqa: E402
from robot_arm.manipulation import MotionCommand, Pose, Target  # noqa: E402
from robot_arm.manipulation.kinematics import rpy_matrix  # noqa: E402
from robot_arm.manipulation.planner import GraspPlanner, PlanningError  # noqa: E402
from robot_arm.manipulation.state_machine import (  # noqa: E402
    AttemptError,
    GraspStateMachine,
    State,
)
from robot_arm.manipulation.types import HardwareFault, PickPlan  # noqa: E402
from robot_arm.manipulation.safety import SafetyError  # noqa: E402
from robot_arm.vision import (  # noqa: E402
    OrbbecCamera,
    YoloTracker,
    camera_point_to_base,
    deproject_pixel,
    load_eye_in_hand,
    median_bbox_depth,
)
from robot_arm.vision.target_provider import TrackedTarget  # noqa: E402


def _safe_cv2_close() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


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


def _stdin_command_queue() -> queue.Queue[str]:
    cmd_q: queue.Queue[str] = queue.Queue()

    def _reader() -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except EOFError:
                break
            if not line:
                break
            cmd_q.put(line.strip().lower())

    threading.Thread(target=_reader, daemon=True).start()
    return cmd_q


def _collect_stones(
    camera: OrbbecCamera,
    tracker: YoloTracker,
    arm: Arx5Backend,
    eef_T_camera: np.ndarray,
    *,
    target_class: str,
    min_confidence: float,
    grasp_depth_offset_m: float,
    distortion: Sequence[float] | None,
) -> tuple[Any, list[Any], list[TrackedTarget], TrackedTarget | None]:
    frame = camera.read()
    detections = tracker.track(frame.color_bgr)
    base_T_eef = arm.base_T_eef()
    stones = _stones_from_frame(
        frame,
        detections,
        target_class=target_class,
        min_confidence=min_confidence,
        grasp_depth_offset_m=grasp_depth_offset_m,
        base_T_eef=base_T_eef,
        eef_T_camera=eef_T_camera,
        distortion=distortion,
    )
    best = _pick_best(stones)
    return frame, detections, stones, best


def _require_orbbec_sdk() -> None:
    import importlib.util

    if importlib.util.find_spec("pyorbbecsdk") is None:
        raise ImportError(
            "当前 Python 缺少 pyorbbecsdk（Gemini 2 相机驱动）。请执行：\n"
            '  python3 -m pip install --user --no-deps "pyorbbecsdk2==2.1.1"\n'
            "  python3 -c \"import pyorbbecsdk; print(pyorbbecsdk.__file__)\"\n"
            "不要使用指向旧 pypi.python.org 的 pip 命令。"
        )


def _load_target_class(names_path: Path, override: str | None) -> str:
    if override:
        return override
    if names_path.is_file():
        names = json.loads(names_path.read_text(encoding="utf-8"))
        if names:
            return str(names[0])
    return DEFAULT_TARGET_CLASS


def _resolve_calibration(config_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else config_root / explicit
        if not path.is_file():
            raise FileNotFoundError(f"手眼标定文件不存在：{path}")
        return path
    primary = config_root / "config" / "eye_in_hand_calibration.json"
    if primary.is_file():
        return primary
    if DEFAULT_CALIB.is_file():
        return DEFAULT_CALIB
    raise FileNotFoundError("未找到 eye_in_hand_calibration.json")


def _stones_from_frame(
    frame: Any,
    detections: list[Any],
    *,
    target_class: str,
    min_confidence: float,
    grasp_depth_offset_m: float,
    base_T_eef: np.ndarray,
    eef_T_camera: np.ndarray,
    distortion: Sequence[float] | None,
) -> list[TrackedTarget]:
    stones: list[TrackedTarget] = []
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
        # RGB-D measures the visible surface. Move the grasp point farther along
        # the camera ray so the fingers close around the stone's body, not its tip.
        ray_norm = float(np.linalg.norm(camera_xyz))
        if ray_norm > 1e-9:
            camera_xyz = np.asarray(camera_xyz, dtype=float) + (
                np.asarray(camera_xyz, dtype=float)
                / ray_norm
                * grasp_depth_offset_m
            )
        base_xyz = camera_point_to_base(camera_xyz, base_T_eef, eef_T_camera)
        stones.append(
            TrackedTarget(
                detection.track_id,
                detection.class_name,
                detection.confidence,
                detection.bbox_xyxy,
                camera_xyz,
                base_xyz,
                frame.timestamp,
            )
        )
    return stones


def _pick_best(stones: Sequence[TrackedTarget]) -> TrackedTarget | None:
    if not stones:
        return None
    return max(stones, key=lambda item: item.confidence)


@dataclass(frozen=True)
class IkStageResult:
    ok: bool
    position_error_m: float | None
    orientation_error_rad: float | None
    limit_detail: str = ""


@dataclass(frozen=True)
class StoneIkReport:
    stone: TrackedTarget
    pregrasp: IkStageResult
    grasp: IkStageResult
    safety_ok: bool
    safety_error: str
    full_plan_ok: bool
    full_plan_error: str

    @property
    def reachable(self) -> bool:
        return self.pregrasp.ok and self.grasp.ok and self.safety_ok

    def summary_line(self) -> str:
        if self.reachable:
            status = "REACHABLE"
        elif self.pregrasp.ok and self.grasp.ok:
            status = "NO_SAFE"
        else:
            status = "NO_IK"
        pre = "OK" if self.pregrasp.ok else "FAIL"
        gr = "OK" if self.grasp.ok else "FAIL"
        safe = "OK" if self.safety_ok else "FAIL"
        stone = self.stone
        xyz = stone.base_xyz
        radius = float(math.hypot(xyz[0], xyz[1]))
        return (
            f"#{stone.track_id} conf={stone.confidence:.2f} {status} "
            f"B=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) r={radius:.3f}m "
            f"pregrasp={pre} grasp={gr} safe={safe}"
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
        if self.safety_error and not self.safety_ok:
            parts.append(f"safe[{self.safety_error}]")
        return f" | {'; '.join(parts)}" if parts else ""


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


def _validate_pick_plan(
    safety: Any,
    start_joints: Sequence[float],
    plan: Any,
) -> None:
    commands = plan.commands
    safety.validate_path(start_joints, commands[0])
    for index in range(len(commands) - 1):
        safety.validate_path(commands[index].joints, commands[index + 1])


class FlexiblePitchGraspPlanner(GraspPlanner):
    """Search one safe grasp pitch in [0, 90] degrees for the whole pick path."""

    def __init__(self, base: GraspPlanner, safety: Any) -> None:
        super().__init__(base.kinematics, base.settings)
        self.safety = safety
        self.last_grasp_pitch_deg: float | None = None

    def _pitch_pose(self, position: Sequence[float], pitch_rad: float) -> Pose:
        xyz = tuple(float(value) for value in position)
        heading = math.atan2(xyz[1], xyz[0])
        rotation = rpy_matrix((0.0, pitch_rad, heading))
        return Pose(
            xyz,  # type: ignore[arg-type]
            tuple(tuple(float(value) for value in row) for row in rotation),  # type: ignore[arg-type]
            self.settings.frame_id,
        )

    def _candidate_pitches(self, grasp_position: Sequence[float]) -> tuple[float, ...]:
        nominal = self.settings.pose_policy.rotation(grasp_position)
        nominal_pitch = math.atan2(
            -float(nominal[2, 0]),
            math.hypot(float(nominal[0, 0]), float(nominal[1, 0])),
        )
        candidates = [float(np.clip(nominal_pitch, 0.0, math.pi / 2.0))]
        candidates.extend(math.radians(value) for value in GRASP_PITCH_CANDIDATES_DEG)
        unique: list[float] = []
        for value in candidates:
            if not any(math.isclose(value, old, abs_tol=1e-6) for old in unique):
                unique.append(value)
        return tuple(unique)

    def _plan_pick_at_pitch(
        self,
        target: Target,
        current: tuple[float, ...],
        pitch_rad: float,
    ) -> PickPlan:
        xyz = np.asarray(target.pose.position, dtype=float)
        pre_z = min(
            float(xyz[2] + self.settings.pregrasp_offset_m),
            self.settings.transit_height_m,
        )
        poses = {
            "pregrasp": self._pitch_pose((xyz[0], xyz[1], pre_z), pitch_rad),
            "grasp": self._pitch_pose(xyz, pitch_rad),
            "lift": self._pitch_pose(
                xyz + (0.0, 0.0, self.settings.lift_offset_m), pitch_rad
            ),
        }
        commands: list[MotionCommand] = []
        previous = current
        for name in ("pregrasp", "grasp"):
            command = self._solve(name, poses[name], previous, self.settings.open_gripper)
            commands.append(command)
            previous = command.joints
        commands.append(self._direct("close", previous, self.settings.closed_gripper))
        commands.append(
            self._solve("lift", poses["lift"], previous, self.settings.closed_gripper)
        )
        return PickPlan(target, tuple(commands))

    @staticmethod
    def _joint_travel(start: Sequence[float], plan: PickPlan) -> float:
        total = 0.0
        previous = np.asarray(start, dtype=float)
        for command in plan.commands:
            current = np.asarray(command.joints, dtype=float)
            total += float(np.linalg.norm(current - previous))
            previous = current
        return total

    def plan_pick_from_current(
        self, target: Target, current_joints: Sequence[float]
    ) -> PickPlan:
        if target.pose.frame_id != self.settings.frame_id:
            raise PlanningError(f"target frame must be {self.settings.frame_id}")
        current = tuple(float(value) for value in current_joints)
        if len(current) != 6:
            raise PlanningError("current_joints must contain six values")

        valid: list[tuple[float, float, PickPlan]] = []
        failures: list[str] = []
        for pitch_rad in self._candidate_pitches(target.pose.position):
            pitch_deg = math.degrees(pitch_rad)
            try:
                plan = self._plan_pick_at_pitch(target, current, pitch_rad)
                _validate_pick_plan(self.safety, current, plan)
                valid.append((self._joint_travel(current, plan), pitch_deg, plan))
            except (PlanningError, SafetyError) as exc:
                failures.append(f"{pitch_deg:.1f}°: {exc}")
        if not valid:
            detail = "; ".join(failures[:4])
            raise PlanningError(f"0°–90° 内没有安全抓取姿态。{detail}")

        _, pitch_deg, selected = min(valid, key=lambda item: item[0])
        self.last_grasp_pitch_deg = pitch_deg
        return selected

    def plan(self, target: Target, current_joints: Sequence[float]) -> PickPlan:
        """Keep the normal full cycle, but use flexible pitch for its pick section."""
        current = tuple(float(value) for value in current_joints)
        if len(current) != 6:
            raise PlanningError("current_joints must contain six values")
        pick = self.plan_pick_from_current(target, self.settings.search_joints)
        commands = [
            self._direct("home", self.settings.home_joints, self.settings.open_gripper),
            self._direct("search", self.settings.search_joints, self.settings.open_gripper),
            *pick.commands,
        ]
        lift = pick.command("lift")
        transit_position = (
            self.settings.place_position[0],
            self.settings.place_position[1],
            min(
                self.settings.transit_height_m,
                self.settings.place_position[2] + self.settings.place_clearance_m,
            ),
        )
        transit_pose = self._pose(transit_position)
        place_pose = self._pose(self.settings.place_position)
        transit = self._solve(
            "transit", transit_pose, lift.joints, self.settings.closed_gripper
        )
        place = self._solve(
            "place", place_pose, transit.joints, self.settings.closed_gripper
        )
        commands.extend(
            (
                transit,
                place,
                self._direct("open", place.joints, self.settings.open_gripper),
                self._direct("retreat", self.settings.home_joints, self.settings.open_gripper),
            )
        )
        return PickPlan(target, tuple(commands))


def _evaluate_stone_ik(
    planner: Any,
    safety: Any,
    stone: TrackedTarget,
    ik_seed: Sequence[float],
) -> StoneIkReport:
    kinematics = planner.kinematics
    xyz = np.asarray(stone.base_xyz, dtype=float)
    stages = {
        "pregrasp": IkStageResult(False, None, None),
        "grasp": IkStageResult(False, None, None),
    }
    full_plan_ok = False
    full_plan_error = ""
    safety_ok = False
    safety_error = ""
    try:
        target = Target(
            target_id=f"{stone.class_name}#{stone.track_id}",
            pose=planner._pose((float(xyz[0]), float(xyz[1]), float(xyz[2]))),
            confidence=float(stone.confidence),
            stamp_s=float(stone.timestamp),
        )
        pick_plan = planner.plan_pick_from_current(target, ik_seed)
        full_plan_ok = True
        for name in ("pregrasp", "grasp"):
            command = pick_plan.command(name)
            pose = command.target_pose
            if pose is None:
                continue
            transform = kinematics.forward_matrix(command.joints)
            position_error = float(
                np.linalg.norm(np.asarray(pose.position) - transform[:3, 3])
            )
            orientation_error = float(
                np.linalg.norm(
                    kinematics._orientation_error(
                        transform[:3, :3], np.asarray(pose.rotation)
                    )
                )
            )
            stages[name] = IkStageResult(
                position_error <= 0.004 and orientation_error <= 0.10,
                position_error,
                orientation_error,
            )
        try:
            _validate_pick_plan(safety, ik_seed, pick_plan)
            safety_ok = True
        except SafetyError as exc:
            safety_error = str(exc)
    except PlanningError as exc:
        full_plan_error = str(exc)

    return StoneIkReport(
        stone=stone,
        pregrasp=stages["pregrasp"],
        grasp=stages["grasp"],
        safety_ok=safety_ok,
        safety_error=safety_error,
        full_plan_ok=full_plan_ok,
        full_plan_error=full_plan_error,
    )


def _evaluate_stones_ik(
    planner: Any,
    safety: Any,
    stones: Sequence[TrackedTarget],
    ik_seed: Sequence[float],
) -> list[StoneIkReport]:
    reports = [_evaluate_stone_ik(planner, safety, stone, ik_seed) for stone in stones]
    reports.sort(key=lambda item: item.stone.confidence, reverse=True)
    return reports


def _ik_report_for_stone(
    reports: Sequence[StoneIkReport],
    stone: TrackedTarget,
) -> StoneIkReport | None:
    for report in reports:
        if report.stone.track_id == stone.track_id:
            return report
    return None


def _draw_preview(
    image: np.ndarray,
    detections: list[Any],
    stones: Sequence[TrackedTarget],
    hud: str,
    *,
    ik_reports: Sequence[StoneIkReport] | None = None,
) -> np.ndarray:
    view = image.copy()
    reachable_ids = {
        report.stone.track_id for report in (ik_reports or ()) if report.reachable
    }
    partial_ids = {
        report.stone.track_id
        for report in (ik_reports or ())
        if not report.reachable and (report.pregrasp.ok or report.grasp.ok)
    }
    stone_ids = {item.track_id for item in stones}
    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection.bbox_xyxy)
        if detection.track_id in reachable_ids:
            color = (0, 220, 80)
        elif detection.track_id in partial_ids:
            color = (0, 200, 255)
        elif detection.track_id in stone_ids:
            color = (40, 40, 240)
        else:
            color = (80, 80, 80)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} #{detection.track_id} {detection.confidence:.2f}"
        cv2.putText(view, label, (x1, max(20, y1 - 6)), 0, 0.55, color, 2, cv2.LINE_AA)
    for stone in stones:
        cx = int((stone.bbox_xyxy[0] + stone.bbox_xyxy[2]) * 0.5)
        cy = int((stone.bbox_xyxy[1] + stone.bbox_xyxy[3]) * 0.5)
        cv2.circle(view, (cx, cy), 5, (0, 0, 255), -1)
        text = (
            f"B=({stone.base_xyz[0]:+.3f},{stone.base_xyz[1]:+.3f},{stone.base_xyz[2]:+.3f})"
        )
        sx1, _, _, sy2 = stone.bbox_xyxy
        cv2.putText(
            view,
            text,
            (int(sx1), int(sy2) + 18),
            0,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    for index, line in enumerate(hud.split("\n")):
        cv2.putText(view, line, (12, 28 + 26 * index), 0, 0.65, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, 28 + 26 * index), 0, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return view


class ManualStoneTarget:
    """锁定一颗通过 IK 和安全检查的石头，供状态机抓取。"""

    def __init__(self) -> None:
        self._target: Target | None = None

    def set(self, stone: TrackedTarget) -> None:
        self._target = Target(
            target_id=f"{stone.class_name}#{stone.track_id}",
            pose=Pose.from_xyz(stone.base_xyz),
            confidence=stone.confidence,
            stamp_s=stone.timestamp,
        )
        print(
            "锁定石头 arm_base (m):",
            " ".join(f"{value:+.4f}" for value in stone.base_xyz),
            f"conf={stone.confidence:.3f}",
        )

    def lock(self) -> Target:
        if self._target is None:
            raise RuntimeError("尚未锁定石头")
        return self._target

    def prepare_retry(self) -> None:
        self._target = None

    def close(self) -> None:
        self._target = None


def preview_until_grasp(
    camera: OrbbecCamera,
    tracker: YoloTracker,
    arm: Arx5Backend,
    eef_T_camera: np.ndarray,
    planner: Any,
    safety: Any,
    *,
    initial_joints: Sequence[float],
    target_class: str,
    min_confidence: float,
    grasp_depth_offset_m: float,
    distortion: Sequence[float] | None,
    use_gui: bool,
    snapshot_path: Path,
    print_interval_s: float = 0.5,
) -> TrackedTarget | None:
    if use_gui:
        return _preview_until_grasp_gui(
            camera,
            tracker,
            arm,
            eef_T_camera,
            planner,
            safety,
            initial_joints=initial_joints,
            target_class=target_class,
            min_confidence=min_confidence,
            grasp_depth_offset_m=grasp_depth_offset_m,
            distortion=distortion,
            print_interval_s=print_interval_s,
        )
    return _preview_until_grasp_terminal(
        camera,
        tracker,
        arm,
        eef_T_camera,
        planner,
        safety,
        initial_joints=initial_joints,
        target_class=target_class,
        min_confidence=min_confidence,
        grasp_depth_offset_m=grasp_depth_offset_m,
        distortion=distortion,
        snapshot_path=snapshot_path,
        print_interval_s=print_interval_s,
    )


def _preview_hud(
    stones: Sequence[TrackedTarget],
    ik_reports: Sequence[StoneIkReport],
    *,
    target_class: str,
    min_confidence: float,
    best: TrackedTarget | None,
) -> str:
    hud = (
        f"stones={len(stones)} | target={target_class} | conf>={min_confidence:.2f}\n"
        "Auto grasp: highest-confidence reachable stone in current scan view\n"
        "绿=IK+安全 OK | 黄=部分 | 红=无解/不安全"
    )
    if best is not None:
        report = _ik_report_for_stone(ik_reports, best)
        ik_text = report.summary_line() if report is not None else "IK=?"
        hud += (
            f"\nready: #{best.track_id} conf={best.confidence:.2f} "
            f"B=({best.base_xyz[0]:+.3f},{best.base_xyz[1]:+.3f},{best.base_xyz[2]:+.3f})\n"
            f"{ik_text}"
        )
    elif not ik_reports:
        hud += "\nIK: stones=0"
    return hud


def _print_ik_status(
    ik_reports: Sequence[StoneIkReport],
    *,
    last_lines: tuple[str, ...],
    last_write: float,
    print_interval_s: float,
    empty_reason: str = "stones=0 | 无有效候选",
) -> tuple[tuple[str, ...], float]:
    lines = tuple(report.summary_line() for report in ik_reports)
    if not lines:
        lines = (empty_reason,)
    now = time.time()
    if lines != last_lines or now - last_write >= print_interval_s:
        for line in lines:
            print(line, flush=True)
        return lines, now
    return last_lines, last_write


def _empty_stones_reason(detections: Sequence[Any], stones: Sequence[TrackedTarget]) -> str:
    if stones:
        return ""
    if not detections:
        return "stones=0 | YOLO 未输出目标框（检查置信度、光照、角度或模型）"
    return (
        f"stones=0 | YOLO 已输出 {len(detections)} 个目标框，"
        "但框内有效深度不足"
    )


def _try_confirm_grasp(
    best: TrackedTarget | None,
    ik_reports: Sequence[StoneIkReport],
) -> TrackedTarget | None:
    if best is None:
        print("当前未检测到石头，无法抓取")
        return None
    report = _ik_report_for_stone(ik_reports, best)
    if report is None:
        print("当前石头 IK 状态未知，无法抓取")
        return None
    if not report.reachable:
        reason = "安全校验未通过" if report.pregrasp.ok and report.grasp.ok else "IK 不可解"
        print(f"{reason}，跳过抓取，继续预览：{report.summary_line()}", flush=True)
        for stage_name, stage in (("pregrasp", report.pregrasp), ("grasp", report.grasp)):
            if stage.limit_detail:
                print(f"  {stage_name} 触限/诊断: {stage.limit_detail}", flush=True)
        if report.safety_error:
            print(f"  安全校验: {report.safety_error}", flush=True)
        if report.full_plan_error:
            print(f"  plan: {report.full_plan_error}", flush=True)
        return None
    if not report.full_plan_ok:
        print(f"完整规划失败，跳过抓取，继续预览：{report.full_plan_error}", flush=True)
        return None
    return best


def _pick_best_reachable(
    ik_reports: Sequence[StoneIkReport],
) -> TrackedTarget | None:
    """Choose the nearest fully feasible target, then break ties by confidence."""
    candidates = [
        report.stone
        for report in ik_reports
        if report.reachable and report.full_plan_ok
    ]
    return min(
        candidates,
        key=lambda stone: (
            float(np.linalg.norm(stone.base_xyz)),
            -float(stone.confidence),
        ),
        default=None,
    )


def _preview_until_grasp_gui(
    camera: OrbbecCamera,
    tracker: YoloTracker,
    arm: Arx5Backend,
    eef_T_camera: np.ndarray,
    planner: Any,
    safety: Any,
    *,
    initial_joints: Sequence[float],
    target_class: str,
    min_confidence: float,
    grasp_depth_offset_m: float,
    distortion: Sequence[float] | None,
    print_interval_s: float,
) -> TrackedTarget | None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    last_lines: tuple[str, ...] = ()
    last_write = 0.0
    try:
        while True:
            frame, detections, stones, best = _collect_stones(
                camera,
                tracker,
                arm,
                eef_T_camera,
                target_class=target_class,
                min_confidence=min_confidence,
                grasp_depth_offset_m=grasp_depth_offset_m,
                distortion=distortion,
            )
            ik_reports = _evaluate_stones_ik(planner, safety, stones, initial_joints)
            last_lines, last_write = _print_ik_status(
                ik_reports,
                last_lines=last_lines,
                last_write=last_write,
                print_interval_s=print_interval_s,
                empty_reason=_empty_stones_reason(detections, stones),
            )
            reachable = _pick_best_reachable(ik_reports)
            if reachable is not None:
                print(
                    f"自动选择可达目标 #{reachable.track_id} "
                    f"conf={reachable.confidence:.2f}，开始抓取。",
                    flush=True,
                )
                return reachable
            hud = _preview_hud(
                stones,
                ik_reports,
                target_class=target_class,
                min_confidence=min_confidence,
                best=best,
            )
            display = _draw_preview(
                frame.color_bgr,
                [replace(d, detail="") for d in detections],
                stones,
                hud,
                ik_reports=ik_reports,
            )
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return None
            if key in (ord("g"), ord("G")):
                confirmed = _try_confirm_grasp(best, ik_reports)
                if confirmed is not None:
                    return confirmed
    finally:
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass
        _safe_cv2_close()


def _preview_until_grasp_terminal(
    camera: OrbbecCamera,
    tracker: YoloTracker,
    arm: Arx5Backend,
    eef_T_camera: np.ndarray,
    planner: Any,
    safety: Any,
    *,
    initial_joints: Sequence[float],
    target_class: str,
    min_confidence: float,
    grasp_depth_offset_m: float,
    distortion: Sequence[float] | None,
    snapshot_path: Path,
    print_interval_s: float,
) -> TrackedTarget | None:
    print("终端模式（OpenCV 无 GUI）：检测到 IK 和安全规划通过的石头后自动抓取。")
    print("输入 q + Enter 可在自动抓取前退出。")
    print(f"带标注预览会写入：{snapshot_path}")
    cmd_q = _stdin_command_queue()
    last_lines: tuple[str, ...] = ()
    last_write = 0.0
    last_snapshot_write = 0.0
    while True:
        frame, detections, stones, best = _collect_stones(
            camera,
            tracker,
            arm,
            eef_T_camera,
            target_class=target_class,
            min_confidence=min_confidence,
            grasp_depth_offset_m=grasp_depth_offset_m,
            distortion=distortion,
        )
        ik_reports = _evaluate_stones_ik(planner, safety, stones, initial_joints)
        last_lines, last_write = _print_ik_status(
            ik_reports,
            last_lines=last_lines,
            last_write=last_write,
            print_interval_s=print_interval_s,
            empty_reason=_empty_stones_reason(detections, stones),
        )
        reachable = _pick_best_reachable(ik_reports)
        if reachable is not None:
            print(
                f"自动选择可达目标 #{reachable.track_id} "
                f"conf={reachable.confidence:.2f}，开始抓取。",
                flush=True,
            )
            return reachable
        hud = _preview_hud(
            stones,
            ik_reports,
            target_class=target_class,
            min_confidence=min_confidence,
            best=best,
        )
        now = time.time()
        if now - last_snapshot_write >= print_interval_s:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            display = _draw_preview(
                frame.color_bgr,
                [replace(d, detail="") for d in detections],
                stones,
                hud,
                ik_reports=ik_reports,
            )
            cv2.imwrite(str(snapshot_path), display)
            last_snapshot_write = now

        while not cmd_q.empty():
            cmd = cmd_q.get_nowait()
            if cmd in ("q", "quit", "exit"):
                return None
            if cmd in ("g", "grasp", "go"):
                confirmed = _try_confirm_grasp(best, ik_reports)
                if confirmed is not None:
                    return confirmed
            if cmd:
                print("未知命令，请输入 g 或 q。")


def _move_j1_to_scan_angle(
    arm: Arx5Backend,
    safety: Any,
    planner: Any,
    *,
    target_rad: float,
) -> tuple[float, ...]:
    """Move J1 to an absolute scan angle while preserving the look-down pose."""
    current = arm.current_joints()
    target = list(current)
    target[0] = float(target_rad)
    lower = float(safety.limits.joint_lower[0])
    upper = float(safety.limits.joint_upper[0])
    if target[0] < lower or target[0] > upper:
        raise RuntimeError(
            f"J1 扫描角 {math.degrees(target_rad):+.0f}° 超出限位 "
            f"[{math.degrees(lower):+.0f}°, {math.degrees(upper):+.0f}°]"
        )
    command = MotionCommand(
        "j1_scan",
        tuple(float(value) for value in target),
        float(planner.settings.open_gripper),
        float(planner.settings.durations_s.get("search", 2.0)),
    )
    try:
        safety.validate_path(current, command)
    except SafetyError as exc:
        raise RuntimeError(
            f"J1 扫描到 {math.degrees(target_rad):+.0f}° 的路径不安全：{exc}"
        ) from exc
    print(
        f"J1 扫描位：{math.degrees(current[0]):+.0f}° -> "
        f"{math.degrees(target_rad):+.0f}°"
    )
    result = arm.execute(command)
    if not result.reached:
        raise RuntimeError(result.reason or "J1 扫描姿态未到位")
    settled = read_joints_from_backend(arm)
    print(f"扫描姿态到位关节(rad): {_joints_line(settled)}")
    return settled


def _scan_one_view_for_reachable_stone(
    camera: OrbbecCamera,
    tracker: YoloTracker,
    arm: Arx5Backend,
    eef_T_camera: np.ndarray,
    planner: Any,
    safety: Any,
    *,
    initial_joints: Sequence[float],
    target_class: str,
    min_confidence: float,
    grasp_depth_offset_m: float,
    distortion: Sequence[float] | None,
    dwell_s: float,
    use_gui: bool,
    snapshot_path: Path,
    print_interval_s: float,
) -> TrackedTarget | None:
    """Observe one fixed arm view for at most ``dwell_s`` seconds.

    A detected stone must still pass the existing pregrasp/grasp IK, full path,
    and safety checks before it is returned for motion.
    """
    deadline = time.monotonic() + dwell_s
    last_lines: tuple[str, ...] = ()
    last_write = 0.0
    last_snapshot_write = 0.0
    while time.monotonic() < deadline:
        try:
            frame, detections, stones, best = _collect_stones(
                camera,
                tracker,
                arm,
                eef_T_camera,
                target_class=target_class,
                min_confidence=min_confidence,
                grasp_depth_offset_m=grasp_depth_offset_m,
                distortion=distortion,
            )
        except RuntimeError as exc:
            if "RGB-D 对齐失败" not in str(exc):
                raise
            print(f"相机瞬时丢帧，跳过当前扫描姿态并保持视频流：{exc}")
            return None

        ik_reports = _evaluate_stones_ik(planner, safety, stones, initial_joints)
        last_lines, last_write = _print_ik_status(
            ik_reports,
            last_lines=last_lines,
            last_write=last_write,
            print_interval_s=print_interval_s,
            empty_reason=_empty_stones_reason(detections, stones),
        )
        reachable = _pick_best_reachable(ik_reports)
        hud = _preview_hud(
            stones,
            ik_reports,
            target_class=target_class,
            min_confidence=min_confidence,
            best=best,
        )
        display = _draw_preview(
            frame.color_bgr,
            [replace(d, detail="") for d in detections],
            stones,
            hud,
            ik_reports=ik_reports,
        )
        if use_gui:
            cv2.imshow(WINDOW, display)
            cv2.waitKey(1)
        elif time.monotonic() - last_snapshot_write >= print_interval_s:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot_path), display)
            last_snapshot_write = time.monotonic()

        # Display/record the successful detection frame before leaving the
        # scan loop.  Returning earlier made first-frame detections produce a
        # complete grasp with a zero-frame recording.
        if reachable is not None:
            print(
                f"扫描发现可抓取石头 #{reachable.track_id} "
                f"conf={reachable.confidence:.2f}，立即抓取。",
                flush=True,
            )
            return reachable
    return None


def _wait_for_next_cycle() -> bool:
    """Return True for ``a`` (next scan), False for ``q`` (safe exit)."""
    while True:
        try:
            command = input("机械臂已回零：输入 a 开始下一轮扫描抓取，输入 q 退出：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if command in ("a", "again", "continue"):
            return True
        if command in ("q", "quit", "exit"):
            return False
        print("未知命令，请输入 a 或 q。")


def _joints_line(joints: Sequence[float]) -> str:
    return format_joints(tuple(joints))


def _eef_pitch_deg(kinematics: Any, joints: Sequence[float]) -> float:
    rotation = kinematics.forward_matrix(joints)[:3, :3]
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    return math.degrees(pitch)


def _prepare_look_down_pose(
    arm: Arx5Backend,
    safety: Any,
    planner: Any,
    *,
    delta_rad: float = J2_LOOK_DOWN_DELTA_RAD,
) -> tuple[float, ...]:
    """识别前调整关节2，使末端俯仰朝下（正 delta = j2 增大，与 search 位一致）。"""
    kinematics = planner.kinematics
    current = arm.current_joints()
    lower = safety.limits.joint_lower
    upper = safety.limits.joint_upper
    duration = float(planner.settings.durations_s.get("search", 2.0))
    requested_deg = abs(math.degrees(delta_rad))
    sign = 1.0 if delta_rad >= 0.0 else -1.0
    if requested_deg <= 0.0:
        current = arm.current_joints()
        print("识别前关节2调整角度为 0°，保持当前角度。")
        return current

    chosen_delta: float | None = None
    chosen_deg = 0.0
    for deg in range(int(round(requested_deg)), 0, -5):
        trial = sign * math.radians(deg)
        target = list(current)
        target[1] = float(current[1]) + trial
        if target[1] < float(lower[1]) or target[1] > float(upper[1]):
            continue
        command = MotionCommand(
            "look_down",
            tuple(float(value) for value in target),
            float(planner.settings.open_gripper),
            duration,
        )
        try:
            safety.validate_path(current, command)
        except SafetyError:
            continue
        chosen_delta = trial
        chosen_deg = float(deg)
        break

    if chosen_delta is None:
        raise RuntimeError(
            f"无法找到满足关节/工作空间限位的识别姿态（请求 j2 {'+' if sign > 0 else '-'}{requested_deg:.0f}°）。"
            "可尝试 --look-down-deg -30 反向，或 --skip-look-down。"
        )

    target = list(current)
    target[1] = float(current[1]) + chosen_delta
    command = MotionCommand(
        "look_down",
        tuple(float(value) for value in target),
        float(planner.settings.open_gripper),
        duration,
    )
    start_tcp = kinematics.forward_position(current)
    target_tcp = kinematics.forward_position(target)
    direction = "增大" if sign > 0 else "减小"
    if chosen_deg < requested_deg - 1e-6:
        print(
            f"注意：请求 j2 {direction} {requested_deg:.0f}° 超出工作空间高度，"
            f"自动改用 {chosen_deg:.0f}°。"
        )
    print(
        "识别前调整姿态：关节2 "
        f"{current[1]:+.4f} -> {target[1]:+.4f} rad ({direction} {chosen_deg:.0f}°) | "
        f"TCP z {start_tcp[2]:+.3f}->{target_tcp[2]:+.3f} m | "
        f"pitch {_eef_pitch_deg(kinematics, current):+.1f}°->{_eef_pitch_deg(kinematics, target):+.1f}°"
    )
    result = arm.execute(command)
    if not result.reached:
        raise RuntimeError(result.reason or "关节2 识别姿态未到位")
    settled = read_joints_from_backend(arm)
    print(f"识别姿态到位关节(rad): {_joints_line(settled)}")
    return settled


def _move_j2_to_scan_angle(
    arm: Arx5Backend,
    safety: Any,
    planner: Any,
    *,
    home_j2_rad: float,
    look_down_deg: float,
    duration_s: float = SCAN_J2_STEP_DURATION_S,
) -> tuple[float, ...]:
    """Set J2 to an absolute look-down angle relative to the home pose."""
    current = arm.current_joints()
    target = list(current)
    target[1] = float(home_j2_rad) + math.radians(float(look_down_deg))
    lower = float(safety.limits.joint_lower[1])
    upper = float(safety.limits.joint_upper[1])
    if target[1] < lower or target[1] > upper:
        raise RuntimeError(
            f"J2 俯视 {look_down_deg:+.0f}° 超出限位 "
            f"[{math.degrees(lower - home_j2_rad):+.0f}°, "
            f"{math.degrees(upper - home_j2_rad):+.0f}°]"
        )
    command = MotionCommand(
        "j2_scan",
        tuple(float(value) for value in target),
        float(planner.settings.open_gripper),
        float(duration_s),
    )
    try:
        safety.validate_path(current, command)
    except SafetyError as exc:
        raise RuntimeError(
            f"J2 扫描到俯视 {look_down_deg:+.0f}° 的路径不安全：{exc}"
        ) from exc
    print(
        f"J2 扫描位：相对零位 {math.degrees(current[1] - home_j2_rad):+.0f}° -> "
        f"{look_down_deg:+.0f}°"
    )
    result = arm.execute(command)
    if not result.reached:
        raise RuntimeError(result.reason or "J2 扫描姿态未到位")
    settled = read_joints_from_backend(arm)
    print(f"扫描姿态到位关节(rad): {_joints_line(settled)}")
    return settled


def _prepare_initial_yaw_pose(
    arm: Arx5Backend,
    safety: Any,
    planner: Any,
    *,
    delta_rad: float = J1_INITIAL_YAW_DELTA_RAD,
) -> tuple[float, ...]:
    """扫描前旋转基座关节1（正值逆时针，负值顺时针）。"""
    if abs(delta_rad) <= 1e-9:
        current = arm.current_joints()
        print("扫描前关节1旋转角度为 0°，保持当前角度。")
        return current
    current = arm.current_joints()
    target = list(current)
    target[0] = float(current[0]) + float(delta_rad)
    direction = "逆时针" if delta_rad > 0.0 else "顺时针"
    requested_deg = abs(math.degrees(delta_rad))
    lower = float(safety.limits.joint_lower[0])
    upper = float(safety.limits.joint_upper[0])
    if target[0] < lower or target[0] > upper:
        raise RuntimeError(
            f"关节1无法从当前角度严格{direction}旋转 "
            f"{requested_deg:.0f}°：目标 {target[0]:+.4f} rad，"
            f"限位 [{lower:+.4f}, {upper:+.4f}]"
        )
    command = MotionCommand(
        "initial_yaw_scan",
        tuple(float(value) for value in target),
        float(planner.settings.open_gripper),
        float(planner.settings.durations_s.get("search", 2.0)),
    )
    try:
        safety.validate_path(current, command)
    except SafetyError as exc:
        raise RuntimeError(f"初始{direction}扫描路径不安全：{exc}") from exc
    print(
        "扫描前调整姿态：关节1 "
        f"{current[0]:+.4f} -> {target[0]:+.4f} rad "
        f"({direction} {math.degrees(delta_rad):+.0f}°)"
    )
    result = arm.execute(command)
    if not result.reached:
        raise RuntimeError(result.reason or f"关节1 初始{direction}扫描姿态未到位")
    settled = read_joints_from_backend(arm)
    print(f"初始扫描姿态到位关节(rad): {_joints_line(settled)}")
    return settled


def _print_joint_limits(safety: Any) -> None:
    lower = safety.limits.joint_lower
    upper = safety.limits.joint_upper
    print(
        "安全限位(rad): "
        + " ".join(f"j{i + 1}[{float(lower[i]):+.3f},{float(upper[i]):+.3f}]" for i in range(6))
    )


def _debug_motion_context(
    *,
    stage: str,
    arm: Arx5Backend,
    safety: Any,
    command: MotionCommand | None = None,
    target: Target | None = None,
    plan: Any | None = None,
) -> None:
    joints = arm.current_joints()
    print(f"[{stage}] 实测关节: {_joints_line(joints)}")
    if command is not None:
        print(f"[{stage}] 命令目标: {_joints_line(command.joints)} gripper={command.gripper:.4f}")
        if command.target_pose is not None:
            pose = command.target_pose
            print(
                f"[{stage}] 命令位姿 arm_base (m): "
                f"xyz=({pose.position[0]:+.4f},{pose.position[1]:+.4f},{pose.position[2]:+.4f})"
            )
    if target is not None:
        pose = target.pose
        print(
            f"[{stage}] 抓取目标 arm_base (m): "
            f"xyz=({pose.position[0]:+.4f},{pose.position[1]:+.4f},{pose.position[2]:+.4f}) "
            f"conf={target.confidence:.3f}"
        )
    if plan is not None:
        print(f"[{stage}] 规划步骤: {' -> '.join(command.name for command in plan.commands)}")
        for step_name in ("pregrasp", "grasp", "close", "lift"):
            try:
                step = plan.command(step_name)
            except Exception:
                continue
            print(f"[{stage}] plan.{step_name}: {_joints_line(step.joints)} gripper={step.gripper:.4f}")
            if step.target_pose is not None:
                pose = step.target_pose
                print(
                    f"[{stage}] plan.{step_name} pose xyz="
                    f"({pose.position[0]:+.4f},{pose.position[1]:+.4f},{pose.position[2]:+.4f})"
                )


def _validate_path_debug(
    arm: Arx5Backend,
    safety: Any,
    command: MotionCommand,
    *,
    stage: str,
) -> None:
    _debug_motion_context(stage=stage, arm=arm, safety=safety, command=command)
    try:
        safety.validate_path(arm.current_joints(), command)
    except SafetyError as exc:
        _print_joint_limits(safety)
        raise RuntimeError(f"{stage} 安全校验失败：{exc}") from exc


def _motion_command(settings: Any, name: str, joints: Sequence[float]) -> MotionCommand:
    return MotionCommand(
        name,
        tuple(float(value) for value in joints),
        float(settings.open_gripper),
        float(settings.durations_s[name]),
    )


def _execute_named(
    arm: Arx5Backend,
    safety: Any,
    command: MotionCommand,
    on_state: Callable[[State], None],
    state: State,
) -> None:
    on_state(state)
    _validate_path_debug(arm, safety, command, stage=state.value)
    result = arm.execute(command)
    if not result.reached:
        _debug_motion_context(stage=f"{state.value} 未到位", arm=arm, safety=safety, command=command)
        raise AttemptError(result.reason or f"{command.name} 未到位")
    print(f"[{state.value}] 到位后关节: {_joints_line(arm.current_joints())}")


def _execute_plan_step(
    arm: Arx5Backend,
    safety: Any,
    plan: Any,
    name: str,
    on_state: Callable[[State], None],
    state: State,
) -> None:
    on_state(state)
    command = plan.command(name)
    _validate_path_debug(arm, safety, command, stage=state.value)
    result = arm.execute(command)
    if not result.reached:
        _debug_motion_context(stage=f"{state.value} 未到位", arm=arm, safety=safety, command=command, plan=plan)
        raise AttemptError(result.reason or f"{name} 未到位")
    print(f"[{state.value}] 到位后关节: {_joints_line(arm.current_joints())}")


def _load_teach_waypoints(path: Path) -> list[tuple[float, ...]]:
    """读取 teach_joint_pose.py 生成的示教 JSON（路点列表会随示教更新）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("waypoints", [])
    if not points:
        raise RuntimeError(f"{path} 中没有路点")
    waypoints: list[tuple[float, ...]] = []
    for index, item in enumerate(points):
        joints = np.asarray(item["joints_rad"], dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"路点 #{index} 关节数据无效")
        waypoints.append(tuple(float(value) for value in joints))
    return waypoints


def _go_home(
    arm: Arx5Backend,
    safety: Any,
    planner: Any,
    on_state: Callable[[State], None],
    *,
    gripper: float,
    label: str,
) -> None:
    settings = planner.settings
    home_joints = tuple(float(value) for value in settings.home_joints)
    closed = float(settings.closed_gripper)
    gripper_label = "闭合" if float(gripper) <= closed + 1e-6 else "张开"
    print(f"{label}回零(rad): {_joints_line(home_joints)}（夹爪{gripper_label}）")
    cmd = MotionCommand(
        "home",
        home_joints,
        float(gripper),
        float(settings.durations_s["home"]),
    )
    _execute_named(arm, safety, cmd, on_state, State.HOME)


def _replay_teach_route(
    arm: Arx5Backend,
    safety: Any,
    planner: Any,
    on_state: Callable[[State], None],
    teach_path: Path,
    *,
    segment_duration_s: float,
    waypoint_hold_s: float,
) -> None:
    """持物闭合回放示教路线，完成后在当前位张开夹爪、停顿、再回零张开。"""
    settings = planner.settings
    closed = float(settings.closed_gripper)
    open_gripper = float(settings.open_gripper)
    waypoints = _load_teach_waypoints(teach_path)
    print(f"开始回放示教路线：{teach_path}（{len(waypoints)} 路点，夹爪保持闭合）")
    for index, joints in enumerate(waypoints):
        print(f"示教路点 {index + 1}/{len(waypoints)}: {_joints_line(joints)}")
        cmd = MotionCommand(
            f"teach_wp{index}",
            joints,
            closed,
            float(segment_duration_s),
        )
        _execute_named(arm, safety, cmd, on_state, State.TRANSIT)
        if waypoint_hold_s > 0.0:
            time.sleep(float(waypoint_hold_s))
    print("示教回放完成，在放置位张开夹爪")
    release_joints = arm.current_joints()
    open_cmd = MotionCommand(
        "open",
        release_joints,
        open_gripper,
        float(settings.durations_s["open"]),
    )
    _execute_named(arm, safety, open_cmd, on_state, State.OPEN)
    print(f"停顿 {PUT_PAUSE_S:.1f} s")
    time.sleep(PUT_PAUSE_S)
    _go_home(arm, safety, planner, on_state, gripper=open_gripper, label="放置后")


def _return_to_home_zero(
    arm: Arx5Backend,
    safety: Any,
    planner: Any,
    on_state: Callable[[State], None],
) -> None:
    """抬升完成后持物回 home_joints（默认全零），不松爪。"""
    _go_home(
        arm,
        safety,
        planner,
        on_state,
        gripper=float(planner.settings.closed_gripper),
        label="抓取后",
    )


def run_grasp_only(
    arm: Arx5Backend,
    planner: Any,
    safety: Any,
    target: Target,
    on_state: Callable[[State], None],
    *,
    initial_joints: Sequence[float],
    teach_path: Path,
    segment_duration_s: float,
    waypoint_hold_s: float,
) -> bool:
    """抓取 → 回零（持物）→ 回放示教放置 → 松爪 → 停顿 → 回零（张开）。"""
    initial = tuple(float(value) for value in initial_joints)
    on_state(State.PLAN)
    _debug_motion_context(stage="PLAN 输入", arm=arm, safety=safety, target=target)
    initial_tcp = planner.kinematics.forward_position(initial)
    print(
        "初始关节 FK TCP (arm_base, m): "
        f"xyz=({initial_tcp[0]:+.3f},{initial_tcp[1]:+.3f},{initial_tcp[2]:+.3f}) "
        f"radius={float(np.hypot(initial_tcp[0], initial_tcp[1])):.3f}"
    )
    target_radius = float(np.hypot(target.pose.position[0], target.pose.position[1]))
    print(f"目标水平半径 {target_radius:.3f} m（规划种子=当前实测关节）")
    current = arm.current_joints()
    try:
        plan = planner.plan_pick_from_current(target, current)
    except Exception as exc:
        _debug_motion_context(stage="PLAN 失败", arm=arm, safety=safety, target=target)
        raise RuntimeError(f"抓取规划失败（可能为 IK/工作空间）：{exc}") from exc
    selected_pitch = getattr(planner, "last_grasp_pitch_deg", None)
    if selected_pitch is not None:
        print(f"自动选择抓取俯仰角: {float(selected_pitch):.1f}°（搜索范围 0°–90°）")
    _debug_motion_context(stage="PLAN 输出", arm=arm, safety=safety, target=target, plan=plan)
    try:
        _validate_pick_plan(safety, current, plan)
    except SafetyError as exc:
        _print_joint_limits(safety)
        _debug_motion_context(stage="PLAN 路径校验失败", arm=arm, safety=safety, target=target, plan=plan)
        raise RuntimeError(f"规划路径安全校验失败：{exc}") from exc
    _execute_plan_step(arm, safety, plan, "pregrasp", on_state, State.PREGRASP)
    _execute_plan_step(arm, safety, plan, "grasp", on_state, State.GRASP)
    _execute_plan_step(arm, safety, plan, "close", on_state, State.CLOSE)
    on_state(State.VERIFY_GRASP)
    if not arm.has_object():
        print("闭合后未检测到持物；自动张开夹爪、撤回 pregrasp 并归零。")
        open_gripper = float(planner.settings.open_gripper)
        current = arm.current_joints()
        _execute_named(
            arm,
            safety,
            MotionCommand(
                "open_after_miss",
                current,
                open_gripper,
                float(planner.settings.durations_s["open"]),
            ),
            on_state,
            State.OPEN,
        )
        pregrasp = plan.command("pregrasp")
        _execute_named(
            arm,
            safety,
            MotionCommand(
                "retreat_after_miss",
                pregrasp.joints,
                open_gripper,
                pregrasp.duration_s,
                pregrasp.target_pose,
            ),
            on_state,
            State.PREGRASP,
        )
        _go_home(
            arm,
            safety,
            planner,
            on_state,
            gripper=open_gripper,
            label="未抓到后",
        )
        print("未抓到目标，机械臂已归零并保持使能。")
        arm.hold_enabled(block=False)
        return False
    _execute_plan_step(arm, safety, plan, "lift", on_state, State.LIFT)
    _return_to_home_zero(arm, safety, planner, on_state)
    _replay_teach_route(
        arm,
        safety,
        planner,
        on_state,
        teach_path,
        segment_duration_s=segment_duration_s,
        waypoint_hold_s=waypoint_hold_s,
    )
    on_state(State.DONE)
    print("抓取+放置完成，机械臂已回零并保持使能；主循环将直接等待 a/q。")
    arm.hold_enabled(block=False)
    return True


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    _require_orbbec_sdk()
    config = load_config(root)
    weights = Path(args.weights)
    if not weights.is_file() and not weights.is_absolute():
        alt = root / weights
        if alt.is_file():
            weights = alt
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在：{weights}")

    target_class = _load_target_class(Path(args.names), args.target_class)
    calib_path = _resolve_calibration(root, Path(args.calibration) if args.calibration else None)
    eef_T_camera = load_eye_in_hand(calib_path)
    print(f"YOLO 权重: {weights}")
    print(f"目标类别: {target_class}")
    print(f"手眼标定: {calib_path}")

    base_planner, safety = build_core(config)
    planner = FlexiblePitchGraspPlanner(base_planner, safety)
    arm = Arx5Backend(
        root,
        config.hardware,
        dry_run=not args.confirm,
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
    tracker = YoloTracker(
        weights=str(weights),
        confidence=float(args.confidence),
        device=str(args.device),
        tracker="bytetrack.yaml",
        image_size=int(args.imgsz),
        target_classes={target_class},
    )
    provider = ManualStoneTarget()
    on_state = lambda state: print(state.value)

    try:
        print(
            "连接 CAN；每轮从零位开始，关节1以 0°、+90°、-90° 为中心，"
            "分别扫描 -20° 到 +20° 偏移（10°/步）；每个 J1 点下关节2依次俯视 "
            "0°、7°、14°、21°、28°、35°（0.1 秒/步）。"
        )
        arm.connect()
        ok, reason = arm.self_check()
        if not ok:
            raise RuntimeError(reason)
        startup_joints = read_joints_from_backend(arm)
        print(f"启动实测关节(rad): {_joints_line(startup_joints)}")
        camera.start()
        if args.headless and args.gui:
            raise RuntimeError("不能同时指定 --headless 与 --gui")
        if args.headless:
            use_gui = False
        elif args.gui:
            if not _opencv_gui_available():
                raise RuntimeError(
                    "已指定 --gui，但当前 OpenCV 无窗口支持。"
                    "请执行：pip uninstall -y opencv-python-headless && pip install opencv-python"
                )
            use_gui = True
        else:
            use_gui = _opencv_gui_available()
        if use_gui:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            print(
                "识别窗口已打开；每个 J1/J2 扫描姿态最多观察 "
                f"{float(args.scan_dwell_s):.1f} 秒，检测到可达石头即自动抓取。"
            )
        else:
            if not args.headless and not os.environ.get("DISPLAY"):
                print("未检测到 DISPLAY，自动使用终端模式。")
            elif not args.headless:
                print("OpenCV 无 GUI（常见为 opencv-python-headless），自动使用终端模式。")
        snapshot_path = Path(args.snapshot)
        if not snapshot_path.is_absolute():
            snapshot_path = root / snapshot_path
        teach_path = Path(args.teach_file)
        if not teach_path.is_absolute():
            teach_path = root / teach_path
        if not args.full_cycle and not teach_path.is_file():
            raise FileNotFoundError(f"示教文件不存在：{teach_path}")
        if not args.full_cycle:
            print(f"示教放置路线: {teach_path}")

        print(
            f"抓取点沿相机射线向石头内部推进 {float(args.grasp_depth_offset) * 1000.0:.0f} mm。"
        )
        print("三个方向都没有可抓目标时回零；每轮结束后输入 a 继续，q 退出。")
        cycle = 0
        while True:
            cycle += 1
            print(f"\n========== 扫描抓取第 {cycle} 轮 ==========")
            _go_home(
                arm,
                safety,
                planner,
                on_state,
                gripper=float(planner.settings.open_gripper),
                label="扫描前",
            )
            stone: TrackedTarget | None = None
            home_j2_rad = float(planner.settings.home_joints[1])
            scan_pitch_degrees = (0.0,) if args.skip_look_down else SCAN_J2_ANGLES_DEG
            view_names = {0.0: "正前方", 90.0: "左侧", -90.0: "右侧"}
            for center_deg in SCAN_J1_CENTERS_DEG:
                view_name = view_names[center_deg]
                for offset_deg in SCAN_J1_OFFSETS_DEG:
                    angle_deg = center_deg + offset_deg
                    scan_seed = _move_j1_to_scan_angle(
                        arm,
                        safety,
                        planner,
                        target_rad=math.radians(angle_deg),
                    )
                    for look_down_deg in scan_pitch_degrees:
                        scan_seed = _move_j2_to_scan_angle(
                            arm,
                            safety,
                            planner,
                            home_j2_rad=home_j2_rad,
                            look_down_deg=look_down_deg,
                        )
                        print(
                            f"{view_name}扫描：J1 {angle_deg:+.0f}° "
                            f"(中心偏移 {offset_deg:+.0f}°)，J2 俯视 {look_down_deg:+.0f}°，"
                            f"最多 {float(args.scan_dwell_s):.1f} 秒。"
                        )
                        stone = _scan_one_view_for_reachable_stone(
                            camera,
                            tracker,
                            arm,
                            eef_T_camera,
                            planner,
                            safety,
                            initial_joints=scan_seed,
                            target_class=target_class,
                            min_confidence=float(args.confidence),
                            grasp_depth_offset_m=float(args.grasp_depth_offset),
                            distortion=config.calibration["distortion_coefficients"],
                            dwell_s=float(args.scan_dwell_s),
                            use_gui=use_gui,
                            snapshot_path=snapshot_path,
                            print_interval_s=float(args.print_interval),
                        )
                        if stone is not None:
                            break
                    if stone is not None:
                        break

                    # 当前 J1 点扫描结束后，J2 从 35° 直接回到 0°。
                    if not args.skip_look_down:
                        scan_seed = _move_j2_to_scan_angle(
                            arm,
                            safety,
                            planner,
                            home_j2_rad=home_j2_rad,
                            look_down_deg=0.0,
                        )
                if stone is not None:
                    break

            if stone is None:
                print("三个观察方向均未发现可抓取石头，回到零位。")
                _go_home(
                    arm,
                    safety,
                    planner,
                    on_state,
                    gripper=float(planner.settings.open_gripper),
                    label="扫描结束",
                )
            else:
                provider.set(stone)
                _print_joint_limits(safety)
                print(f"当前关节: {_joints_line(arm.current_joints())}")
                if args.full_cycle:
                    machine = GraspStateMachine(
                        arm, provider, planner, safety, on_state,
                        max_retries=0, hold_after_done=False, manage_resources=False,
                    )
                    result = machine.run()
                    print("状态历史:", " -> ".join(result.history))
                    if not result.success:
                        print(f"本次抓取失败：{result.error}")
                    _go_home(
                        arm, safety, planner, on_state,
                        gripper=float(planner.settings.open_gripper), label="完整周期后",
                    )
                else:
                    grasped = run_grasp_only(
                        arm, planner, safety, provider.lock(), on_state,
                        initial_joints=scan_seed, teach_path=teach_path,
                        segment_duration_s=float(args.segment_duration),
                        waypoint_hold_s=float(args.waypoint_hold),
                    )
                    print("本轮抓取成功。" if grasped else "本轮未成功抓取，已回零。")
            provider.prepare_retry()
            arm.hold_enabled(block=False)
            if not _wait_for_next_cycle():
                print("收到退出指令，机械臂保持零位并退出。")
                return 0
    except (AttemptError, HardwareFault, RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"抓取失败：{exc}")
        try:
            print("发生错误，正在将机械臂安全回到零位……")
            _go_home(
                arm,
                safety,
                planner,
                on_state,
                gripper=float(planner.settings.open_gripper),
                label="异常后",
            )
            arm.hold_enabled(block=False)
            print("机械臂已回到零位并保持使能。")
        except Exception as home_exc:
            print(f"警告：异常后自动回零失败：{home_exc}")
        return 2
    finally:
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass
        try:
            camera.close()
        except Exception:
            pass
        try:
            tracker.close()
        except cv2.error:
            pass
        arm.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="robot_arm 根目录")
    parser.add_argument("--confirm", action="store_true", help="连接 CAN 并允许真机运动")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="石头检测 YOLO 权重（默认 models/yolo/best.pt）",
    )
    parser.add_argument(
        "--names",
        default=str(YOLO_DIR / "best.names.json"),
        help="类别名 JSON",
    )
    parser.add_argument("--target-class", default="", help="覆盖类别名，默认读 names 文件")
    parser.add_argument(
        "--calibration",
        default="",
        help="手眼 JSON；默认 config/eye_in_hand_calibration.json",
    )
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO 置信度阈值")
    parser.add_argument("--device", default="auto", help="YOLO 设备：auto/cpu/0")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument(
        "--grasp-depth-offset",
        type=float,
        default=0.025,
        help="从可见表面沿相机射线向石头内部推进的距离(m，默认 0.025)",
    )
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="抓取后继续放置、张开夹爪并撤离（默认只抓到抬升后保持）",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="强制终端模式（每轮结束后 a/q + Enter），不弹 OpenCV 窗口",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="强制使用 OpenCV 窗口（无 GUI 支持时报错）",
    )
    parser.add_argument(
        "--snapshot",
        default=str(PREVIEW_SNAPSHOT.relative_to(ROOT)),
        help="终端模式下写入带标注的预览图路径",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="预览阶段终端 IK 状态重复打印最小间隔(秒)",
    )
    parser.add_argument(
        "--look-down-deg",
        type=float,
        default=10.0,
        help="识别前关节2调整角度（度）；正数=关节2增大/俯视朝下，负数=反向；超限自动缩小",
    )
    parser.add_argument(
        "--skip-look-down",
        action="store_true",
        help="跳过识别前的关节2调整",
    )
    parser.add_argument(
        "--scan-dwell-s",
        type=float,
        default=SCAN_DWELL_S,
        help="每个 J1/J2 扫描姿态的最长识别时间（秒，默认 0.1）",
    )
    parser.add_argument(
        "--scan-look-down-max-deg",
        type=float,
        default=SCAN_LOOK_DOWN_MAX_DEG,
        help="每个 J1 方位的第二个 J2 俯视角（默认 +15°）",
    )
    parser.add_argument(
        "--teach-file",
        default=str(DEFAULT_TEACH.relative_to(ROOT)),
        help="示教放置 JSON（teach_joint_pose.py 输出，默认 outputs/teach_joint_pose.json）",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=2.5,
        help="回放示教时移动到每个路点的时长(秒)",
    )
    parser.add_argument(
        "--waypoint-hold",
        type=float,
        default=0.5,
        help="回放示教到达每个路点后停留(秒)",
    )
    args = parser.parse_args(argv)
    if not args.confirm:
        parser.error("真机抓取必须添加 --confirm")
    if not 0.0 <= args.grasp_depth_offset <= 0.10:
        parser.error("--grasp-depth-offset 必须在 0.0 到 0.10 m 之间")
    if args.scan_dwell_s <= 0.0:
        parser.error("--scan-dwell-s 必须大于 0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
