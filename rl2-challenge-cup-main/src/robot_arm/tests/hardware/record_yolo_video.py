"""Record the Gemini 2 YOLO preview to an exported video file.

This is a manual hardware utility and is not collected by pytest.  It opens
only the camera; it does not connect to CAN or move the arm.
"""

from __future__ import annotations

__test__ = False

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2

ROOT = Path(__file__).resolve().parents[2]
YOLO_DIR = ROOT / "models/yolo"
DEFAULT_WEIGHTS = YOLO_DIR / "best.pt"
DEFAULT_OUTPUT = ROOT / "tests/hardware/outputs/yolo_recording.mp4"
WINDOW = "Gemini 2 YOLO recording"
PREVIEW_SIZE = (960, 540)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_arm.config import load_config  # noqa: E402
from robot_arm.vision import OrbbecCamera, YoloTracker  # noqa: E402


def _target_class(names_path: Path, override: str) -> str:
    if override:
        return override
    if names_path.is_file():
        names = json.loads(names_path.read_text(encoding="utf-8"))
        if names:
            return str(names[0])
    return "rock_obstacle"


def _preview(image):
    height, width = image.shape[:2]
    scale = min(PREVIEW_SIZE[0] / width, PREVIEW_SIZE[1] / height, 1.0)
    if scale >= 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    config = load_config(root)
    camera_cfg = config.camera["camera"]
    weights = Path(args.weights).resolve()
    names = Path(args.names).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target_class = _target_class(names, args.target_class)

    camera = OrbbecCamera(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        fps=int(camera_cfg["fps"]),
        depth_width=int(camera_cfg.get("depth_width", camera_cfg["width"])),
        depth_height=int(camera_cfg.get("depth_height", 800)),
        min_depth_m=float(camera_cfg["min_depth_m"]),
        max_depth_m=float(camera_cfg["max_depth_m"]),
    )
    tracker = YoloTracker(
        weights=str(weights),
        confidence=float(args.confidence),
        device=str(args.device),
        tracker="bytetrack.yaml",
        image_size=int(args.imgsz),
        target_classes={target_class},
    )
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.output_fps),
        (int(camera_cfg["width"]), int(camera_cfg["height"])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 MP4 视频：{output}")

    started = time.monotonic()
    frames_written = 0
    try:
        camera.start()
        if not args.headless:
            flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0)
            cv2.namedWindow(WINDOW, flags)
            cv2.resizeWindow(WINDOW, *PREVIEW_SIZE)
        print(f"开始录制：{output}")
        print(f"目标类别：{target_class}；按 q 结束录制。")
        while args.duration <= 0.0 or time.monotonic() - started < args.duration:
            try:
                frame = camera.read()
            except RuntimeError as exc:
                if "RGB-D 对齐失败" in str(exc):
                    print(f"相机瞬时丢帧，继续录制：{exc}")
                    continue
                raise
            detections = tracker.track(frame.color_bgr)
            display = tracker.draw(frame.color_bgr, detections)
            elapsed = time.monotonic() - started
            cv2.putText(
                display,
                f"REC {elapsed:07.1f}s  frames={frames_written}",
                (16, display.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(display)
            frames_written += 1
            if not args.headless:
                cv2.imshow(WINDOW, _preview(display))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("收到 Ctrl+C，结束录制。")
    finally:
        writer.release()
        camera.close()
        tracker.close()

    if frames_written == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("没有成功写入视频帧")
    print(
        f"视频已导出：{output} "
        f"({frames_written} 帧，编码帧率 {float(args.output_fps):.1f} FPS)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="robot_arm 根目录")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO 权重")
    parser.add_argument("--names", default=str(YOLO_DIR / "best.names.json"), help="类别名 JSON")
    parser.add_argument("--target-class", default="", help="覆盖目标类别")
    parser.add_argument("--confidence", type=float, default=0.65, help="检测置信度阈值")
    parser.add_argument("--device", default="auto", help="YOLO 设备：auto/cpu/0")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="导出 MP4 路径")
    parser.add_argument("--output-fps", type=float, default=10.0, help="导出视频帧率")
    parser.add_argument("--duration", type=float, default=0.0, help="录制秒数；0 表示直到 q/Ctrl+C")
    parser.add_argument("--headless", action="store_true", help="不显示窗口，仅录制")
    args = parser.parse_args(argv)
    if args.output_fps <= 0.0:
        parser.error("--output-fps 必须大于 0")
    if args.duration < 0.0:
        parser.error("--duration 不能小于 0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
