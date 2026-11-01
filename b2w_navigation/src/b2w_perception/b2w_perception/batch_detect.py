"""Offline image/video detection tool for development machines.

This tool runs on the development PC, not on the robot. It batch-processes
image/video files with the Ultralytics YOLO model (which needs PyTorch) and
produces unified JSON evidence plus annotated media. The online robot node
uses ``yolo_target_perception_node.py`` with ONNX Runtime instead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config_io import load_target_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="samples/images")
    parser.add_argument("--video-dir", default="samples/videos")
    parser.add_argument("--output-dir", default="output")
    # Offline tools load the original Ultralytics .pt checkpoint; the online
    # node uses the exported best.onnx instead.
    parser.add_argument("--model", default="assets/models/yolo11n.pt")
    parser.add_argument("--confidence", default="0.25")
    parser.add_argument("--target-config", default="src/b2w_perception/config/target_classes.yaml")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--frame-id", default="camera_color_optical_frame")
    parser.add_argument("--frame-stride", default="5")
    parser.add_argument("--max-frames", default="180")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--disable-tracking", action="store_true")
    parser.add_argument("--enable-demo-mapping", action="store_true")
    parser.add_argument("--preprocess-mode", choices=("auto", "resize", "letterbox"), default=None)
    parser.add_argument("--output-width", default=None)
    parser.add_argument("--output-height", default=None)
    parser.add_argument("--post-mapping-nms-iou", default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.confidence = float(args.confidence)
    output = Path(args.output_dir)
    image_output = output / "images"
    video_output = output / "videos"
    image_output.mkdir(parents=True, exist_ok=True)
    video_output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    # Load the model once.  The old implementation spawned a fresh Python
    # process (and reloaded the 5 MB checkpoint) for every image, making a
    # 100-frame report take several minutes on a CPU and leaving partial
    # evidence when interrupted.
    import cv2
    from .target_detector_node import _build_yolo_detector, _preprocess
    from .visualization import draw_detections

    config = load_target_config(args.target_config)
    detector = _build_yolo_detector(args, config)

    # Pass 1: detect every supported image and record its JSON result.
    image_index = 0
    for image in sorted(Path(args.image_dir).glob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        name = image.stem
        json_path = image_output / f"{name}.json"
        timestamp = time.time()
        source = cv2.imread(str(image))
        if source is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        processed, _ = _preprocess(source, args, config)
        detections = detector.detect_image(processed, timestamp=timestamp)
        cv2.imwrite(str(image_output / f"{name}_annotated.jpg"), draw_detections(processed, detections))
        _write_json(json_path, [item.to_dict() for item in detections])
        records.append(
            {
                "source": str(image),
                "media_type": "image",
                # Image directories represent an ordered frame sequence;
                # preserve that index so yolo_eval does not collapse all
                # detections onto frame 0.
                "frame_index": image_index,
                "timestamp": timestamp,
                "preprocess": _image_preprocess_metadata(image, args),
                "detections": _read_json(json_path),
            }
        )
        image_index += 1

    # Pass 2: run video detection/tracking and collect per-frame records.
    for video in sorted(Path(args.video_dir).glob("*")):
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        name = video.stem
        json_path = video_output / f"{name}.json"
        event_path = video_output / f"{name}_events.json"
        command = _common_command(args) + [
            "--video", str(video),
            "--frame-stride", args.frame_stride, "--max-frames", args.max_frames,
            "--out-video", str(video_output / f"{name}_annotated.mp4"),
            "--out-json", str(json_path), "--out-events-json", str(event_path),
        ]
        if args.disable_tracking:
            command.append("--disable-tracking")
        _run(command)
        for item in _read_json(json_path):
            records.append({"source": str(video), "media_type": "video", **item})
        events.extend(_read_json(event_path))

    _write_json(output / "detections.json", records)
    _write_json(output / "events.json", events)
    print(f"saved {len(records)} media records to {output / 'detections.json'}")
    print(f"saved {len(events)} perception events to {output / 'events.json'}")
    return 0


def _common_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, "-m", "b2w_perception.target_detector_node",
        "--detector", "yolo", "--model", args.model,
        "--confidence", args.confidence, "--target-config", args.target_config,
        "--device", args.device, "--tracker", args.tracker,
    ]
    if args.enable_demo_mapping:
        command.append("--enable-demo-mapping")
    for option, value in (
        ("--preprocess-mode", args.preprocess_mode),
        ("--output-width", args.output_width),
        ("--output-height", args.output_height),
        ("--post-mapping-nms-iou", args.post_mapping_nms_iou),
    ):
        if value is not None:
            command.extend([option, str(value)])
    return command


def _image_preprocess_metadata(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Batch image metadata requires opencv-python.") from exc
    from .yolo_detector_adapter import preprocess_image

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    config = load_target_config(args.target_config)["preprocess"]
    _, metadata = preprocess_image(
        image,
        output_width=int(args.output_width or config["output_width"]),
        output_height=int(args.output_height or config["output_height"]),
        mode=args.preprocess_mode or str(config["mode"]),
        letterbox_value=int(config["letterbox_value"]),
    )
    return metadata


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
