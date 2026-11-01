"""Run Everang grasping and continuously record the complete camera stream."""
from __future__ import annotations

__test__ = False

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import test_everang as everang

DEFAULT_OUTPUT = everang.ROOT / "tests/hardware/outputs/everang_grasp_recording.mp4"


class ContinuousRecorder:
    def __init__(self, output: Path, fps: float, max_width: int) -> None:
        self.output, self.fps, self.max_width = output, float(fps), int(max_width)
        self.writer: cv2.VideoWriter | None = None
        self.frames_written = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._pending_annotated: Any | None = None

    def set_annotated(self, frame: Any) -> None:
        with self._lock:
            self._pending_annotated = frame.copy()

    def write_camera_frame(self, frame: Any) -> None:
        with self._lock:
            output_frame = (
                self._pending_annotated
                if self._pending_annotated is not None
                else frame.copy()
            )
            self._pending_annotated = None
            elapsed = time.monotonic() - self.started
            cv2.putText(
                output_frame, f"REC {elapsed:07.1f}s",
                (max(12, output_frame.shape[1] - 190), output_frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
            )
            encode_frame = self._downscale(output_frame)
            if self.writer is None:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                height, width = encode_frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.output), cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps, (width, height),
                )
                if not self.writer.isOpened():
                    self.writer.release()
                    self.writer = None
                    raise RuntimeError(f"无法创建抓取录像：{self.output}")
                print(f"连续录像开始：{self.output} ({width}x{height}, {self.fps:.1f} FPS)")
            self.writer.write(encode_frame)
            self.frames_written += 1

    def _downscale(self, frame: Any) -> Any:
        height, width = frame.shape[:2]
        if width <= self.max_width:
            return frame
        output_width = self.max_width - self.max_width % 2
        output_height = round(height * output_width / width)
        output_height -= output_height % 2
        return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        with self._lock:
            if self.writer is None:
                print("录像未生成：相机没有成功输出任何帧。")
                return
            self.writer.release()
            self.writer = None
            print(
                f"完整抓取录像已导出：{self.output} "
                f"({self.frames_written} 帧，{self.fps:.1f} FPS)"
            )


class ContinuousRecordingCamera:
    """Use one background reader for both recording and grasp detection."""
    def __init__(self, camera: Any, recorder: ContinuousRecorder) -> None:
        self._camera, self._recorder = camera, recorder
        self._condition = threading.Condition()
        self._latest: Any | None = None
        self._sequence = self._last_delivered = 0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._camera.start()
        self._thread = threading.Thread(
            target=self._capture_loop, name="everang-camera-recorder", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._camera.read()
                self._recorder.write_camera_frame(frame.color_bgr)
                with self._condition:
                    self._latest = frame
                    self._sequence += 1
                    self._condition.notify_all()
            except RuntimeError as exc:
                if "RGB-D 对齐失败" in str(exc):
                    print(f"录像相机瞬时丢帧，继续采集：{exc}")
                    continue
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                return
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                return

    def read(self) -> Any:
        self.start()
        with self._condition:
            deadline = time.monotonic() + 3.0
            while self._sequence <= self._last_delivered and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("连续录像相机 3 秒内没有新帧")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError(f"连续录像相机失败：{self._error}") from self._error
            self._last_delivered = self._sequence
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._camera.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--record-output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--record-width", type=int, default=960)
    own_args, everang_args = parser.parse_known_args(argv)
    if own_args.record_fps <= 0.0:
        parser.error("--record-fps 必须大于 0")
    if own_args.record_width < 320:
        parser.error("--record-width 必须不小于 320")

    output = Path(own_args.record_output)
    if not output.is_absolute():
        output = everang.ROOT / output
    recorder = ContinuousRecorder(output.resolve(), own_args.record_fps, own_args.record_width)
    original_camera_class = everang.OrbbecCamera
    original_draw_preview = everang._draw_preview

    def camera_factory(*args: Any, **kwargs: Any) -> ContinuousRecordingCamera:
        return ContinuousRecordingCamera(original_camera_class(*args, **kwargs), recorder)

    def draw_and_record(*args: Any, **kwargs: Any) -> Any:
        frame = original_draw_preview(*args, **kwargs)
        recorder.set_annotated(frame)
        return frame

    everang.OrbbecCamera = camera_factory
    everang._draw_preview = draw_and_record
    try:
        return everang.main(everang_args)
    finally:
        everang.OrbbecCamera = original_camera_class
        everang._draw_preview = original_draw_preview
        recorder.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
