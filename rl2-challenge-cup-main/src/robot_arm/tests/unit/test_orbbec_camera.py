import sys
import types

import numpy as np
import pytest

from robot_arm.vision import OrbbecCamera
from robot_arm.vision import orbbec_camera as camera_module


class FrameSet:
    def __init__(self, color, depth):
        self.color, self.depth = color, depth

    def as_frame_set(self):
        return self

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


def test_fake_orbbec_contract_aligns_and_converts_mm_to_m(monkeypatch):
    width, height = 1280, 720

    class Intrinsic:
        fx, fy, cx, cy = 690.868103, 690.933105, 644.078735, 360.654358

    class Profile:
        def get_intrinsic(self):
            return Intrinsic()

        def get_distortion(self):
            return types.SimpleNamespace(
                k1=0.1, k2=0.2, p1=0.01, p2=0.02, k3=0.3, k4=0.4, k5=0.5, k6=0.6
            )

    class Profiles:
        def get_video_stream_profile(self, *args):
            requests.append(args)
            return Profile()

        def get_default_video_stream_profile(self):
            return Profile()

    class ColorFrame:
        def get_width(self): return width
        def get_height(self): return height

    class DepthFrame(ColorFrame):
        def get_data(self): return np.full((height, width), 500, dtype=np.uint16).tobytes()
        def get_depth_scale(self): return 1.0

    requests = []
    frames = FrameSet(ColorFrame(), DepthFrame())

    class Pipeline:
        def __init__(self):
            self.synced = self.started = self.stopped = False

        def get_stream_profile_list(self, _sensor): return Profiles()
        def enable_frame_sync(self): self.synced = True
        def start(self, _config): self.started = True
        def wait_for_frames(self, _timeout): return frames
        def stop(self): self.stopped = True

    pipeline = Pipeline()

    class Config:
        def enable_stream(self, _profile): pass

    class AlignFilter:
        def __init__(self, align_to_stream):
            assert align_to_stream == "color"
            self.calls = 0

        def process(self, value):
            self.calls += 1
            if self.calls == 1:
                return None
            return value

    sdk = types.SimpleNamespace(
        Pipeline=lambda: pipeline,
        Config=Config,
        AlignFilter=AlignFilter,
        OBSensorType=types.SimpleNamespace(COLOR_SENSOR="rgb", DEPTH_SENSOR="depth"),
        OBStreamType=types.SimpleNamespace(COLOR_STREAM="color"),
        OBFormat=types.SimpleNamespace(RGB="rgb8", MJPG="mjpg", Y16="y16"),
    )
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
    monkeypatch.setattr(
        camera_module,
        "_frame_to_bgr",
        lambda _frame, _sdk: np.zeros((height, width, 3), dtype=np.uint8),
    )
    calibration = np.array(
        [[Intrinsic.fx, 0.0, Intrinsic.cx], [0.0, Intrinsic.fy, Intrinsic.cy], [0.0, 0.0, 1.0]]
    )
    camera = OrbbecCamera(calibrated_camera_matrix=calibration)
    frame = camera.read()
    assert pipeline.synced and pipeline.started
    assert (1280, 800, "y16", 30) in requests
    assert frame.color_bgr.shape == (height, width, 3)
    assert frame.depth_m.shape == (height, width)
    assert float(frame.depth_m[0, 0]) == pytest.approx(0.5)
    assert frame.distortion_coefficients == pytest.approx(
        [0.1, 0.2, 0.01, 0.02, 0.3, 0.4, 0.5, 0.6]
    )
    camera.close()
    assert pipeline.stopped


def test_color_only_mode_does_not_open_depth_stream(monkeypatch):
    class Intrinsic:
        fx, fy, cx, cy = 690.0, 691.0, 640.0, 360.0

    class Distortion:
        k1 = k2 = k3 = k4 = k5 = k6 = p1 = p2 = 0.0

    class Profile:
        def get_intrinsic(self): return Intrinsic()
        def get_distortion(self): return Distortion()

    class Profiles:
        def get_video_stream_profile(self, *_args): return Profile()

    class ColorFrame:
        pass

    class Frames:
        def get_color_frame(self): return ColorFrame()

    class Pipeline:
        def __init__(self): self.sensors = []
        def get_stream_profile_list(self, sensor):
            self.sensors.append(sensor)
            return Profiles()
        def start(self, _config): pass
        def wait_for_frames(self, _timeout): return Frames()
        def stop(self): pass

    pipeline = Pipeline()
    sdk = types.SimpleNamespace(
        Pipeline=lambda: pipeline,
        Config=lambda: types.SimpleNamespace(enable_stream=lambda _profile: None),
        OBSensorType=types.SimpleNamespace(COLOR_SENSOR="rgb", DEPTH_SENSOR="depth"),
        OBFormat=types.SimpleNamespace(RGB="rgb8", MJPG="mjpg"),
    )
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
    monkeypatch.setattr(
        camera_module,
        "_frame_to_bgr",
        lambda _frame, _sdk: np.zeros((720, 1280, 3), dtype=np.uint8),
    )

    frame = OrbbecCamera(color_only=True).read()

    assert pipeline.sensors == ["rgb"]
    assert frame.color_bgr.shape == (720, 1280, 3)
    assert frame.depth_m.shape == (0, 0)
    assert frame.distortion_coefficients == pytest.approx(np.zeros(8))
