import time
import threading

import numpy as np
import pytest

from robot_arm.manipulation import Target
from robot_arm.vision import target_provider as target_provider_module
from robot_arm.vision.target_provider import TargetProvider, TrackedTarget, stable_target
from robot_arm.vision.orbbec_camera import RgbdFrame
from robot_arm.vision.yolo_tracker import Detection, draw_depth


def item(track_id, xyz, confidence=0.8):
    return TrackedTarget(track_id, "science_target", confidence, (0, 0, 10, 10), np.asarray(xyz), np.asarray(xyz), time.time())


def test_stable_target_requires_same_stable_id():
    frames = [[item(1, (0.3, 0.0, 0.2))], [item(1, (0.302, 0.0, 0.2))], [item(1, (0.301, 0.0, 0.2))]]
    target = stable_target(frames, 3, 0.01)
    assert target is not None and target.track_id == 1
    assert stable_target([frames[0], [item(2, (0.3, 0, 0.2))], frames[2]], 3, 0.01) is None


class Camera:
    def close(self):
        pass


class Tracker:
    def close(self):
        pass


def test_provider_returns_shared_target_contract(monkeypatch):
    provider = TargetProvider(
        Camera(),
        Tracker(),
        np.eye(4),
        lambda: np.eye(4),
        show=False,
        continuous_tracking=False,
    )
    locked = item(7, (0.3, 0.1, 0.2), 0.9)
    monkeypatch.setattr(provider, "lock_observation", lambda: locked)
    target = provider.lock()
    assert isinstance(target, Target)
    assert target.target_id == "science_target#7"
    assert target.pose.position == (0.3, 0.1, 0.2)


def test_detection_detail_is_optional():
    detection = Detection(3, "science_target", 0.9, (1, 2, 3, 4), "z=0.5m")
    assert detection.center == (2.0, 3.0)


def test_depth_visualization_has_color_and_boxes():
    image = draw_depth(np.ones((8, 8), dtype=np.float32), [Detection(1, "science_target", 0.9, (1, 1, 6, 6))])
    assert image.shape == (8, 8, 3)


def test_observe_applies_acquire_confidence():
    class FrameCamera(Camera):
        def read(self):
            return RgbdFrame(
                np.zeros((20, 20, 3), dtype=np.uint8),
                np.ones((20, 20), dtype=np.float32),
                np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]),
                time.time(),
            )

    class DetectionTracker(Tracker):
        def track(self, _image):
            return [
                Detection(1, "science_target", 0.49, (2, 2, 18, 18)),
                Detection(2, "science_target", 0.51, (2, 2, 18, 18)),
            ]

    provider = TargetProvider(
        FrameCamera(),
        DetectionTracker(),
        np.eye(4),
        lambda: np.eye(4),
        acquire_confidence=0.5,
        show=False,
        continuous_tracking=False,
    )
    assert [target.track_id for target in provider.observe()] == [2]


def test_observe_uses_distortion_from_the_active_camera_profile(monkeypatch):
    """A saved hand-eye calibration must not override a changed RGB-D profile."""

    active_distortion = np.array([0.11, -0.02, 0.003, -0.004, 0.005])
    saved_distortion = np.array([9.0, 9.0, 9.0, 9.0, 9.0])

    class FrameCamera(Camera):
        def read(self):
            return RgbdFrame(
                np.zeros((20, 20, 3), dtype=np.uint8),
                np.ones((20, 20), dtype=np.float32),
                np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]),
                time.time(),
                distortion_coefficients=active_distortion,
            )

    class DetectionTracker(Tracker):
        def track(self, _image):
            return [Detection(1, "science_target", 0.9, (2, 2, 18, 18))]

    observed = {}

    def fake_deproject(_pixel, _depth, _matrix, distortion):
        observed["distortion"] = np.asarray(distortion)
        return np.array((0.2, 0.0, 1.0))

    monkeypatch.setattr(target_provider_module, "deproject_pixel", fake_deproject)
    provider = TargetProvider(
        FrameCamera(),
        DetectionTracker(),
        np.eye(4),
        lambda: np.eye(4),
        distortion=saved_distortion,
        show=False,
        continuous_tracking=False,
    )

    assert len(provider.observe()) == 1
    assert observed["distortion"] == pytest.approx(active_distortion)


def test_background_display_does_not_read_robot_pose():
    class FrameCamera(Camera):
        def read(self):
            return RgbdFrame(
                np.zeros((20, 20, 3), dtype=np.uint8),
                np.ones((20, 20), dtype=np.float32),
                np.eye(3),
                time.time(),
            )

    class DisplayTracker(Tracker):
        def track(self, _image):
            return []

        def show(self, *_args):
            pass

    provider = TargetProvider(
        FrameCamera(),
        DisplayTracker(),
        np.eye(4),
        lambda: (_ for _ in ()).throw(AssertionError("background read robot pose")),
    )
    provider._display_once()


def test_close_joins_monitor_before_closing_devices(monkeypatch):
    order = []
    entered, release = threading.Event(), threading.Event()

    class OrderedCamera(Camera):
        def close(self):
            order.append("camera")

    class OrderedTracker(Tracker):
        def close(self):
            order.append("tracker")

    provider = TargetProvider(OrderedCamera(), OrderedTracker(), np.eye(4), lambda: np.eye(4))
    monkeypatch.setattr(provider, "lock_observation", lambda: item(1, (0.3, 0.0, 0.2)))

    def blocked_display():
        entered.set()
        release.wait()
        order.append("thread")

    monkeypatch.setattr(provider, "_display_once", blocked_display)
    provider.lock()
    assert entered.wait(1.0)
    closer = threading.Thread(target=provider.close)
    closer.start()
    time.sleep(0.02)
    assert order == []
    release.set()
    closer.join(1.0)
    assert not closer.is_alive()
    assert order == ["thread", "camera", "tracker"]
