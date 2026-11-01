import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "hardware/test_eye_in_hand_calibration.py"
SPEC = importlib.util.spec_from_file_location("eye_in_hand_calibration_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def make_transform(rvec, translation):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    return calibration._transform(rotation, np.asarray(translation, dtype=float))


def test_multiview_solver_returns_camera_to_eef_transform():
    expected = make_transform((0.20, -0.15, 0.30), (0.04, -0.02, 0.08))
    base_T_board = make_transform((0.10, 0.05, -0.20), (0.50, 0.10, 0.15))
    samples = []
    for index in range(50):
        angle = index * 0.37
        base_T_eef = make_transform(
            (
                0.5 * np.sin(angle),
                0.4 * np.cos(angle * 0.7),
                0.6 * np.sin(angle * 0.4 + 0.2),
            ),
            (
                0.25 + 0.08 * np.cos(angle),
                0.08 * np.sin(angle * 0.8),
                0.25 + 0.06 * np.cos(angle * 0.5),
            ),
        )
        camera_T_board = np.linalg.inv(base_T_eef @ expected) @ base_T_board
        samples.append(
            calibration.Sample(base_T_eef, camera_T_board, (0.0,) * 6, 0.1)
        )

    method, actual, metrics = calibration.solve_hand_eye(samples)

    assert method in metrics
    assert np.allclose(actual, expected, atol=1e-8)
    assert metrics[method]["board_translation_rms_m"] < 1e-8
    assert metrics[method]["board_rotation_rms_deg"] < 1e-5


def test_result_payload_keeps_project_transform_direction():
    transform = make_transform((0.10, 0.20, -0.10), (0.01, 0.02, 0.03))
    sample = calibration.Sample(np.eye(4), np.eye(4), (0.0,) * 6, 0.2)
    metrics = {
        "Park": {
            "board_translation_rms_m": 0.001,
            "board_rotation_rms_deg": 0.2,
        }
    }

    result = calibration._result_payload(
        [sample] * 15, "Park", transform, metrics, np.eye(3), np.zeros(8)
    )

    assert result["transform_direction"] == "camera_to_eef"
    assert result["camera_to_eef"]["translation"] == [0.01, 0.02, 0.03]
    assert result["num_samples"] == 15
