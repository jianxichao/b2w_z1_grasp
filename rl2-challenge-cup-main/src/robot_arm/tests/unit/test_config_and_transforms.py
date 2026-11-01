import json
import sys
from copy import deepcopy

import numpy as np
import pytest

from robot_arm.config import _default_root, load_config
from robot_arm.app import _demo_class_map
from robot_arm.vision.target_provider import (
    camera_point_to_base,
    deproject_pixel,
    load_eye_in_hand,
    median_bbox_depth,
)


def test_config_and_calibration_are_valid():
    config = load_config()
    transform = load_eye_in_hand(config.resolve("config/eye_in_hand_calibration.json"))
    assert config.camera["camera"]["model"] == "Orbbec Gemini 2"
    assert config.hardware["joint_lower_rad"] == config.safety["joint_lower_rad"]
    assert transform.shape == (4, 4)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)


def test_robot_config_drives_runtime_forward_kinematics():
    from robot_arm.manipulation import Arx5Kinematics

    config = load_config()
    changed_robot = deepcopy(config.robot)
    changed_robot["kinematics"]["base_correction_xyz_m"][0] += 0.05
    original = Arx5Kinematics.from_config(config.robot)
    changed = Arx5Kinematics.from_config(changed_robot)
    joints = config.motion["home_joints"]
    delta = changed.forward_position(joints) - original.forward_position(joints)
    assert delta == pytest.approx((0.05, 0.0, 0.0))


def test_installed_config_is_found_beside_target_directory(tmp_path, monkeypatch):
    site = tmp_path / "site"
    config_dir = site / "share/robot_arm/config"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(site)])
    monkeypatch.setattr("robot_arm.config.__file__", str(site / "robot_arm/config.py"))
    assert _default_root() == site / "share/robot_arm"


def test_invalid_calibration_is_rejected(tmp_path):
    config = load_config()
    for name in ("robot.yaml", "motion.yaml", "safety.yaml", "simulation.yaml", "hardware.yaml", "camera.yaml", "target_classes.yaml"):
        (tmp_path / "config").mkdir(exist_ok=True)
        (tmp_path / "config" / name).write_bytes((config.root / "config" / name).read_bytes())
    value = dict(config.calibration)
    value["camera_to_eef"] = dict(value["camera_to_eef"])
    value["camera_to_eef"]["rotation_matrix"] = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    (tmp_path / "config/eye_in_hand_calibration.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="刚体变换"):
        load_config(tmp_path)


def test_depth_deprojection_and_eye_in_hand_chain():
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[7:13, 7:13] = 1.25
    value = median_bbox_depth(depth, (2, 2, 18, 18), min_pixels=9)
    assert value == pytest.approx(1.25)
    intrinsic = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    camera = deproject_pixel((10, 10), value, intrinsic)
    corrected = deproject_pixel((10, 10), value, intrinsic, [0, 0, 0, 0, 0])
    assert np.allclose(corrected, camera)
    eef_T_camera = np.eye(4)
    eef_T_camera[:3, 3] = (0.1, 0.0, 0.0)
    base_T_eef = np.eye(4)
    base_T_eef[:3, 3] = (0.2, 0.3, 0.4)
    assert np.allclose(camera_point_to_base(camera, base_T_eef, eef_T_camera), (0.3, 0.3, 1.65))


def test_depth_median_ignores_invalid_values():
    depth = np.array([[0.0, np.nan], [0.5, 0.7]], dtype=np.float32)
    assert median_bbox_depth(depth, (0, 0, 2, 2), center_ratio=1.0, min_pixels=2) == pytest.approx(0.6)
    assert median_bbox_depth(depth, (0, 0, 2, 2), center_ratio=1.0, min_pixels=3) is None


def test_demo_mapping_is_dry_run_only():
    mapping = {"enabled": True, "class_map": {"cup": "science_target"}}
    assert _demo_class_map("dry-run", mapping) == {"cup": "science_target"}
    with pytest.raises(ValueError, match="真机模式禁止"):
        _demo_class_map("real", mapping)
