from pathlib import Path

import pytest


try:
    import mujoco
except ImportError:
    mujoco = None


def test_models_load():
    if mujoco is None:
        pytest.skip("未安装 MuJoCo")
    root = Path(__file__).resolve().parents[2]
    for path in (root / "mjcf/scenes/arx_x5_standalone.xml", root / "mjcf/scenes/arx_x5_contact_grasp.xml"):
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.nq >= 8


def test_joint_limit_tool_model_loads():
    if mujoco is None:
        pytest.skip("未安装 MuJoCo")
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "mjcf/scenes/arx_x5_standalone.xml"))
    assert model.site("gripper_grasp_center").id >= 0
