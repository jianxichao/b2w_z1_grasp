"""Compile the simplified ARX X5 MJCF and execute all five fixed templates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .arx_x5_actions import default_config_path, load_action_config


def default_model_path() -> Path:
    source_path = Path(__file__).resolve().parents[1] / "mjcf" / "arx_x5_simplified.xml"
    if source_path.exists():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("rl2_manipulation")) / "mjcf" / "arx_x5_simplified.xml"
    except (ImportError, LookupError) as exc:
        raise RuntimeError("cannot locate arx_x5_simplified.xml; pass --model") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="path to arx_x5_simplified.xml")
    parser.add_argument("--config", type=Path, help="path to grasp_poses.yaml")
    parser.add_argument("--joint-tolerance", type=float, default=0.03)
    args = parser.parse_args(argv)

    try:
        import mujoco
    except ImportError as exc:
        parser.error("MuJoCo Python package is missing; install with: pip install mujoco")
        raise AssertionError("unreachable") from exc

    model_path = args.model if args.model is not None else default_model_path()
    config_path = args.config if args.config is not None else default_config_path()
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    config = load_action_config(config_path)

    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    )
    expected_joints = tuple(joint.name for joint in config.joints) + ("gripper_opening",)
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if joint_names != expected_joints:
        raise RuntimeError(f"joint contract mismatch: expected {expected_joints}, got {joint_names}")
    if actuator_names != config.mujoco_actuator_names:
        raise RuntimeError(
            f"actuator contract mismatch: expected {config.mujoco_actuator_names}, got {actuator_names}"
        )

    print(f"MUJOCO_VERSION={mujoco.__version__}")
    print(f"MODEL_OK nq={model.nq} nv={model.nv} njnt={model.njnt} nu={model.nu}")
    print("JOINTS=" + ",".join(joint_names))
    print("ACTUATORS=" + ",".join(actuator_names))

    mujoco.mj_resetDataKeyframe(model, data, 0)
    passed = True
    for action in config.ordered_actions():
        target_ctrl = config.mujoco_ctrl(action)
        data.ctrl[:] = target_ctrl
        steps = int(action.duration_s / model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(model, data)

        joint_error = max(abs(data.qpos[index] - action.joint_pos[index]) for index in range(6))
        gripper_error = abs(data.qpos[6] - target_ctrl[6])
        finite = bool(all(abs(value) < 1e6 for value in data.qpos))
        action_passed = finite and joint_error <= args.joint_tolerance
        passed = passed and action_passed
        label = "ACTION_OK" if action_passed else "ACTION_FAILED"
        print(
            f"{label} name={action.name} steps={steps} "
            f"max_joint_error={joint_error:.5f} "
            f"gripper_error={gripper_error:.5f} finite={finite}"
        )

    print("RESULT=" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
