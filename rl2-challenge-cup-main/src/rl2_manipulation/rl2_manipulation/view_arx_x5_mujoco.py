"""Open an interactive MuJoCo viewer and animate the five ARX X5 templates."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Optional, Sequence

from .arx_x5_actions import default_config_path, load_action_config
from .validate_arx_x5_mujoco import default_model_path


def _step_and_sync(mujoco, model, data, viewer, sleep_s: float, orbit_step: float) -> bool:
    if not viewer.is_running():
        return False
    started = time.perf_counter()
    mujoco.mj_step(model, data)
    if orbit_step:
        viewer.cam.azimuth = (viewer.cam.azimuth + orbit_step) % 360.0
    viewer.sync()
    remaining = sleep_s - (time.perf_counter() - started)
    if remaining > 0.0:
        time.sleep(remaining)
    return True


def _smooth_fraction(step: int, steps: int) -> float:
    fraction = step / max(steps, 1)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _drive_gripper(
    mujoco,
    model,
    data,
    viewer,
    target_opening: float,
    duration_s: float,
    sleep_s: float,
    orbit_step: float,
) -> bool:
    """Animate the gripper actuator instead of snapping it like a switch."""

    start_opening = float(data.ctrl[6])
    steps = max(1, int(duration_s / model.opt.timestep))
    for step in range(1, steps + 1):
        smooth = _smooth_fraction(step, steps)
        data.ctrl[6] = start_opening + smooth * (target_opening - start_opening)
        if not _step_and_sync(mujoco, model, data, viewer, sleep_s, orbit_step):
            return False
    data.ctrl[6] = target_opening
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="path to arx_x5_simplified.xml")
    parser.add_argument("--config", type=Path, help="path to grasp_poses.yaml")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--hold", type=float, default=0.8, help="seconds to hold each pose")
    parser.add_argument("--repeat", action="store_true", help="repeat until the window closes")
    parser.add_argument(
        "--orbit-speed",
        type=float,
        default=0.0,
        help="camera orbit speed in degrees per simulated second; use 0 to disable",
    )
    args = parser.parse_args(argv)
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.hold < 0.0:
        parser.error("--hold cannot be negative")

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        parser.error("MuJoCo is missing; run the ARX X5 setup script first")
        raise AssertionError("unreachable") from exc

    model_path = (args.model or default_model_path()).resolve()
    config_path = (args.config or default_config_path()).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    config = load_action_config(config_path)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    sleep_s = model.opt.timestep / args.speed
    hold_steps = max(1, int(args.hold / model.opt.timestep))
    orbit_step = args.orbit_speed * model.opt.timestep

    print(f"MODEL={model_path}")
    print(f"CONFIG={config_path}")
    print(f"CAMERA_ORBIT_DEG_PER_S={args.orbit_speed:.1f}")
    print("Close the viewer window or press Ctrl+C in this terminal to stop.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = (-0.05, 0.0, 0.27)
            viewer.cam.distance = 0.92
            viewer.cam.azimuth = 115.0
            viewer.cam.elevation = -16.0

            while viewer.is_running():
                for action in config.ordered_actions():
                    if not viewer.is_running():
                        return 0
                    start_ctrl = data.ctrl.copy()
                    target_ctrl = config.mujoco_ctrl(action)
                    motion_steps = max(1, int(action.duration_s / model.opt.timestep))
                    print(
                        f"ACTION_START name={action.name} mode={action.mode} "
                        f"gripper={action.gripper:.1f} timing={action.gripper_timing} "
                        f"gripper_duration={action.gripper_duration_s:.2f}s"
                    )

                    if action.gripper_timing == "before_motion":
                        if not _drive_gripper(
                            mujoco,
                            model,
                            data,
                            viewer,
                            target_ctrl[6],
                            action.gripper_duration_s,
                            sleep_s,
                            orbit_step,
                        ):
                            return 0

                    for step in range(1, motion_steps + 1):
                        smooth = _smooth_fraction(step, motion_steps)
                        for index in range(6):
                            data.ctrl[index] = start_ctrl[index] + smooth * (
                                target_ctrl[index] - start_ctrl[index]
                            )
                        if action.gripper_timing == "after_motion":
                            data.ctrl[6] = start_ctrl[6]
                        if not _step_and_sync(
                            mujoco, model, data, viewer, sleep_s, orbit_step
                        ):
                            return 0

                    if action.gripper_timing == "after_motion":
                        if not _drive_gripper(
                            mujoco,
                            model,
                            data,
                            viewer,
                            target_ctrl[6],
                            action.gripper_duration_s,
                            sleep_s,
                            orbit_step,
                        ):
                            return 0

                    data.ctrl[:6] = target_ctrl[:6]
                    data.ctrl[6] = target_ctrl[6]
                    for _ in range(hold_steps):
                        if not _step_and_sync(
                            mujoco, model, data, viewer, sleep_s, orbit_step
                        ):
                            return 0
                    print(f"ACTION_DONE  name={action.name}")

                if args.repeat:
                    print("SEQUENCE_DONE restarting=true")
                    continue

                print("SEQUENCE_DONE. Inspect the model, then close the viewer window.")
                while viewer.is_running():
                    if not _step_and_sync(
                        mujoco, model, data, viewer, sleep_s, orbit_step
                    ):
                        return 0
                return 0
    except KeyboardInterrupt:
        print("Viewer stopped by user.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
