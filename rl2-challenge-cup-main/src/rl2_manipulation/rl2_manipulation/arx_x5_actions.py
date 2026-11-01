"""Validated ARX X5 fixed-action templates shared by ROS and MuJoCo adapters.

The module deliberately has no ROS or MuJoCo import. This keeps the command
contract testable on a development computer and lets the integration owner map
the same validated command to either ``RobotCmd`` or ``data.ctrl``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml


REQUIRED_SEQUENCE = ("home", "pre_grasp", "grasp", "lift", "drop_to_bin")
VALID_GRIPPER_TIMINGS = {"before_motion", "after_motion"}


class ActionConfigError(ValueError):
    """Raised when the action configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    axis: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    lower: float
    upper: float


@dataclass(frozen=True)
class GripperSpec:
    command_closed: float
    command_open: float
    opening_closed_m: float
    opening_open_m: float

    def command_to_opening(self, command: float) -> float:
        """Linearly map the documented 0..5 command to the 0..0.08 m gap."""

        if not self.command_closed <= command <= self.command_open:
            raise ActionConfigError(
                f"gripper command {command} is outside "
                f"[{self.command_closed}, {self.command_open}]"
            )
        command_span = self.command_open - self.command_closed
        opening_span = self.opening_open_m - self.opening_closed_m
        fraction = (command - self.command_closed) / command_span
        return self.opening_closed_m + fraction * opening_span


@dataclass(frozen=True)
class ActionTemplate:
    name: str
    mode: int
    joint_pos: Tuple[float, float, float, float, float, float]
    gripper: float
    gripper_timing: str
    gripper_duration_s: float
    duration_s: float
    source_classification: str
    description: str

    def robot_cmd_payload(self) -> Dict[str, Any]:
        """Return fields compatible with the supplied ``RobotCmd.msg``."""

        return {
            "end_pos": [0.0] * 6,
            "joint_pos": list(self.joint_pos),
            "gripper": self.gripper,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ArxX5ActionConfig:
    model: str
    command_topic: str
    status_topic: str
    modes: Mapping[str, int]
    joints: Tuple[JointSpec, ...]
    gripper: GripperSpec
    sequence: Tuple[str, ...]
    templates: Mapping[str, ActionTemplate]
    mujoco_actuator_names: Tuple[str, ...]
    source_path: Path

    def action(self, name: str) -> ActionTemplate:
        try:
            return self.templates[name]
        except KeyError as exc:
            choices = ", ".join(self.sequence)
            raise ActionConfigError(f"unknown action {name!r}; choose from {choices}") from exc

    def ordered_actions(self, names: Optional[Iterable[str]] = None) -> Tuple[ActionTemplate, ...]:
        selected = self.sequence if names is None else tuple(names)
        return tuple(self.action(name) for name in selected)

    def mujoco_ctrl(self, action: ActionTemplate) -> Tuple[float, ...]:
        """Return controls in the documented MJCF actuator order."""

        return (*action.joint_pos, self.gripper.command_to_opening(action.gripper))

    def mujoco_ctrl_by_name(self, action: ActionTemplate) -> Mapping[str, float]:
        """Return actuator-name controls without assuming global indices."""

        return dict(zip(self.mujoco_actuator_names, self.mujoco_ctrl(action)))

    def write_mujoco_ctrl(
        self,
        action: ActionTemplate,
        actuator_name_to_id: Mapping[str, int],
        ctrl: Any,
    ) -> None:
        """Write arm controls by actuator name into a complete-robot ctrl array.

        A complete model can place chassis actuators before or after the arm, so
        fixed ``data.ctrl[0:7]`` indices are unsafe outside the standalone model.
        """

        missing = [
            name for name in self.mujoco_actuator_names if name not in actuator_name_to_id
        ]
        if missing:
            raise ActionConfigError(
                "MuJoCo model is missing ARX X5 actuators: " + ", ".join(missing)
            )

        for name, value in self.mujoco_ctrl_by_name(action).items():
            actuator_id = actuator_name_to_id[name]
            if isinstance(actuator_id, bool) or not isinstance(actuator_id, int):
                raise ActionConfigError(f"actuator id for {name} must be an integer")
            if actuator_id < 0 or actuator_id >= len(ctrl):
                raise ActionConfigError(
                    f"actuator id {actuator_id} for {name} is outside ctrl length {len(ctrl)}"
                )
            ctrl[actuator_id] = value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionConfigError(f"{field} must be numeric")
    return float(value)


def _vector(value: Any, length: int, field: str) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ActionConfigError(f"{field} must be a sequence of length {length}")
    if len(value) != length:
        raise ActionConfigError(f"{field} must contain {length} values, got {len(value)}")
    return tuple(_number(item, f"{field}[{index}]") for index, item in enumerate(value))


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionConfigError(f"{field} must be a mapping")
    return value


def default_config_path() -> Path:
    """Resolve the source-tree or installed ROS package configuration."""

    override = os.environ.get("RL2_ARX_X5_CONFIG")
    if override:
        return Path(override).expanduser().resolve()

    source_path = Path(__file__).resolve().parents[1] / "config" / "grasp_poses.yaml"
    if source_path.exists():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("rl2_manipulation")) / "config" / "grasp_poses.yaml"
    except (ImportError, LookupError) as exc:
        raise ActionConfigError(
            "cannot locate grasp_poses.yaml; pass --config or set RL2_ARX_X5_CONFIG"
        ) from exc


def load_action_config(path: Optional[os.PathLike[str] | str] = None) -> ArxX5ActionConfig:
    """Load and fully validate the ARX X5 action configuration."""

    config_path = Path(path) if path is not None else default_config_path()
    config_path = config_path.expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ActionConfigError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ActionConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    root = _mapping(raw, "root")
    if root.get("schema_version") != 1:
        raise ActionConfigError("schema_version must be 1")

    arm = _mapping(root.get("arm"), "arm")
    if arm.get("arm_joint_count") != 6:
        raise ActionConfigError("arm.arm_joint_count must be 6")

    raw_modes = _mapping(arm.get("modes"), "arm.modes")
    modes = {str(name): int(_number(value, f"arm.modes.{name}")) for name, value in raw_modes.items()}
    if modes.get("joint_position_control") != 5:
        raise ActionConfigError("joint_position_control mode must be 5 for the supplied SDK")

    raw_joints = arm.get("joints")
    if not isinstance(raw_joints, list) or len(raw_joints) != 6:
        raise ActionConfigError("arm.joints must contain exactly six joint specifications")

    joints = []
    for index, raw_joint in enumerate(raw_joints):
        field = f"arm.joints[{index}]"
        joint = _mapping(raw_joint, field)
        lower = _number(joint.get("lower"), f"{field}.lower")
        upper = _number(joint.get("upper"), f"{field}.upper")
        if lower >= upper:
            raise ActionConfigError(f"{field} has invalid limits [{lower}, {upper}]")
        joints.append(
            JointSpec(
                name=str(joint.get("name", "")),
                parent=str(joint.get("parent", "")),
                child=str(joint.get("child", "")),
                axis=_vector(joint.get("axis"), 3, f"{field}.axis"),
                origin_xyz=_vector(joint.get("origin_xyz"), 3, f"{field}.origin_xyz"),
                origin_rpy=_vector(joint.get("origin_rpy"), 3, f"{field}.origin_rpy"),
                lower=lower,
                upper=upper,
            )
        )

    expected_joint_names = tuple(f"joint{index}" for index in range(1, 7))
    actual_joint_names = tuple(joint.name for joint in joints)
    if actual_joint_names != expected_joint_names:
        raise ActionConfigError(
            f"joint names/order must be {expected_joint_names}, got {actual_joint_names}"
        )

    raw_gripper = _mapping(arm.get("gripper"), "arm.gripper")
    gripper = GripperSpec(
        command_closed=_number(raw_gripper.get("command_closed"), "arm.gripper.command_closed"),
        command_open=_number(raw_gripper.get("command_open"), "arm.gripper.command_open"),
        opening_closed_m=_number(
            raw_gripper.get("opening_closed_m"), "arm.gripper.opening_closed_m"
        ),
        opening_open_m=_number(raw_gripper.get("opening_open_m"), "arm.gripper.opening_open_m"),
    )
    if gripper.command_closed >= gripper.command_open:
        raise ActionConfigError("gripper command range must be increasing")
    if gripper.opening_closed_m >= gripper.opening_open_m:
        raise ActionConfigError("gripper opening range must be increasing")

    raw_sequence = root.get("sequence")
    if not isinstance(raw_sequence, list):
        raise ActionConfigError("sequence must be a list")
    sequence = tuple(str(name) for name in raw_sequence)
    if sequence != REQUIRED_SEQUENCE:
        raise ActionConfigError(f"sequence must be exactly {REQUIRED_SEQUENCE}, got {sequence}")

    raw_templates = _mapping(root.get("templates"), "templates")
    templates: Dict[str, ActionTemplate] = {}
    allowed_modes = set(modes.values())
    for name in sequence:
        field = f"templates.{name}"
        data = _mapping(raw_templates.get(name), field)
        mode = int(_number(data.get("mode"), f"{field}.mode"))
        if mode not in allowed_modes:
            raise ActionConfigError(f"{field}.mode {mode} is not a documented SDK mode")
        joint_pos = _vector(data.get("joint_pos"), 6, f"{field}.joint_pos")
        if mode == modes["joint_position_control"]:
            for joint, target in zip(joints, joint_pos):
                if not joint.lower <= target <= joint.upper:
                    raise ActionConfigError(
                        f"{field}: {joint.name} target {target} is outside "
                        f"[{joint.lower}, {joint.upper}]"
                    )
        gripper_command = _number(data.get("gripper"), f"{field}.gripper")
        gripper.command_to_opening(gripper_command)
        timing = str(data.get("gripper_timing", ""))
        if timing not in VALID_GRIPPER_TIMINGS:
            raise ActionConfigError(
                f"{field}.gripper_timing must be one of {sorted(VALID_GRIPPER_TIMINGS)}"
            )
        duration_s = _number(data.get("duration_s"), f"{field}.duration_s")
        if duration_s <= 0.0:
            raise ActionConfigError(f"{field}.duration_s must be positive")
        gripper_duration_s = _number(
            data.get("gripper_duration_s", 0.45),
            f"{field}.gripper_duration_s",
        )
        if gripper_duration_s <= 0.0:
            raise ActionConfigError(f"{field}.gripper_duration_s must be positive")

        templates[name] = ActionTemplate(
            name=name,
            mode=mode,
            joint_pos=joint_pos,  # type: ignore[arg-type]
            gripper=gripper_command,
            gripper_timing=timing,
            gripper_duration_s=gripper_duration_s,
            duration_s=duration_s,
            source_classification=str(data.get("source_classification", "")),
            description=str(data.get("description", "")),
        )

    mujoco = _mapping(root.get("mujoco"), "mujoco")
    actuator_names_raw = mujoco.get("actuator_names")
    if not isinstance(actuator_names_raw, list) or len(actuator_names_raw) != 7:
        raise ActionConfigError("mujoco.actuator_names must contain six joints plus gripper")
    actuator_names = tuple(str(name) for name in actuator_names_raw)

    return ArxX5ActionConfig(
        model=str(arm.get("model", "")),
        command_topic=str(arm.get("command_topic", "")),
        status_topic=str(arm.get("status_topic", "")),
        modes=modes,
        joints=tuple(joints),
        gripper=gripper,
        sequence=sequence,
        templates=templates,
        mujoco_actuator_names=actuator_names,
        source_path=config_path,
    )


def _json_record(config: ArxX5ActionConfig, action: ActionTemplate) -> Dict[str, Any]:
    return {
        "action": action.name,
        "duration_s": action.duration_s,
        "gripper_timing": action.gripper_timing,
        "gripper_duration_s": action.gripper_duration_s,
        "robot_cmd": action.robot_cmd_payload(),
        "mujoco_actuator_names": list(config.mujoco_actuator_names),
        "mujoco_ctrl": list(config.mujoco_ctrl(action)),
        "source_classification": action.source_classification,
    }


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and print ARX X5 fixed action templates")
    parser.add_argument("--config", type=Path, help="path to grasp_poses.yaml")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--action", choices=REQUIRED_SEQUENCE, help="print one action")
    selection.add_argument("--sequence", action="store_true", help="print the complete five-action sequence")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    config = load_action_config(args.config)
    names = (args.action,) if args.action else config.sequence
    actions = config.ordered_actions(names)

    if args.json:
        print(json.dumps([_json_record(config, action) for action in actions], ensure_ascii=False, indent=2))
        return 0

    print(
        f"VALIDATED model={config.model} joints={len(config.joints)} "
        f"actions={len(actions)} config={config.source_path}"
    )
    for index, action in enumerate(actions, start=1):
        joint_text = ", ".join(f"{value:.3f}" for value in action.joint_pos)
        ctrl_text = ", ".join(f"{value:.3f}" for value in config.mujoco_ctrl(action))
        print(
            f"{index:02d} {action.name:<12} mode={action.mode} "
            f"joint_pos=[{joint_text}] gripper={action.gripper:.2f} "
            f"timing={action.gripper_timing} duration={action.duration_s:.1f}s "
            f"gripper_duration={action.gripper_duration_s:.2f}s"
        )
        print(f"   mujoco_ctrl=[{ctrl_text}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
