from pathlib import Path
import tempfile
import unittest

import yaml

from rl2_manipulation.arx_x5_actions import (
    ActionConfigError,
    REQUIRED_SEQUENCE,
    load_action_config,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config" / "grasp_poses.yaml"


class ArxX5ActionConfigTests(unittest.TestCase):
    def test_required_sequence_and_six_joint_contract(self) -> None:
        config = load_action_config(CONFIG_PATH)

        self.assertEqual(config.sequence, REQUIRED_SEQUENCE)
        self.assertEqual(
            [joint.name for joint in config.joints], [f"joint{i}" for i in range(1, 7)]
        )
        self.assertEqual(len(config.mujoco_actuator_names), 7)

    def test_every_template_is_joint_mode_and_within_documented_limits(self) -> None:
        config = load_action_config(CONFIG_PATH)
        joint_mode = config.modes["joint_position_control"]

        for action in config.ordered_actions():
            self.assertEqual(action.mode, joint_mode)
            self.assertEqual(action.mode, 5)
            self.assertEqual(len(action.joint_pos), 6)
            for target, joint in zip(action.joint_pos, config.joints):
                self.assertLessEqual(joint.lower, target)
                self.assertLessEqual(target, joint.upper)

    def test_gripper_state_and_mujoco_mapping(self) -> None:
        config = load_action_config(CONFIG_PATH)

        self.assertEqual(config.action("pre_grasp").gripper, 5.0)
        self.assertEqual(config.action("grasp").gripper, 0.0)
        self.assertEqual(config.action("lift").gripper, 0.0)
        self.assertEqual(config.action("drop_to_bin").gripper, 5.0)
        for action in config.ordered_actions():
            self.assertGreaterEqual(action.gripper_duration_s, 0.45)
        self.assertAlmostEqual(config.mujoco_ctrl(config.action("grasp"))[-1], 0.0)
        self.assertAlmostEqual(config.mujoco_ctrl(config.action("drop_to_bin"))[-1], 0.08)

    def test_named_ctrl_mapping_preserves_chassis_actuators(self) -> None:
        config = load_action_config(CONFIG_PATH)
        complete_model_order = (
            "chassis_left_wheel",
            "chassis_right_wheel",
            *config.mujoco_actuator_names,
        )
        name_to_id = {name: index for index, name in enumerate(complete_model_order)}
        ctrl = [9.0] * len(complete_model_order)

        action = config.action("grasp")
        config.write_mujoco_ctrl(action, name_to_id, ctrl)

        self.assertEqual(ctrl[:2], [9.0, 9.0])
        self.assertEqual(tuple(ctrl[2:]), config.mujoco_ctrl(action))

    def test_robot_cmd_payload_matches_supplied_message_shape(self) -> None:
        config = load_action_config(CONFIG_PATH)
        payload = config.action("grasp").robot_cmd_payload()

        self.assertEqual(list(payload), ["end_pos", "joint_pos", "gripper", "mode"])
        self.assertEqual(len(payload["end_pos"]), 6)
        self.assertEqual(len(payload["joint_pos"]), 6)
        self.assertEqual(payload["mode"], 5)

    def test_out_of_limit_target_is_rejected(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["templates"]["grasp"]["joint_pos"][1] = 0.5
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Path(temp_dir) / "invalid.yaml"
            invalid.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ActionConfigError, "joint2 target"):
                load_action_config(invalid)


if __name__ == "__main__":
    unittest.main()
