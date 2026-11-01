from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from rl2_manipulation.arx_x5_actions import load_action_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config" / "grasp_poses.yaml"
MJCF_DIR = PACKAGE_ROOT / "mjcf"


class ArxX5MjcfTests(unittest.TestCase):
    def test_mjcf_files_are_well_formed_xml(self) -> None:
        for name in ("arx_x5_body.xml", "arx_x5_actuators.xml", "arx_x5_simplified.xml"):
            self.assertEqual(ET.parse(MJCF_DIR / name).getroot().tag, "mujoco")

    def test_mjcf_joint_limits_match_documented_config(self) -> None:
        config = load_action_config(CONFIG_PATH)
        body_root = ET.parse(MJCF_DIR / "arx_x5_body.xml").getroot()
        xml_joints = {node.attrib["name"]: node for node in body_root.iter("joint")}

        for joint in config.joints:
            self.assertIn(joint.name, xml_joints)
            lower, upper = map(float, xml_joints[joint.name].attrib["range"].split())
            self.assertEqual(lower, joint.lower)
            self.assertEqual(upper, joint.upper)

    def test_mjcf_actuator_order_matches_control_contract(self) -> None:
        config = load_action_config(CONFIG_PATH)
        actuator_root = ET.parse(MJCF_DIR / "arx_x5_actuators.xml").getroot()
        actuator_names = tuple(node.attrib["name"] for node in actuator_root)

        self.assertEqual(actuator_names, config.mujoco_actuator_names)
        self.assertEqual(
            [node.attrib["joint"] for node in actuator_root],
            [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "gripper_opening",
            ],
        )


if __name__ == "__main__":
    unittest.main()
