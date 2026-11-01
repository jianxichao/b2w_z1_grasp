from glob import glob
from setuptools import find_packages, setup

package_name = "rl2_manipulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/urdf", glob("urdf/*.xacro")),
        (f"share/{package_name}/mjcf", glob("mjcf/*.xml")),
        (f"lib/{package_name}", ["scripts/arm_runtime_node", "scripts/vision_target_node"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    tests_require=["pytest"],
    extras_require={"mujoco": ["mujoco>=3.0"]},
    zip_safe=True,
    maintainer="RL2 Challenge Cup Team",
    maintainer_email="0305pkj@sjtu.edu.cn",
    description="ARX X5 grasp Action, JointState and TF bridge.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "arm_action_server = rl2_manipulation.arm_action_server:main",
            "arx_x5_action_player = rl2_manipulation.arx_x5_actions:cli_main",
            "validate_arx_x5_mujoco = rl2_manipulation.validate_arx_x5_mujoco:main",
            "calibration_helper = rl2_manipulation.calibration_helper:main",
        ],
    },
)
