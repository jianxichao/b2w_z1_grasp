from glob import glob
from setuptools import find_packages, setup

package_name = "rl2_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/config", glob("config/*.xml")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RL2 Challenge Cup Team",
    maintainer_email="0305pkj@sjtu.edu.cn",
    description="Navigation, odometry, obstacle avoidance, and return-to-base adapters.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cmd_vel_safety_node = rl2_nav.cmd_vel_safety_node:main",
            "frontier_explorer_node = rl2_nav.frontier_explorer_node:main",
            "mission_supervisor_node = rl2_nav.mission_supervisor_node:main",
            "base_cave_transit_node = rl2_nav.base_cave_transit_node:main",
            "cave_boundary_guard_node = rl2_nav.cave_boundary_guard_node:main",
            "odom_path_node = rl2_nav.odom_path_node:main",
            "slam_pose_jump_guard_node = rl2_nav.slam_pose_jump_guard_node:main",
        ],
    },
)
