from glob import glob
from setuptools import find_packages, setup

package_name = "b2w_hardware"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RL2 Challenge Cup Team",
    maintainer_email="0305pkj@sjtu.edu.cn",
    description="Unitree B2W hardware adapters.",
    license="MIT",
    options={
        "develop": {"script_dir": f"$base/lib/{package_name}"},
        "install": {"install_scripts": f"$base/lib/{package_name}"},
    },
    entry_points={
        "console_scripts": [
            "sport_odom_node = b2w_hardware.sport_odom_node:main",
            "map_pose_node = b2w_hardware.map_pose_node:main",
            "unitree_slam_client = b2w_hardware.unitree_slam_client:main",
            "official_occupancy_grid_bridge = b2w_hardware.official_occupancy_grid_bridge:main",
            "b2w_health_check_node = b2w_hardware.b2w_health_check_node:main",
            "pointcloud_restamp_node = b2w_hardware.pointcloud_restamp_node:main",
            "b2w_sport_cmd_bridge = b2w_hardware.b2w_sport_cmd_bridge:main",
        ],
    },
)
