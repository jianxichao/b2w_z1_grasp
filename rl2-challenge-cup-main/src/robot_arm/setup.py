"""兼容 ROS 2 Humble 自带的旧版 setuptools。"""

from glob import glob

from setuptools import find_packages, setup


setup(
    name="robot_arm",
    version="0.1.0",
    description="ARX X5 仿真、真机与 Orbbec YOLO 抓取程序",
    author="Robot Arm Team",
    author_email="robot-arm-team@users.noreply.github.com",
    maintainer="Robot Arm Team",
    maintainer_email="robot-arm-team@users.noreply.github.com",
    url="https://github.com/robot-arm-team/robot-arm",
    license="MIT",
    long_description="ARX X5 仿真、真机与 Orbbec YOLO 抓取程序。",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # The Unitree dock uses Ubuntu 20.04 / ROS 2 Foxy / Python 3.8, while the
    # operator laptop uses Humble / Python 3.10.  Keep the hardware-independent
    # package usable on both sides of that boundary.
    python_requires=">=3.8",
    install_requires=["numpy>=1.21,<2", "PyYAML>=5.4"],
    extras_require={
        "simulation": ["mujoco>=3.0"],
        "vision": ["opencv-python>=4.5", "ultralytics>=8.3"],
        "test": ["pytest>=6.2"],
        "full": [
            "mujoco>=3.0",
            "opencv-python>=4.5",
            "ultralytics>=8.3",
            "pytest>=6.2",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/robot_arm"]),
        ("share/robot_arm", ["package.xml"]),
        ("share/robot_arm/tools", glob("tools/*.py")),
        ("share/robot_arm/config", glob("config/*")),
        ("share/robot_arm/mjcf/robot", glob("mjcf/robot/*.xml")),
        ("share/robot_arm/mjcf/scenes", glob("mjcf/scenes/*.xml")),
        ("share/robot_arm/meshes/arx_x5", glob("meshes/arx_x5/*")),
        ("share/robot_arm/models/yolo", glob("models/yolo/*")),
        ("share/robot_arm/third_party/arx5_sdk/python", glob("third_party/arx5_sdk/python/arx5_interface*.so")),
        ("share/robot_arm/third_party/arx5_sdk/lib/x86_64", glob("third_party/arx5_sdk/lib/x86_64/*.so")),
        ("share/robot_arm/third_party/arx5_sdk/models", ["third_party/arx5_sdk/models/X5.urdf"]),
        ("share/robot_arm/third_party/arx5_sdk/models/meshes/X5", glob("third_party/arx5_sdk/models/meshes/X5/*")),
    ],
    entry_points={"console_scripts": ["robot-arm = robot_arm.app:main"]},
)
