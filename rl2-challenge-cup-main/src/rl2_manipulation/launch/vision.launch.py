"""只启动 Gemini 2 与 YOLO 目标发布节点。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory("rl2_manipulation")) / "config/arm_runtime.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="dry-run"),
            DeclareLaunchArgument("robot_config_root", default_value=""),
            Node(
                package="rl2_manipulation",
                executable="vision_target_node",
                name="vision_target_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "mode": LaunchConfiguration("mode"),
                        "robot_config_root": LaunchConfiguration("robot_config_root"),
                    },
                ],
            ),
        ]
    )
