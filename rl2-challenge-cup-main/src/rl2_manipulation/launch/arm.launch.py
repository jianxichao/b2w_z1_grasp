"""机械臂 ROS2 统一启动。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory("rl2_manipulation")) / "config/arm_runtime.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="dry-run"),
            DeclareLaunchArgument("robot_config_root", default_value=""),
            DeclareLaunchArgument("mount_calibrated", default_value="false"),
            DeclareLaunchArgument("base_parent_frame", default_value="base_link"),
            DeclareLaunchArgument("start_vision", default_value="false"),
            Node(
                package="rl2_manipulation",
                executable="arm_runtime_node",
                name="arm_runtime_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "mode": LaunchConfiguration("mode"),
                        "robot_config_root": LaunchConfiguration("robot_config_root"),
                        "mount_calibrated": LaunchConfiguration("mount_calibrated"),
                        "base_parent_frame": LaunchConfiguration("base_parent_frame"),
                    },
                ],
            ),
            Node(
                package="rl2_manipulation",
                executable="vision_target_node",
                name="vision_target_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_vision")),
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
