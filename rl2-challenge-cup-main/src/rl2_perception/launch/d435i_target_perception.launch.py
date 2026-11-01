"""Start D435i ONNX target recognition without controlling robot motion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("color_topic", default_value="/d435i/color/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/d435i/color/camera_info"),
            DeclareLaunchArgument("depth_topic", default_value="/d435i/aligned_depth_to_color/image_raw"),
            DeclareLaunchArgument("odom_topic", default_value="/slam/odom"),
            LogInfo(msg="Starting read-only D435i target perception; no motion commands are published."),
            Node(
                package="rl2_perception",
                executable="yolo_target_perception_node",
                name="yolo_target_perception_node",
                output="screen",
                parameters=[
                    PathJoinSubstitution([FindPackageShare("rl2_perception"), "config", "d435i_target_perception.yaml"]),
                    {
                        "model_path": PathJoinSubstitution([FindPackageShare("rl2_perception"), "models", "yolo", "best.onnx"]),
                        "class_names_path": PathJoinSubstitution([FindPackageShare("rl2_perception"), "models", "yolo", "best.names.json"]),
                        "color_topic": LaunchConfiguration("color_topic"),
                        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                    },
                ],
            ),
        ]
    )
