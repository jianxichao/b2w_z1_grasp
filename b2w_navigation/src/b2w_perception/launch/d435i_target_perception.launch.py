"""Start D435i PT / ONNX target recognition without controlling robot motion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path", default_value="/home/robot/unitree_ws/sponge_best.pt"),
            DeclareLaunchArgument("robot_params", default_value=PathJoinSubstitution([FindPackageShare("b2w_nav"), "config", "b2w_robot.yaml"])),
            DeclareLaunchArgument("color_topic", default_value="/b2w/front/color/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/b2w/front/color/camera_info"),
            DeclareLaunchArgument("depth_topic", default_value="/b2w/front/aligned_depth_to_color/image_raw"),
            DeclareLaunchArgument("odom_topic", default_value="/slam/odom"),
            LogInfo(msg="Starting read-only D435i target perception; no motion commands are published."),
            Node(
                package="b2w_perception",
                executable="yolo_target_perception_node",
                name="yolo_target_perception_node",
                output="screen",
                parameters=[
                    PathJoinSubstitution([FindPackageShare("b2w_perception"), "config", "d435i_target_perception.yaml"]),
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        
                        "color_topic": LaunchConfiguration("color_topic"),
                        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                    },
                    ParameterFile(LaunchConfiguration("robot_params"), allow_substs=True),
                ],
            ),
        ]
    )
