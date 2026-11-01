"""Bring up only the Unitree Go2W official SLAM hardware bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("request_topic", default_value="/api/slam_operate/request"),
            DeclareLaunchArgument("response_topic", default_value="/api/slam_operate/response"),
            DeclareLaunchArgument("slam_info_topic", default_value="/slam_info"),
            DeclareLaunchArgument("slam_key_info_topic", default_value="/slam_key_info"),
            DeclareLaunchArgument(
                "official_odom_topic", default_value="/unitree/slam_mapping/odom"
            ),
            DeclareLaunchArgument("pointcloud_topic", default_value="/unitree/slam_mapping/points"),
            DeclareLaunchArgument("clearing_pointcloud_topic", default_value=""),
            DeclareLaunchArgument("navigation_map_topic", default_value="/navigation_map"),
            DeclareLaunchArgument(
                "command_status_topic", default_value="/unitree_slam/command_status"
            ),
            DeclareLaunchArgument("current_pose_topic", default_value="/slam/current_pose"),
            DeclareLaunchArgument("odom_topic", default_value="/slam/odom"),
            DeclareLaunchArgument("map_frame", default_value="slam_map"),
            DeclareLaunchArgument("base_frame", default_value="base_footprint"),
            DeclareLaunchArgument("pcd_path", default_value="/home/unitree/rl2.pcd"),
            DeclareLaunchArgument("navigate_mode", default_value="1"),
            DeclareLaunchArgument("navigate_speed", default_value="0.10"),
            DeclareLaunchArgument("recharge_command_enabled", default_value="false"),
            DeclareLaunchArgument("map_size_m", default_value="25.0"),
            DeclareLaunchArgument("resolution", default_value="0.05"),
            DeclareLaunchArgument("max_range_m", default_value="8.0"),
            DeclareLaunchArgument("pointcloud_in_map", default_value="true"),
            LogInfo(msg="Starting Unitree Go2W official SLAM bridge."),
            Node(
                package="rl2_hardware",
                executable="unitree_slam_client",
                name="unitree_slam_client",
                output="screen",
                parameters=[
                    {
                        "request_topic": LaunchConfiguration("request_topic"),
                        "response_topic": LaunchConfiguration("response_topic"),
                        "slam_info_topic": LaunchConfiguration("slam_info_topic"),
                        "slam_key_info_topic": LaunchConfiguration("slam_key_info_topic"),
                        "official_odom_topic": LaunchConfiguration(
                            "official_odom_topic"
                        ),
                        "current_pose_topic": LaunchConfiguration("current_pose_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "map_frame": LaunchConfiguration("map_frame"),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "pcd_path": LaunchConfiguration("pcd_path"),
                        "navigate_mode": LaunchConfiguration("navigate_mode"),
                        "navigate_speed": LaunchConfiguration("navigate_speed"),
                        "recharge_command_enabled": LaunchConfiguration(
                            "recharge_command_enabled"
                        ),
                    }
                ],
            ),
            Node(
                package="rl2_hardware",
                executable="official_occupancy_grid_bridge",
                name="official_occupancy_grid_bridge",
                output="screen",
                parameters=[
                    {
                        "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                        "clearing_pointcloud_topic": LaunchConfiguration("clearing_pointcloud_topic"),
                        "current_pose_topic": LaunchConfiguration("current_pose_topic"),
                        "navigation_map_topic": LaunchConfiguration("navigation_map_topic"),
                        "command_status_topic": LaunchConfiguration("command_status_topic"),
                        "reset_on_start_mapping": True,
                        "map_frame": LaunchConfiguration("map_frame"),
                        "map_size_m": LaunchConfiguration("map_size_m"),
                        "resolution": LaunchConfiguration("resolution"),
                        "max_range_m": LaunchConfiguration("max_range_m"),
                        "pointcloud_in_map": LaunchConfiguration("pointcloud_in_map"),
                        "clear_map_on_input_stale": True,
                        "input_stale_timeout_s": 2.0,
                    }
                ],
            ),
            Node(
                package="rl2_hardware",
                executable="go2w_health_check_node",
                name="go2w_health_check_node",
                output="screen",
                parameters=[
                    {
                        "current_pose_topic": LaunchConfiguration("current_pose_topic"),
                        "navigation_map_topic": LaunchConfiguration("navigation_map_topic"),
                    }
                ],
            ),
        ]
    )
