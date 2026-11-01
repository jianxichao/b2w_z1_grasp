"""Nav2/Sport pipeline using Unitree API-1804 relocation outputs."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    forwarded = (
        "start_rviz",
        "start_perception",
        "start_supervisor",
        "supervisor_escape_enabled",
        "frontier_max_goals",
        "frontier_failure_stop_threshold",
        "frontier_auto_wall_escape",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("start_perception", default_value="false"),
            DeclareLaunchArgument("start_supervisor", default_value="false"),
            DeclareLaunchArgument("supervisor_escape_enabled", default_value="false"),
            DeclareLaunchArgument("frontier_max_goals", default_value="1"),
            DeclareLaunchArgument("frontier_failure_stop_threshold", default_value="8"),
            DeclareLaunchArgument("frontier_auto_wall_escape", default_value="false"),
            LogInfo(
                msg=(
                    "Persistent-map relocation pipeline. Do NOT call start_mapping; "
                    "call /unitree_slam/init_pose with the saved PCD after launch. "
                    "All motion gates remain disabled."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("rl2_nav"),
                            "launch",
                            "autonomous_mapping_sport_bringup.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "pointcloud_topic": "/unitree/slam_relocation/global_map",
                    "scan_pointcloud_topic": "/unitree/slam_relocation/points",
                    "official_odom_topic": "/unitree/slam_relocation/odom",
                    "map_frame": "map",
                    **{name: LaunchConfiguration(name) for name in forwarded},
                }.items(),
            ),
        ]
    )
