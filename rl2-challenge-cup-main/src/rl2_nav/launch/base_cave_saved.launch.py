"""Start the base/cave tool with the recorded site waypoints."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("rl2_nav"), "config", "base_cave_waypoints.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("go_to_base_first", default_value="true"),
            DeclareLaunchArgument("waypoint_config", default_value=default_config),
            DeclareLaunchArgument("target_search_radius_m", default_value="1.0"),
            DeclareLaunchArgument("start_search_radius_m", default_value="1.0"),
            LogInfo(msg="Loading recorded base and cave A/B coordinates; dry-run defaults true."),
            Node(
                package="rl2_nav",
                executable="base_cave_transit_node",
                name="base_cave_transit_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("waypoint_config"),
                    {
                        "dry_run": LaunchConfiguration("dry_run"),
                        "go_to_base_first": LaunchConfiguration("go_to_base_first"),
                        "target_search_radius_m": LaunchConfiguration("target_search_radius_m"),
                        "start_search_radius_m": LaunchConfiguration("start_search_radius_m"),
                    },
                ],
            ),
        ]
    )
