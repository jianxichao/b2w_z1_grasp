"""Launch the base-to-cave waypoint transit node; dry-run is the default."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    names = (
        "dry_run", "go_to_base_first", "base_valid", "base_x", "base_y",
        "base_yaw", "cave_valid", "cave_x", "cave_y", "cave_yaw",
        "cave_a_valid", "cave_a_x", "cave_a_y",
        "cave_b_valid", "cave_b_x", "cave_b_y", "cave_entry_yaw",
        "waypoint_output_path",
        "global_costmap_topic", "compute_path_action",
        "traversable_cost_max", "target_search_radius_m",
        "start_search_radius_m", "max_plan_candidates",
        "waypoint_reached_tolerance_m",
    )
    values = {name: LaunchConfiguration(name) for name in names}
    return LaunchDescription(
        [
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("go_to_base_first", default_value="true"),
            DeclareLaunchArgument("base_valid", default_value="false"),
            DeclareLaunchArgument("base_x", default_value="0.0"),
            DeclareLaunchArgument("base_y", default_value="0.0"),
            DeclareLaunchArgument("base_yaw", default_value="0.0"),
            DeclareLaunchArgument("cave_valid", default_value="false"),
            DeclareLaunchArgument("cave_x", default_value="0.0"),
            DeclareLaunchArgument("cave_y", default_value="0.0"),
            DeclareLaunchArgument("cave_yaw", default_value="0.0"),
            DeclareLaunchArgument("cave_a_valid", default_value="false"),
            DeclareLaunchArgument("cave_a_x", default_value="0.0"),
            DeclareLaunchArgument("cave_a_y", default_value="0.0"),
            DeclareLaunchArgument("cave_b_valid", default_value="false"),
            DeclareLaunchArgument("cave_b_x", default_value="0.0"),
            DeclareLaunchArgument("cave_b_y", default_value="0.0"),
            DeclareLaunchArgument("cave_entry_yaw", default_value="0.0"),
            DeclareLaunchArgument("waypoint_output_path", default_value=""),
            DeclareLaunchArgument("global_costmap_topic", default_value="/global_costmap/costmap"),
            DeclareLaunchArgument("compute_path_action", default_value="/compute_path_to_pose"),
            DeclareLaunchArgument("traversable_cost_max", default_value="98"),
            DeclareLaunchArgument("target_search_radius_m", default_value="1.0"),
            DeclareLaunchArgument("start_search_radius_m", default_value="1.0"),
            DeclareLaunchArgument("max_plan_candidates", default_value="40"),
            DeclareLaunchArgument("waypoint_reached_tolerance_m", default_value="0.25"),
            LogInfo(msg="Base/cave transit starts in dry-run mode unless explicitly disabled."),
            Node(
                package="rl2_nav",
                executable="base_cave_transit_node",
                name="base_cave_transit_node",
                output="screen",
                parameters=[values],
            ),
        ]
    )
