"""Bring up a fail-safe, non-composed Nav2 velocity chain for GO2-W."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    log_level = LaunchConfiguration("log_level")

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key="",
            param_rewrites={"use_sim_time": use_sim_time, "autostart": autostart},
            convert_types=True,
        ),
        allow_substs=True,
    )
    tf_remaps = [("/tf", "tf"), ("/tf_static", "tf_static")]
    common = {
        "output": "screen",
        "parameters": [configured_params],
        "arguments": ["--ros-args", "--log-level", log_level],
    }

    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("rl2_nav"), "config", "nav2_params.yaml"]
                ),
            ),
            LogInfo(
                msg=(
                    "Starting isolated Nav2 velocity chain: /cmd_vel_raw -> "
                    "/cmd_vel_nav. Only cmd_vel_safety_node may publish /cmd_vel."
                )
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                remappings=tf_remaps + [("cmd_vel", "/cmd_vel_raw")],
                **common,
            ),
            Node(
                package="nav2_smoother",
                executable="smoother_server",
                name="smoother_server",
                remappings=tf_remaps,
                **common,
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                remappings=tf_remaps,
                **common,
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                remappings=tf_remaps + [("cmd_vel", "/cmd_vel_raw")],
                **common,
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                remappings=tf_remaps,
                **common,
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                remappings=tf_remaps,
                **common,
            ),
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                remappings=tf_remaps
                + [
                    ("cmd_vel", "/cmd_vel_raw"),
                    ("cmd_vel_smoothed", "/cmd_vel_nav"),
                ],
                **common,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                arguments=["--ros-args", "--log-level", log_level],
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": autostart,
                        "node_names": lifecycle_nodes,
                    }
                ],
            ),
        ]
    )
