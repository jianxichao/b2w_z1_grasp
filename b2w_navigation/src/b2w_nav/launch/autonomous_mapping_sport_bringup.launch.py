"""B2W native lidar mapping, copied Nav2/Frontier, and gated sport output.

The optional official backend retains the original Unitree mapping adapter.
Both velocity gates and the mission start disabled.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    robot_params_path = LaunchConfiguration("robot_params_file")
    robot_params = ParameterFile(robot_params_path, allow_substs=True)
    official = IfCondition(PythonExpression(["'", LaunchConfiguration("sensor_backend"), "' == 'official'"]))
    builtin = IfCondition(PythonExpression(["'", LaunchConfiguration("sensor_backend"), "' == 'builtin'"]))
    native_mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("b2w_nav"), "launch", "builtin_mapping.launch.py"])),
        condition=builtin,
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    scan_pointcloud_topic = LaunchConfiguration("scan_pointcloud_topic")
    official_odom_topic = LaunchConfiguration("official_odom_topic")
    map_topic = LaunchConfiguration("map_topic")
    map_frame = LaunchConfiguration("map_frame")
    base_frame = LaunchConfiguration("base_frame")
    start_rviz = LaunchConfiguration("start_rviz")
    frontier_max_goals = LaunchConfiguration("frontier_max_goals")
    frontier_failure_stop_threshold = LaunchConfiguration(
        "frontier_failure_stop_threshold"
    )
    exploration_min_x = LaunchConfiguration("exploration_min_x")
    exploration_max_x = LaunchConfiguration("exploration_max_x")
    exploration_min_y = LaunchConfiguration("exploration_min_y")
    exploration_max_y = LaunchConfiguration("exploration_max_y")
    frontier_auto_wall_escape = LaunchConfiguration("frontier_auto_wall_escape")
    start_perception = LaunchConfiguration("start_perception")
    start_supervisor = LaunchConfiguration("start_supervisor")
    mission_auto_start = LaunchConfiguration("mission_auto_start")
    target_goal_count = LaunchConfiguration("target_goal_count")
    supervisor_escape_enabled = LaunchConfiguration("supervisor_escape_enabled")
    exploration_square_half_size_m = LaunchConfiguration(
        "exploration_square_half_size_m"
    )
    exploration_boundary_margin_m = LaunchConfiguration(
        "exploration_boundary_margin_m"
    )
    target_posture_enabled = LaunchConfiguration("target_posture_enabled")
    stop_after_target_interaction = LaunchConfiguration(
        "stop_after_target_interaction"
    )
    target_crouch_hold_s = LaunchConfiguration("target_crouch_hold_s")
    target_posture_retry_max = LaunchConfiguration("target_posture_retry_max")
    target_posture_retry_settle_s = LaunchConfiguration(
        "target_posture_retry_settle_s"
    )
    lock_visual_goal_after_stable_candidate = LaunchConfiguration(
        "lock_visual_goal_after_stable_candidate"
    )
    locked_approach_min_confidence = LaunchConfiguration(
        "locked_approach_min_confidence"
    )
    frontier_unavailable_scan_delay_s = LaunchConfiguration(
        "frontier_unavailable_scan_delay_s"
    )

    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("b2w_hardware"), "launch", "b2w_slam_bringup.launch.py"]
            )
        ),
        condition=official,
        launch_arguments={
            "pointcloud_topic": pointcloud_topic,
            "clearing_pointcloud_topic": scan_pointcloud_topic,
            "official_odom_topic": official_odom_topic,
            "map_frame": map_frame,
            "base_frame": base_frame,
            "pointcloud_in_map": "true",
            "navigation_map_topic": "/navigation_map_raw",
            "robot_params": robot_params_path,
        }.items(),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("b2w_nav"), "launch", "navigation.launch.py"]
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time, "autostart": "true", "params_file": LaunchConfiguration("nav_params_file")}.items(),
    )

    cave_boundary_guard = Node(
        condition=official,
        package="b2w_nav",
        executable="cave_boundary_guard_node",
        name="cave_boundary_guard_node",
        output="screen",
        parameters=[
            {
                "input_map_topic": "/navigation_map_raw",
                "output_map_topic": map_topic,
                "gate_topic": "/base_cave_mission/cave_gate",
                "enabled_on_start": False,
                "wall_thickness_m": 0.30,
            },
            robot_params,
        ],
    )

    cloud_restamp = Node(
        condition=official,
        package="b2w_hardware",
        executable="pointcloud_restamp_node",
        name="mapping_cloud_restamp",
        output="screen",
        parameters=[
            {
                "input_topic": scan_pointcloud_topic,
                "output_topic": "/unitree/slam_mapping/points_synced",
                "frame_id": map_frame,
            },
            robot_params,
        ],
    )

    cloud_to_scan = Node(
        condition=official,
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="mapping_cloud_to_scan",
        output="screen",
        remappings=[
            ("cloud_in", "/unitree/slam_mapping/points_synced"),
            ("scan", "/scan"),
        ],
        parameters=[
            {
                "target_frame": base_frame,
                "transform_tolerance": 0.20,
                "min_height": -0.25,
                "max_height": 0.60,
                "angle_min": -3.14159,
                "angle_max": 3.14159,
                "angle_increment": 0.00873,
                "scan_time": 0.10,
                "range_min": 0.20,
                "range_max": 8.0,
                "use_inf": True,
                "inf_epsilon": 1.0,
            },
            robot_params,
        ],
    )

    safety = Node(
        package="b2w_nav",
        executable="cmd_vel_safety_node",
        name="cmd_vel_safety_node",
        output="screen",
        parameters=[
            {
                "input_topic": "/cmd_vel_nav",
                "recovery_input_topic": "/cmd_vel_recovery",
                "output_topic": "/cmd_vel_sport",
                "odom_topic": "/slam/odom",
                "scan_topic": "/scan",
                "enabled_on_start": False,
                "allow_reverse": False,
                "recovery_max_reverse": 0.0,
                "max_linear_x": 0.15,
                "max_linear_y": 0.0,
                "min_effective_linear_y": 0.15,
                "allow_lateral_navigation": False,
                "max_angular_z": 0.20,
                "min_effective_angular_z": 0.0,
                "max_linear_accel": 0.50,
                "max_linear_y_accel": 0.40,
                "max_angular_accel": 0.65,
                "angular_passthrough": True,
                "stuck_recovery_enabled": False,
                "stuck_command_min_speed_mps": 0.12,
                "stuck_timeout_sec": 3.0,
                "stuck_min_progress_m": 0.05,
                "stuck_recovery_speed_mps": 0.20,
                "stuck_recovery_duration_sec": 1.0,
                "stuck_recovery_side_clearance_m": 0.60,
                "stuck_recovery_cooldown_sec": 5.0,
                "lethal_start_recovery_enabled": False,
                "global_costmap_timeout_sec": 1.0,
                "lethal_start_cost_threshold": 99,
                "lethal_start_trigger_delay_sec": 0.15,
                "lethal_start_max_attempts": 2,
            },
            robot_params,
        ],
    )

    sport_bridge = Node(
        package="b2w_hardware",
        executable="b2w_sport_cmd_bridge",
        name="b2w_sport_cmd_bridge",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("b2w_hardware"), "config", "b2w_sport_bridge.yaml"]
            ),
            {
                "allow_reverse": False,
            },
            robot_params,
        ],
    )

    explorer = Node(
        package="b2w_nav",
        executable="frontier_explorer_node",
        name="frontier_explorer_node",
        output="screen",
        parameters=[
            {
                "enabled": False,
                "map_topic": map_topic,
                "map_frame": map_frame,
                "base_frame": base_frame,
                "replan_interval_sec": 2.5,
                "candidate_stability_required_cycles": 3,
                "candidate_stability_radius_m": 0.60,
                "goal_timeout_sec": 12.0,
                "prefer_nearest_until_first_success": False,
                "global_costmap_goal_clearance_m": 0.32,
                "global_costmap_goal_clearance_cost_max": 65,
                "path_clearance_m": 0.25,
                "max_goals_per_enable": frontier_max_goals,
                "failure_stop_threshold": frontier_failure_stop_threshold,
                "exploration_min_x": exploration_min_x,
                "exploration_max_x": exploration_max_x,
                "exploration_min_y": exploration_min_y,
                "exploration_max_y": exploration_max_y,
                "auto_wall_escape_enabled": frontier_auto_wall_escape,
                "backup_escape_enabled": False,
                "backup_distance_m": 1.0,
                "backup_speed_mps": 0.20,
                "backup_timeout_sec": 15.0,
                "backup_cooldown_sec": 3.0,
                "backup_max_attempts": 3,
                "backup_failure_threshold": 1,
                "wall_escape_min_movement_m": 0.25,
            },
            robot_params,
        ],
    )

    trajectory = Node(
        package="b2w_nav",
        executable="odom_path_node",
        name="odom_path_node",
        output="screen",
        parameters=[
            {
                "odom_topic": "/slam/odom",
                "path_topic": "/slam/trajectory",
                "min_distance_m": 0.02,
                "max_points": 10000,
            },
            robot_params,
        ],
    )

    slam_pose_jump_guard = Node(
        package="b2w_nav",
        executable="slam_pose_jump_guard_node",
        name="slam_pose_jump_guard",
        output="screen",
        parameters=[
            {
                "odom_topic": "/slam/odom",
                "map_topic": map_topic,
                "jump_distance_m": 1.0,
                "stable_step_distance_m": 0.10,
                "stable_duration_s": 3.0,
                "minimum_known_map_cells": 1000,
            },
            robot_params,
        ],
    )

    supervisor = Node(
        package="b2w_nav",
        executable="mission_supervisor_node",
        name="mission_supervisor_node",
        output="screen",
        condition=IfCondition(start_supervisor),
        parameters=[
            {
                "task_event_topic": "/task_event",
                "target_pose_topic": "/perception/target_pose",
                "target_status_topic": "/target_detection_status",
                "current_pose_topic": "/slam/current_pose",
                "map_topic": map_topic,
                "navigate_action": "/navigate_to_pose",
                "frontier_enable_service": "/frontier_explorer_node/enable",
                "map_frame": map_frame,
                "auto_start": mission_auto_start,
                "target_goal_count": target_goal_count,
                "frontier_goals_before_scan": 3,
                "ignored_target_radius_m": 0.8,
                "exploration_square_enabled": True,
                "exploration_square_half_size_m": exploration_square_half_size_m,
                "exploration_boundary_margin_m": exploration_boundary_margin_m,
                "candidate_confidence_threshold": 0.55,
                "visual_candidate_required_frames": 5,
                "visual_candidate_stability_radius_m": 0.75,
                "target_confirm_confidence_threshold": 0.65,
                "target_confirm_required_frames": 3,
                "target_loss_confidence_threshold": 0.25,
                "target_loss_required_frames": 10,
                "target_verify_timeout_s": 6.0,
                "target_seen_hold_s": 2.0,
                "target_posture_enabled": target_posture_enabled,
                "stop_after_target_interaction": stop_after_target_interaction,
                "lock_visual_goal_after_stable_candidate": lock_visual_goal_after_stable_candidate,
                "locked_approach_min_confidence": locked_approach_min_confidence,
                "arm_integration_enabled": False,
                "arm_action_name": "/arm/grasp_target",
                "arm_target_pose_topic": "/robot_arm/target_pose",
                "arm_target_wait_timeout_s": 10.0,
                "arm_target_max_age_s": 1.0,
                "arm_action_timeout_s": 120.0,
                "target_crouch_hold_s": target_crouch_hold_s,
                "target_posture_retry_max": target_posture_retry_max,
                "target_posture_retry_settle_s": target_posture_retry_settle_s,
                "target_posture_guard_settle_s": 0.5,
                "target_pre_crouch_stand_settle_s": 1.0,
                "target_crouch_transition_s": 2.5,
                "target_stand_settle_s": 4.0,
                "target_association_radius_m": 0.50,
                "memory_reanchor_required_frames": 4,
                "memory_reanchor_stability_radius_m": 0.35,
                "memory_reanchor_max_correction_m": 1.50,
                "memory_reanchor_min_visible_distance_m": 0.65,
                "memory_reanchor_max_visible_distance_m": 4.0,
                "memory_reanchor_half_fov_deg": 30.0,
                "memory_reanchor_camera_forward_offset_m": 0.15,
                "target_confirm_timeout_s": 1.5,
                "candidate_reject_radius_m": 0.8,
                "candidate_retry_cooldown_s": 10.0,
                "post_nav_confirm_timeout_s": 3.0,
                "scan_fallback_min_sec": 20.0,
                "frontier_unavailable_scan_delay_s": frontier_unavailable_scan_delay_s,
                "scan_topic": "/scan",
                "recovery_cmd_vel_topic": "/cmd_vel_recovery",
                "escape_enabled": supervisor_escape_enabled,
                "escape_lateral_trigger_m": 0.55,
                "escape_lateral_destination_clearance_m": 0.80,
                "escape_lateral_speed_mps": 0.25,
                "escape_lateral_duration_s": 1.2,
                "escape_min_movement_m": 0.25,
                "escape_backup_distance_m": 1.0,
                "escape_backup_speed_mps": 0.20,
                "escape_backup_timeout_s": 15.0,
                "escape_cooldown_s": 3.0,
                "approach_standoff_m": 1.0,
                "approach_safety_radius_m": 0.80,
                "approach_allow_unknown_cells": False,
                "approach_num_angles": 24,
                "approach_max_candidates": 12,
                "approach_arrival_tolerance_m": 0.10,
                "approach_arrival_max_distance_m": 1.15,
                "approach_standoff_retry_max": 0,
                "approach_timeout_s": 25.0,
                "approach_near_stall_distance_m": 0.70,
                "approach_near_stall_timeout_s": 4.0,
                "approach_progress_epsilon_m": 0.04,
                "approach_visual_loss_grace_s": 12.0,
                "approach_topup_standoff_m": 1.0,
                "scan_attempts_max": 6,
                "scan_rotations_before_relocate": 2,
                "scan_segment_yaw_deg": 60.0,
                "scan_segment_pause_s": 1.0,
                "spin_action": "/spin",
                "spin_timeout_s": 20.0,
                "scan_point_min_distance_m": 0.75,
                "scan_point_max_distance_m": 3.0,
                "scan_point_revisit_radius_m": 1.0,
                "scan_point_timeout_s": 60.0,
                "target_hold_s": 5.0,
                "return_timeout_s": 90.0,
                "return_prealign_enabled": True,
                "return_prealign_threshold_deg": 30.0,
                "return_prealign_timeout_s": 15.0,
                "goal_cancel_settle_s": 0.5,
            },
            robot_params,
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="b2w_mapping_rviz",
        output="screen",
        condition=IfCondition(start_rviz),
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("b2w_nav"), "rviz", "navigation_only.rviz"]
            ),
        ],
    )

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("b2w_perception"),
                    "launch",
                    "d435i_target_perception.launch.py",
                ]
            )
        ),
        condition=IfCondition(start_perception),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("nav_params_file", default_value=PathJoinSubstitution([FindPackageShare("b2w_nav"), "config", "nav2_params.yaml"])),
            DeclareLaunchArgument("robot_params_file", default_value=PathJoinSubstitution([FindPackageShare("b2w_nav"), "config", "b2w_robot.yaml"])),
            DeclareLaunchArgument("sensor_backend", default_value="builtin", choices=["builtin", "official"]),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "pointcloud_topic", default_value="/unitree/slam_mapping/points"
            ),
            DeclareLaunchArgument(
                "scan_pointcloud_topic",
                default_value="/unitree/slam_mapping/points",
                description="Live cloud used for obstacle LaserScan generation.",
            ),
            DeclareLaunchArgument(
                "official_odom_topic",
                default_value="/unitree/slam_mapping/odom",
            ),
            DeclareLaunchArgument("map_topic", default_value="/navigation_map"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("base_frame", default_value="base_footprint"),
            DeclareLaunchArgument("start_rviz", default_value="false"),
            DeclareLaunchArgument(
                "start_perception",
                default_value="false",
                description="Start fixed chassis RGB-D target detection (requires real depth and intrinsics).",
            ),
            DeclareLaunchArgument(
                "start_supervisor",
                default_value="false",
                description=(
                    "Start the mission supervisor (search -> approach -> "
                    "hold 5s -> return to base)."
                ),
            ),
            DeclareLaunchArgument(
                "mission_auto_start",
                default_value="false",
                description=(
                    "Automatically begin the search mission after the supervisor "
                    "receives fresh map and pose data. Set false during staged "
                    "video preparation, then call /mission_supervisor_node/start."
                ),
            ),
            DeclareLaunchArgument(
                "target_goal_count",
                default_value="1",
                description=(
                    "Number of successful target interactions required before "
                    "returning to the recorded base pose. Use 1 for staged "
                    "single-target integration tests."
                ),
            ),
            DeclareLaunchArgument(
                "supervisor_escape_enabled",
                default_value="false",
                description=(
                    "Allow mission confirm/return failure recovery: lateral first, "
                    "then bounded reverse backup. Enable only after a staged real-robot test."
                ),
            ),
            DeclareLaunchArgument(
                "exploration_square_half_size_m",
                default_value="5.0",
                description=(
                    "Half-size of the square mission geofence centered at the "
                    "recorded start pose (5.0 means a 10m x 10m search area)."
                ),
            ),
            DeclareLaunchArgument(
                "exploration_boundary_margin_m",
                default_value="0.35",
                description=(
                    "Inset safety band inside the exploration square. Frontier, "
                    "visual-target, and scan goals are rejected within this many "
                    "metres of the mission boundary."
                ),
            ),
            DeclareLaunchArgument(
                "target_posture_enabled",
                default_value="false",
                description=(
                    "After each confirmed visual target: StopMove, crouch for "
                    "target_crouch_hold_s seconds, stand, then resume search. "
                    "Disabled by default."
                ),
            ),
            DeclareLaunchArgument(
                "stop_after_target_interaction",
                default_value="true",
                description=(
                    "After one target interaction finishes, leave the mission "
                    "idle instead of returning or resuming search. Intended "
                    "for staged base-only validation."
                ),
            ),
            DeclareLaunchArgument(
                "target_posture_retry_max",
                default_value="1",
                description=(
                    "Maximum BalanceStand posture-reset retries after a "
                    "pre-crouch rejection. Exhausted targets are recorded "
                    "and skipped without being counted successful."
                ),
            ),
            DeclareLaunchArgument(
                "target_posture_retry_settle_s",
                default_value="2.0",
                description=(
                    "BalanceStand settling time before the one posture retry."
                ),
            ),
            DeclareLaunchArgument(
                "target_crouch_hold_s",
                default_value="3.0",
                description=(
                    "Seconds the robot stays crouched at each confirmed "
                    "target before standing up again."
                ),
            ),
            DeclareLaunchArgument(
                "lock_visual_goal_after_stable_candidate",
                default_value="true",
                description=(
                    "Freeze a stable high-confidence visual target during its "
                    "approach, so near-field camera loss cannot cancel the "
                    "geometric arrival and optional crouch marker."
                ),
            ),
            DeclareLaunchArgument(
                "locked_approach_min_confidence",
                default_value="0.70",
                description=(
                    "Minimum confidence for freezing a spatially stable visual "
                    "candidate through near-field camera loss. Weaker "
                    "candidates require fresh close-range verification."
                ),
            ),
            DeclareLaunchArgument(
                "frontier_unavailable_scan_delay_s",
                default_value="4.0",
                description=(
                    "Continuous no-safe/no-reachable-Frontier duration before "
                    "switching to an in-place visual scan."
                ),
            ),
            DeclareLaunchArgument(
                "frontier_max_goals",
                default_value="1",
                description=(
                    "Frontier goal-attempt limit per enable interval. Zero keeps "
                    "exploration continuous; the mission supervisor decides when "
                    "to preempt it for visual scanning or target approach."
                ),
            ),
            DeclareLaunchArgument(
                "frontier_failure_stop_threshold", default_value="8"
            ),
            DeclareLaunchArgument("exploration_min_x", default_value="0.0"),
            DeclareLaunchArgument("exploration_max_x", default_value="0.0"),
            DeclareLaunchArgument("exploration_min_y", default_value="0.0"),
            DeclareLaunchArgument("exploration_max_y", default_value="0.0"),
            DeclareLaunchArgument(
                "frontier_auto_wall_escape", default_value="false"
            ),
            LogInfo(
                msg=(
                    "Staging B2W mapping + Nav2 + Sport pipeline. Motion remains "
                    "DISABLED until both cmd_vel_safety_node and "
                    "b2w_sport_cmd_bridge are explicitly enabled. Do not run rl_sar."
                )
            ),
            native_mapping,
            hardware,
            cave_boundary_guard,
            cloud_restamp,
            cloud_to_scan,
            nav2,
            safety,
            sport_bridge,
            explorer,
            slam_pose_jump_guard,
            trajectory,
            supervisor,
            perception,
            rviz,
        ]
    )
