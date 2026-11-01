"""Mission supervisor for the multi-target search -> approach -> return chain.

Cooperation with frontier_explorer_node (unchanged):
- SEARCHING keeps frontier exploration running and re-enables it after every
  approach so the robot keeps looking for the next target.
- On a confirmed, unlocked target pose (or ``target_seen:*`` event) the
  supervisor disables frontier exploration through its existing ``~/enable``
  service and takes over the Nav2 action server to approach a standoff pose.
- Reached targets are locked so the same target is never approached twice.
- Once the mapped area reaches a coverage threshold and fewer than the target
  goal count have been reached, the supervisor runs a bounded scan sequence:
  spin slowly in place; if no new target, navigate to a distant known free
  point and spin again, up to ``scan_attempts_max`` spins, then return to base.

Motion safety remains owned by cmd_vel_safety_node and b2w_sport_cmd_bridge;
this node never publishes velocity commands.
"""

from __future__ import annotations

from collections import deque
import copy
import json
import math
from typing import Deque, List, Optional, Tuple

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
import rclpy
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from unitree_go.msg import SportModeState

from b2w_interfaces.action import GraspTarget
from b2w_nav.frontier_explorer import FrontierExplorer
from b2w_nav.frontier_explorer import (
    choose_lateral_escape_direction,
    side_sector_clearances,
)
from b2w_nav.mission_supervisor_core import (
    MissionState,
    arrival_within_tolerance,
    candidate_is_rejected,
    distance_to_square_edge,
    direct_approach_goal_xy,
    find_farthest_free_point,
    find_nearest_free_point,
    is_frontier_unavailable_event,
    parse_target_seen,
    pick_approach_candidate,
    point_inside_square_with_margin,
    square_fence_bounds,
    should_start_scan_after_frontier_unavailable,
    target_is_locked,
    yaw_to_face_target,
)


SPORT_API_STAND_UP = 1004
SPORT_API_STAND_DOWN = 1005
SPORT_API_BALANCE_STAND = 1002

SPORT_MODE_STAND_DOWN = 5
SPORT_MODE_STAND_UP = 6
SPORT_MODE_BALANCE_STAND = 1


def sport_mode_confirms_api(api_id: int, mode: int) -> bool:
    """Confirm a posture request from the robot state when API replies are lost."""
    expected_modes = {
        SPORT_API_STAND_DOWN: SPORT_MODE_STAND_DOWN,
        SPORT_API_STAND_UP: SPORT_MODE_STAND_UP,
        SPORT_API_BALANCE_STAND: SPORT_MODE_BALANCE_STAND,
    }
    return expected_modes.get(int(api_id)) == int(mode)


class MissionSupervisorNode(Node):
    """Own the Nav2 goal stream for the mission chain when a target appears."""

    def __init__(self) -> None:
        super().__init__("mission_supervisor_node")

        self.declare_parameter("task_event_topic", "/task_event")
        self.declare_parameter("node_status_topic", "/mission_supervisor_node/status")
        self.declare_parameter("target_pose_topic", "/perception/target_pose")
        self.declare_parameter("target_status_topic", "/target_detection_status")
        self.declare_parameter("current_pose_topic", "/slam/current_pose")
        self.declare_parameter("map_topic", "/navigation_map")
        self.declare_parameter("goal_topic", "/mission/goal")
        self.declare_parameter("manual_goal_topic", "/manual_grasp_goal")
        self.declare_parameter("manual_base_topic", "/manual_grasp_base")
        self.declare_parameter("navigate_action", "/navigate_to_pose")
        self.declare_parameter("spin_action", "/spin")
        self.declare_parameter("frontier_enable_service", "/frontier_explorer_node/enable")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("auto_start", False)

        # Mission goals
        self.declare_parameter("target_goal_count", 3)
        self.declare_parameter("frontier_goals_before_scan", 3)
        self.declare_parameter("ignored_target_radius_m", 0.8)
        # 已见未取目标记忆：识别/扫描期看到但未选中的目标坐标入队，
        # 完成一个目标后优先直达记忆坐标做局部视觉确认，而不是重启
        # frontier 盲搜。选定目标期间新目击只记录不分心（防反复横跳）。
        self.declare_parameter("pending_target_memory_enabled", True)
        # A remembered coordinate is only an initial hint.  If fresh,
        # high-confidence observations form a stable cluster near (but not
        # associated with) that hint, permit one bounded re-anchor and then
        # freeze the corrected position again.  This preserves the useful
        # near-field camera-loss lock without crouching at a stale memory.
        self.declare_parameter("memory_reanchor_required_frames", 4)
        self.declare_parameter("memory_reanchor_stability_radius_m", 0.35)
        self.declare_parameter("memory_reanchor_max_correction_m", 1.50)
        # Only let fresh vision correct a remembered coordinate while that
        # coordinate should physically be observable by the head camera.
        # Distances here are horizontal map-plane projections (XY), never the
        # camera-to-object 3-D slant range.  Once the remembered XY position
        # enters the measured near-field blind zone, freeze memory:
        # a different yellow patch must not drag the goal away from the cube.
        self.declare_parameter("memory_reanchor_min_visible_distance_m", 0.65)
        self.declare_parameter("memory_reanchor_max_visible_distance_m", 4.0)
        self.declare_parameter("memory_reanchor_half_fov_deg", 30.0)
        self.declare_parameter("memory_reanchor_camera_forward_offset_m", 0.15)
        # 以任务起点为中心的正方形探索围栏（边长 = 2 * half_size）。
        self.declare_parameter("exploration_square_enabled", True)
        self.declare_parameter("exploration_square_half_size_m", 3.0)
        self.declare_parameter("exploration_boundary_margin_m", 0.35)
        # Visual target policy: a moderate-confidence observation may divert
        # exploration for a closer look; only a stable, high-confidence view
        # at the standoff pose counts as one mission target.
        self.declare_parameter("candidate_confidence_threshold", 0.55)
        self.declare_parameter("visual_candidate_required_frames", 5)
        self.declare_parameter("visual_candidate_stability_radius_m", 0.75)
        # The final image is taken after the robot has stopped at a target-
        # facing standoff pose.  Three stable 0.50+ observations are enough
        # for the real camera's frame rate while still rejecting a one-frame
        # false positive.
        self.declare_parameter("target_confirm_confidence_threshold", 0.65)
        self.declare_parameter("target_confirm_required_frames", 3)
        self.declare_parameter("target_loss_confidence_threshold", 0.25)
        self.declare_parameter("target_loss_required_frames", 10)
        self.declare_parameter("target_verify_timeout_s", 6.0)
        self.declare_parameter("target_seen_hold_s", 2.0)
        self.declare_parameter("target_association_radius_m", 0.50)
        self.declare_parameter("target_confirm_timeout_s", 1.5)
        # A head-mounted camera often loses a target once the robot reaches
        # its standoff ring.  When a candidate was stable *and* already met
        # the high-confidence threshold, lock its map pose for the whole
        # approach and let geometric Nav2 arrival confirm the interaction.
        # This keeps the base from abandoning/replanning at the near-field
        # camera blind spot.  Low-confidence candidates retain the historic
        # verify-at-standoff behavior.
        self.declare_parameter("lock_visual_goal_after_stable_candidate", True)
        # Only strong candidates may bypass near-field re-verification when
        # they leave the head camera FOV.  Lower-confidence candidates can
        # still be approached, but must pass target_confirm_* at arrival.
        self.declare_parameter("locked_approach_min_confidence", 0.70)
        # Optional target marker.  It is disabled by default: crouching moves
        # both lidar and camera, so the map accumulation is paused explicitly
        # while this action is in progress.
        self.declare_parameter("target_posture_enabled", False)
        # Staged base-only validation mode: after one confirmed target and its
        # optional posture action, remain idle instead of resuming exploration
        # or returning to base.  The normal mission default remains unchanged.
        self.declare_parameter("stop_after_target_interaction", False)
        self.declare_parameter("target_crouch_hold_s", 3.0)
        self.declare_parameter("target_posture_guard_settle_s", 0.5)
        self.declare_parameter("target_pre_crouch_stand_settle_s", 1.0)
        self.declare_parameter("target_crouch_transition_s", 2.5)
        self.declare_parameter("target_stand_settle_s", 4.0)
        self.declare_parameter("target_posture_ack_timeout_s", 5.0)
        # A terrain-dependent posture rejection gets one bounded recovery:
        # normalize to BalanceStand, settle, then retry the crouch sequence.
        # A second rejection permanently skips that target for this mission.
        self.declare_parameter("target_posture_retry_max", 1)
        self.declare_parameter("target_posture_retry_settle_s", 2.0)
        self.declare_parameter(
            "sport_status_topic", "/b2w_sport_cmd_bridge/status"
        )
        self.declare_parameter("sport_state_topic", "/lf/sportmodestate")
        self.declare_parameter(
            "sport_stop_service", "/b2w_sport_cmd_bridge/stop_move"
        )
        self.declare_parameter(
            "sport_stand_down_service", "/b2w_sport_cmd_bridge/stand_down"
        )
        self.declare_parameter(
            "sport_stand_up_service", "/b2w_sport_cmd_bridge/stand_up"
        )
        self.declare_parameter(
            "sport_balance_stand_service",
            "/b2w_sport_cmd_bridge/balance_stand",
        )
        self.declare_parameter(
            "map_accumulation_service",
            "/official_occupancy_grid_bridge/set_accumulation",
        )
        self.declare_parameter(
            "slam_jump_guard_monitoring_service",
            "/slam_pose_jump_guard/monitoring",
        )
        self.declare_parameter(
            "velocity_odom_jump_monitoring_service",
            "/cmd_vel_safety_node/odom_jump_monitoring",
        )
        # Optional navigation -> arm handshake.  Navigation supplies only the
        # semantic arrival decision; the final grasp pose must come from the
        # arm-mounted RGB-D camera in arm_base, not from an uncalibrated
        # map-to-arm transform.
        self.declare_parameter("arm_integration_enabled", False)
        self.declare_parameter("arm_action_name", "/arm/grasp_target")
        self.declare_parameter("arm_target_pose_topic", "/robot_arm/target_pose")
        # Dog -> arm target report: every confirmed target is published once
        # as latched JSON carrying the map position plus a transform-free
        # robot-relative bearing/distance, so the arm stack can search with
        # its own camera even after the target leaves the head camera's view.
        # Independent of arm_integration_enabled: the report is useful for a
        # later arm integration even while the grasp handshake stays off.
        self.declare_parameter(
            "arm_target_report_topic", "/mission_supervisor_node/target_report"
        )
        self.declare_parameter("arm_target_wait_timeout_s", 10.0)
        self.declare_parameter("arm_target_max_age_s", 1.0)
        self.declare_parameter("arm_target_clock_skew_fallback_s", 300.0)
        self.declare_parameter("arm_action_timeout_s", 120.0)
        self.declare_parameter("candidate_reject_radius_m", 0.8)
        self.declare_parameter("candidate_retry_cooldown_s", 10.0)
        # 转圈后导航到候选点，到达后等待确认的时长。
        self.declare_parameter("post_nav_confirm_timeout_s", 3.0)
        # 探索耗尽后允许转 scan 的最短任务时长（防止一启动就误触发）。
        self.declare_parameter("scan_fallback_min_sec", 20.0)
        # Frontier 的无候选事件每个连续区间只发一次；由 supervisor
        # 计时并切换到原地视觉扫描，避免永久停在 SEARCHING。
        self.declare_parameter("frontier_unavailable_scan_delay_s", 8.0)
        # 官方 SLAM 数据流丢失的自动中止：跳变守护锁定（RECOVERING 或
        # READY_FOR_MANUAL_REARM，两种状态下速度门均已关闭）或
        # /slam/current_pose 超龄时，任务无法继续安全运动，中止并回 IDLE。
        self.declare_parameter(
            "slam_guard_status_topic", "/b2w_nav/slam_pose_jump_guard/status"
        )
        self.declare_parameter("slam_loss_abort_enabled", True)
        self.declare_parameter("slam_pose_max_age_s", 5.0)
        self.declare_parameter("slam_loss_confirm_s", 2.0)
        # 速度门关闭感知：任务进行中 /cmd_vel_safety_node/status 报
        # enabled=false（人为撤防或里程计跳变自动撤防）并持续
        # motion_gate_closed_confirm_s 秒时，继续下发 Nav2/Spin 目标只会
        # 逐个烧满超时（现场曾空转约 10 分钟），因此直接中止任务回 IDLE。
        self.declare_parameter(
            "motion_gate_status_topic", "/cmd_vel_safety_node/status"
        )
        self.declare_parameter("motion_gate_abort_enabled", True)
        self.declare_parameter("motion_gate_closed_confirm_s", 2.0)
        # supervisor 自身目标失败时的脱困（先横移，横移不动再倒车）。
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("recovery_cmd_vel_topic", "/cmd_vel_recovery")
        self.declare_parameter("escape_enabled", False)
        self.declare_parameter("escape_scan_timeout_s", 0.50)
        self.declare_parameter("escape_lateral_trigger_m", 0.55)
        self.declare_parameter("escape_lateral_destination_clearance_m", 0.80)
        self.declare_parameter("escape_lateral_speed_mps", 0.25)
        self.declare_parameter("escape_lateral_duration_s", 1.2)
        self.declare_parameter("escape_min_movement_m", 0.25)
        self.declare_parameter("escape_backup_distance_m", 1.0)
        self.declare_parameter("escape_backup_speed_mps", 0.20)
        self.declare_parameter("escape_backup_timeout_s", 15.0)
        self.declare_parameter("escape_cooldown_s", 3.0)

        # Approach
        # Command a close but non-contact standoff.  Nav2 commonly settles a
        # few centimetres short, so arrival uses a separate, wider physical
        # band below instead of repeatedly replanning around the target.
        self.declare_parameter("approach_standoff_m", 0.35)
        self.declare_parameter("approach_safety_radius_m", 0.26)
        self.declare_parameter("approach_allow_unknown_cells", True)
        self.declare_parameter("approach_num_angles", 24)
        self.declare_parameter("approach_max_candidates", 12)
        self.declare_parameter("approach_failed_radius_m", 0.30)
        self.declare_parameter("approach_failed_memory", 16)
        # With the camera's ground projection about 0.15m ahead of
        # base_footprint, a 0.35m target XY radius leaves the cube projection
        # about 0.20m in front of the camera/body.  The 10cm lower-side band
        # accepts 0.25-0.35m once the hard maximum below is applied; accepting
        # 0.50m here made the posture trigger too far from a cube.
        self.declare_parameter("approach_arrival_tolerance_m", 0.10)
        self.declare_parameter("approach_arrival_max_distance_m", 0.35)
        self.declare_parameter("approach_timeout_s", 25.0)
        # Nav2 can remain RUNNING while oscillating around a wall-side
        # standoff.  Once a locked target is already in the near field, use
        # measured target distance progress as an earlier fuse: if the best
        # distance has not improved by the configured epsilon for this long,
        # cancel that route and reuse the strict arrival/top-up geometry.
        # This does not accept a farther target and does not bypass Nav2
        # collision checking; it only avoids waiting for the full timeout.
        self.declare_parameter("approach_near_stall_distance_m", 0.70)
        self.declare_parameter("approach_near_stall_timeout_s", 4.0)
        self.declare_parameter("approach_progress_epsilon_m", 0.04)
        # A forward-facing camera can temporarily lose a target while the
        # robot turns toward the standoff goal.  Do not cancel a valid Nav2
        # approach on a few missing detection frames; only abandon it after
        # this continuous loss window has elapsed.
        self.declare_parameter("approach_visual_loss_grace_s", 12.0)
        # 停靠点失败重试：approach 目标失败（超时/中止/到达距离不达标）时，
        # 把该停靠点写入 _failed_approach_goals（find_approach_standoffs 与
        # pick_approach_candidate 已按它过滤），换一个方向重发至多
        # approach_standoff_retry_max 次；重试用尽才放弃该视觉候选。
        # Cross-side retries are unsafe near a physical target: the robot can
        # circle around it and collide.  Keep them opt-in for large venues.
        self.declare_parameter("approach_standoff_retry_max", 0)
        # Bounded top-up walk: when Nav2 reports success but the robot stopped
        # short of the acceptance ring (safe early stop), walk one direct hop
        # toward the frozen anchor along the robot->target ray and re-check
        # arrival, instead of abandoning a visually good target (field run
        # 2026-09-27: stopped 0.74m away and idled).  0 disables.
        self.declare_parameter("approach_topup_max", 1)
        self.declare_parameter("approach_topup_max_distance_m", 1.6)
        # A top-up aims inside the normal 0.35m ring so Nav2's 8cm goal
        # tolerance and centimetre-scale pose noise do not finish outside the
        # hard 0.35m arrival limit (field failures stopped at 0.351/0.36m).
        self.declare_parameter("approach_topup_standoff_m", 0.26)

        # Scan sequence
        self.declare_parameter("scan_attempts_max", 6)
        # 连续多个扫描段都以失败告终通常意味着当前位置没有安全旋转空间。
        # 达到该次数后执行一次受限脱困并换扫描点，而不是终止整项任务。
        self.declare_parameter("scan_failed_segments_abort_max", 4)
        self.declare_parameter("scan_rotations_before_relocate", 2)
        # A full scan is divided into six small turns.  Pausing after each
        # 60-degree turn gives RGB-D/YOLO stable frames instead of asking it
        # to classify a continuously moving image.
        self.declare_parameter("scan_segment_yaw_deg", 60.0)
        self.declare_parameter("scan_segment_pause_s", 1.0)
        # 60° @ max_rotational_vel 0.15rad/s 理论只需 ~7s；20s 已是 3 倍
        # 余量。60s 会让"转不动"（门关/轮锁）一段就空等一分钟。
        self.declare_parameter("spin_timeout_s", 20.0)
        self.declare_parameter("scan_point_min_distance_m", 0.75)
        self.declare_parameter("scan_point_max_distance_m", 3.0)
        self.declare_parameter("scan_point_revisit_radius_m", 1.0)
        self.declare_parameter("scan_point_timeout_s", 60.0)

        # Timing
        self.declare_parameter("target_hold_s", 5.0)
        self.declare_parameter("return_timeout_s", 90.0)
        self.declare_parameter("return_prealign_enabled", True)
        self.declare_parameter("return_prealign_threshold_deg", 30.0)
        self.declare_parameter("return_prealign_timeout_s", 15.0)
        self.declare_parameter("return_position_tolerance_m", 0.30)
        self.declare_parameter("return_target_search_radius_m", 1.0)
        self.declare_parameter("return_free_threshold", 65)
        self.declare_parameter("goal_cancel_settle_s", 0.5)
        self.declare_parameter("control_rate_hz", 4.0)

        self._task_event_topic = str(self.get_parameter("task_event_topic").value)
        self._node_status_topic = str(self.get_parameter("node_status_topic").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._target_goal_count = max(1, int(self.get_parameter("target_goal_count").value))
        self._frontier_goals_before_scan = max(
            1, int(self.get_parameter("frontier_goals_before_scan").value)
        )
        self._ignored_target_radius_m = float(
            self.get_parameter("ignored_target_radius_m").value
        )
        self._candidate_confidence_threshold = float(
            self.get_parameter("candidate_confidence_threshold").value
        )
        self._visual_candidate_required_frames = max(
            1, int(self.get_parameter("visual_candidate_required_frames").value)
        )
        self._visual_candidate_stability_radius_m = max(
            0.05,
            float(self.get_parameter("visual_candidate_stability_radius_m").value),
        )
        self._target_confirm_confidence_threshold = float(
            self.get_parameter("target_confirm_confidence_threshold").value
        )
        self._target_confirm_required_frames = max(
            1, int(self.get_parameter("target_confirm_required_frames").value)
        )
        self._target_loss_confidence_threshold = float(
            self.get_parameter("target_loss_confidence_threshold").value
        )
        self._target_loss_required_frames = max(
            1, int(self.get_parameter("target_loss_required_frames").value)
        )
        self._target_verify_timeout_s = float(
            self.get_parameter("target_verify_timeout_s").value
        )
        self._target_seen_hold_s = float(
            self.get_parameter("target_seen_hold_s").value
        )
        self._target_association_radius_m = float(
            self.get_parameter("target_association_radius_m").value
        )
        self._target_confirm_timeout_s = float(
            self.get_parameter("target_confirm_timeout_s").value
        )
        self._lock_visual_goal_after_stable_candidate = bool(
            self.get_parameter("lock_visual_goal_after_stable_candidate").value
        )
        self._locked_approach_min_confidence = float(
            self.get_parameter("locked_approach_min_confidence").value
        )
        self._target_posture_enabled = bool(
            self.get_parameter("target_posture_enabled").value
        )
        self._target_posture_retry_max = max(
            0, int(self.get_parameter("target_posture_retry_max").value)
        )
        self._target_posture_retry_settle_s = max(
            0.0,
            float(self.get_parameter("target_posture_retry_settle_s").value),
        )
        self._stop_after_target_interaction = bool(
            self.get_parameter("stop_after_target_interaction").value
        )
        self._target_crouch_hold_s = max(
            0.0, float(self.get_parameter("target_crouch_hold_s").value)
        )
        self._target_posture_guard_settle_s = max(
            0.1, float(self.get_parameter("target_posture_guard_settle_s").value)
        )
        self._target_pre_crouch_stand_settle_s = max(
            0.0,
            float(
                self.get_parameter("target_pre_crouch_stand_settle_s").value
            ),
        )
        self._target_crouch_transition_s = max(
            0.0, float(self.get_parameter("target_crouch_transition_s").value)
        )
        self._target_stand_settle_s = max(
            0.0, float(self.get_parameter("target_stand_settle_s").value)
        )
        self._target_posture_ack_timeout_s = max(
            0.2, float(self.get_parameter("target_posture_ack_timeout_s").value)
        )
        self._arm_integration_enabled = bool(
            self.get_parameter("arm_integration_enabled").value
        )
        self._arm_target_wait_timeout_s = max(
            0.5, float(self.get_parameter("arm_target_wait_timeout_s").value)
        )
        self._arm_target_max_age_s = max(
            0.05, float(self.get_parameter("arm_target_max_age_s").value)
        )
        self._arm_target_clock_skew_fallback_s = max(
            self._arm_target_max_age_s,
            float(self.get_parameter("arm_target_clock_skew_fallback_s").value),
        )
        self._arm_action_timeout_s = max(
            1.0, float(self.get_parameter("arm_action_timeout_s").value)
        )
        self._candidate_reject_radius_m = float(
            self.get_parameter("candidate_reject_radius_m").value
        )
        self._candidate_retry_cooldown_s = float(
            self.get_parameter("candidate_retry_cooldown_s").value
        )
        self._post_nav_confirm_timeout_s = float(
            self.get_parameter("post_nav_confirm_timeout_s").value
        )
        self._scan_fallback_min_sec = float(
            self.get_parameter("scan_fallback_min_sec").value
        )
        self._frontier_unavailable_scan_delay_s = max(
            0.0,
            float(self.get_parameter("frontier_unavailable_scan_delay_s").value),
        )
        self._slam_guard_status_topic = str(
            self.get_parameter("slam_guard_status_topic").value
        )
        self._slam_loss_abort_enabled = bool(
            self.get_parameter("slam_loss_abort_enabled").value
        )
        self._slam_pose_max_age_s = max(
            1.0, float(self.get_parameter("slam_pose_max_age_s").value)
        )
        self._slam_loss_confirm_s = max(
            0.0, float(self.get_parameter("slam_loss_confirm_s").value)
        )
        self._motion_gate_status_topic = str(
            self.get_parameter("motion_gate_status_topic").value
        )
        self._motion_gate_abort_enabled = bool(
            self.get_parameter("motion_gate_abort_enabled").value
        )
        self._motion_gate_closed_confirm_s = max(
            0.0,
            float(self.get_parameter("motion_gate_closed_confirm_s").value),
        )
        self._escape_lateral_trigger_m = float(
            self.get_parameter("escape_lateral_trigger_m").value
        )
        self._escape_lateral_destination_clearance_m = float(
            self.get_parameter("escape_lateral_destination_clearance_m").value
        )
        self._escape_lateral_speed_mps = float(
            self.get_parameter("escape_lateral_speed_mps").value
        )
        self._escape_lateral_duration_s = float(
            self.get_parameter("escape_lateral_duration_s").value
        )
        self._escape_min_movement_m = float(
            self.get_parameter("escape_min_movement_m").value
        )
        self._escape_backup_distance_m = float(
            self.get_parameter("escape_backup_distance_m").value
        )
        self._escape_backup_speed_mps = float(
            self.get_parameter("escape_backup_speed_mps").value
        )
        self._escape_backup_timeout_s = float(
            self.get_parameter("escape_backup_timeout_s").value
        )
        self._exploration_square_enabled = bool(
            self.get_parameter("exploration_square_enabled").value
        )
        self._exploration_square_half_size_m = float(
            self.get_parameter("exploration_square_half_size_m").value
        )
        self._exploration_boundary_margin_m = max(
            0.0,
            float(self.get_parameter("exploration_boundary_margin_m").value),
        )
        if self._exploration_square_enabled and (
            self._exploration_boundary_margin_m
            >= self._exploration_square_half_size_m
        ):
            self.get_logger().warn(
                "exploration_boundary_margin_m must be smaller than the square "
                "half-size; disabling the boundary margin for this mission."
            )
            self._exploration_boundary_margin_m = 0.0
        self._approach_standoff_m = float(self.get_parameter("approach_standoff_m").value)
        self._approach_standoff_retry_max = max(
            0, int(self.get_parameter("approach_standoff_retry_max").value)
        )
        self._approach_topup_max = max(
            0, int(self.get_parameter("approach_topup_max").value)
        )
        self._approach_topup_max_distance_m = float(
            self.get_parameter("approach_topup_max_distance_m").value
        )
        self._approach_topup_standoff_m = max(
            0.05, float(self.get_parameter("approach_topup_standoff_m").value)
        )
        self._approach_safety_radius_m = float(
            self.get_parameter("approach_safety_radius_m").value
        )
        self._approach_allow_unknown_cells = bool(
            self.get_parameter("approach_allow_unknown_cells").value
        )
        self._approach_num_angles = max(
            4, int(self.get_parameter("approach_num_angles").value)
        )
        self._approach_max_candidates = max(
            1, int(self.get_parameter("approach_max_candidates").value)
        )
        self._approach_failed_radius_m = float(
            self.get_parameter("approach_failed_radius_m").value
        )
        self._approach_arrival_tolerance_m = float(
            self.get_parameter("approach_arrival_tolerance_m").value
        )
        self._approach_arrival_max_distance_m = max(
            0.05,
            float(self.get_parameter("approach_arrival_max_distance_m").value),
        )
        self._approach_timeout_s = float(self.get_parameter("approach_timeout_s").value)
        self._approach_near_stall_distance_m = max(
            self._approach_arrival_max_distance_m,
            float(self.get_parameter("approach_near_stall_distance_m").value),
        )
        self._approach_near_stall_timeout_s = max(
            1.0,
            float(self.get_parameter("approach_near_stall_timeout_s").value),
        )
        self._approach_progress_epsilon_m = max(
            0.005,
            float(self.get_parameter("approach_progress_epsilon_m").value),
        )
        self._approach_visual_loss_grace_s = max(
            0.0,
            float(self.get_parameter("approach_visual_loss_grace_s").value),
        )
        self._scan_attempts_max = max(
            1, int(self.get_parameter("scan_attempts_max").value)
        )
        self._scan_failed_segments_abort_max = max(
            1, int(self.get_parameter("scan_failed_segments_abort_max").value)
        )
        self._scan_segment_yaw_rad = math.radians(
            min(180.0, max(10.0, float(self.get_parameter("scan_segment_yaw_deg").value)))
        )
        self._scan_segments_per_rotation = max(
            1, int(round((2.0 * math.pi) / self._scan_segment_yaw_rad))
        )
        # Make each full scan exactly one circle even if a different segment
        # angle is configured later.
        self._scan_segment_yaw_rad = (
            (2.0 * math.pi) / self._scan_segments_per_rotation
        )
        self._scan_segment_pause_s = max(
            0.0, float(self.get_parameter("scan_segment_pause_s").value)
        )
        self._spin_timeout_s = float(self.get_parameter("spin_timeout_s").value)
        self._scan_point_min_distance_m = float(
            self.get_parameter("scan_point_min_distance_m").value
        )
        self._scan_point_max_distance_m = float(
            self.get_parameter("scan_point_max_distance_m").value
        )
        self._scan_rotations_before_relocate = max(
            1, int(self.get_parameter("scan_rotations_before_relocate").value)
        )
        self._scan_point_revisit_radius_m = max(
            0.0, float(self.get_parameter("scan_point_revisit_radius_m").value)
        )
        self._scan_point_timeout_s = float(
            self.get_parameter("scan_point_timeout_s").value
        )
        self._target_hold_s = float(self.get_parameter("target_hold_s").value)
        self._return_timeout_s = float(self.get_parameter("return_timeout_s").value)
        self._return_prealign_enabled = bool(
            self.get_parameter("return_prealign_enabled").value
        )
        self._return_prealign_threshold_rad = math.radians(
            max(
                0.0,
                float(self.get_parameter("return_prealign_threshold_deg").value),
            )
        )
        self._return_prealign_timeout_s = max(
            1.0,
            float(self.get_parameter("return_prealign_timeout_s").value),
        )
        self._return_position_tolerance_m = float(
            self.get_parameter("return_position_tolerance_m").value
        )
        self._return_target_search_radius_m = float(
            self.get_parameter("return_target_search_radius_m").value
        )
        self._return_free_threshold = int(
            self.get_parameter("return_free_threshold").value
        )
        self._goal_cancel_settle_s = float(
            self.get_parameter("goal_cancel_settle_s").value
        )
        self._auto_start = bool(self.get_parameter("auto_start").value)

        # Mission state
        self._state = MissionState.IDLE
        self._base_pose: Optional[PoseStamped] = None
        self._target_pose: Optional[PoseStamped] = None
        self._robot_xy: Optional[Tuple[float, float]] = None
        self._robot_yaw = 0.0
        self._latest_map: Optional[OccupancyGrid] = None
        self._reached_target_count = 0
        self._locked_targets: List[Tuple[float, float]] = []
        # With the optional crouch/stand marker enabled, geometric arrival is
        # only provisional. A target is committed to the mission count after
        # the complete posture sequence has succeeded.
        self._pending_target_completion: Optional[
            Tuple[float, float, str, Optional[float]]
        ] = None
        self._target_posture_retry_count = 0
        # (x, y, reason): targets skipped after exhausting the posture retry.
        # They remain locked so vision/memory cannot loop at the same location.
        self._unserviceable_targets: List[Tuple[float, float, str]] = []
        # (x, y, confidence) of seen-but-not-chosen targets.  These coordinates
        # form mission scene memory; selecting one does not discard the others.
        self._pending_targets: List[Tuple[float, float, float]] = []
        self._frontier_goal_count = 0
        self._candidate_xy: Optional[Tuple[float, float]] = None
        # A visual observation is deliberately not allowed to preempt Nav2 on
        # its first frame.  This track is only promoted after consecutive,
        # spatially consistent RGB-D observations.
        self._pending_visual_candidate: Optional[Tuple[float, float, float]] = None
        self._pending_visual_candidate_frames = 0
        self._target_high_confidence_frames = 0
        self._target_low_confidence_frames = 0
        self._target_verify_until = 0.0
        self._approach_visual_loss_deadline = 0.0
        self._rejected_candidates: Deque[Tuple[float, float, float]] = deque(maxlen=50)
        self._scan_attempts = 0
        self._scan_rotations_at_location = 0
        self._scan_segment_index = 0
        self._scan_pause_until = 0.0
        self._scan_failed_segments_streak = 0
        self._scan_visited_points: Deque[Tuple[float, float]] = deque(maxlen=24)
        self._scan_best_candidate: Optional[Tuple[float, float, float]] = None
        self._confirm_until = 0.0
        self._mission_start_time = 0.0
        # SLAM-loss watchdog (see _check_slam_loss_abort): latest pose-jump
        # guard state, receipt time of the last /slam/current_pose, and the
        # debounce start for a suspected loss.
        self._slam_guard_state = ""
        self._current_pose_at_s = 0.0
        self._slam_loss_since = 0.0
        # Motion-gate watchdog (see _check_motion_gate_abort): latest
        # cmd_vel_safety enabled flag/reason from the latched status topic and
        # the debounce start for a closed gate.  None = never heard from the
        # gate, in which case the watchdog stays passive.
        self._motion_gate_enabled: Optional[bool] = None
        self._motion_gate_reason = ""
        self._motion_gate_closed_since = 0.0
        # Node status snapshot (see _publish_node_status): last /task_event
        # name and the last 1 Hz status publish time.
        self._last_event = ""
        self._last_status_pub_s = 0.0
        self._frontier_unavailable_since = 0.0
        self._frontier_unavailable_reason = ""
        self._latest_scan = None
        self._latest_scan_ns = 0
        self._recovery_active = False
        self._recovery_phase = ""
        self._recovery_direction = 0.0
        self._recovery_start_xy: Optional[Tuple[float, float]] = None
        self._recovery_start_ns = 0
        self._recovery_until_ns = 0
        self._recovery_pending: Optional[str] = None
        self._recovery_last_end_ns = 0
        self._failed_approach_goals: Deque[Tuple[float, float]] = deque(
            maxlen=max(1, int(self.get_parameter("approach_failed_memory").value))
        )
        # Standoff-retry bookkeeping (see _on_approach_failed): attempts used
        # for the current visual candidate, and the standoff actually sent so
        # a failure can blacklist that exact point.
        self._approach_standoff_attempts = 0
        self._last_approach_goal_xy: Optional[Tuple[float, float]] = None
        # Top-up walk bookkeeping (see _on_approach_succeeded): attempts used
        # for the current candidate and the one-shot request for the next
        # _send_approach_goal call to aim directly at the frozen anchor.
        self._approach_topup_used = 0
        self._approach_direct_topup = False
        self._approach_best_distance_m: Optional[float] = None
        self._approach_last_progress_time = 0.0
        self._approach_source = ""
        self._memory_reanchor_candidate: Optional[Tuple[float, float, float]] = None
        self._memory_reanchor_frames = 0
        self._memory_reanchor_used = False
        self._memory_reanchor_required_frames = max(
            2, int(self.get_parameter("memory_reanchor_required_frames").value)
        )
        self._memory_reanchor_stability_radius_m = max(
            0.05,
            float(
                self.get_parameter("memory_reanchor_stability_radius_m").value
            ),
        )
        self._memory_reanchor_max_correction_m = max(
            self._target_association_radius_m,
            float(self.get_parameter("memory_reanchor_max_correction_m").value),
        )
        self._memory_reanchor_min_visible_distance_m = max(
            0.0,
            float(
                self.get_parameter("memory_reanchor_min_visible_distance_m").value
            ),
        )
        self._memory_reanchor_max_visible_distance_m = max(
            self._memory_reanchor_min_visible_distance_m,
            float(
                self.get_parameter("memory_reanchor_max_visible_distance_m").value
            ),
        )
        self._memory_reanchor_half_fov_rad = math.radians(
            min(
                180.0,
                max(
                    0.0,
                    float(self.get_parameter("memory_reanchor_half_fov_deg").value),
                ),
            )
        )
        self._memory_reanchor_camera_forward_offset_m = float(
            self.get_parameter("memory_reanchor_camera_forward_offset_m").value
        )

        # Nav2 action state
        self._nav_client = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter("navigate_action").value),
        )
        self._goal_handle = None
        self._goal_pending = False
        self._goal_active = False
        self._goal_label = ""
        self._goal_generation = 0
        self._goal_start_time = 0.0

        # Spin action state
        self._spin_client = ActionClient(
            self,
            Spin,
            str(self.get_parameter("spin_action").value),
        )
        self._spin_handle = None
        self._spin_active = False
        self._spin_generation = 0
        self._spin_start_time = 0.0

        # Return pre-alignment uses an independent Spin action state.  The
        # chosen relative angle is fixed once, so an almost-180-degree turn
        # cannot flip left/right as the global path is replanned.
        self._return_spin_client = ActionClient(
            self,
            Spin,
            str(self.get_parameter("spin_action").value),
        )
        self._return_spin_handle = None
        self._return_spin_generation = 0
        self._return_alignment_active = False
        self._return_alignment_start_time = 0.0

        # Arm action state.  A target is counted only after the complete
        # GraspTarget action reports success.
        self._arm_client = ActionClient(
            self,
            GraspTarget,
            str(self.get_parameter("arm_action_name").value),
        )
        self._arm_goal_handle = None
        self._arm_goal_pending = False
        self._arm_goal_active = False
        self._arm_goal_generation = 0
        self._arm_goal_start_time = 0.0
        self._arm_wait_started_at = 0.0
        self._arm_wait_deadline = 0.0
        self._arm_target_sequence = 0
        self._arm_target_xy: Optional[Tuple[float, float]] = None
        self._arm_target_source = ""
        self._arm_target_confidence: Optional[float] = None
        self._arm_feedback_stage = ""

        self._next_goal_allowed_time = 0.0
        self._hold_until = 0.0
        self._target_posture_phase = ""
        self._target_posture_expected_api = 0
        self._target_posture_ack: Optional[bool] = None
        self._target_posture_ack_info = ""
        self._target_posture_ack_deadline = 0.0
        self._approach_target_locked = False
        self._active_return_xy: Optional[Tuple[float, float]] = None
        self._manual_mission_active = False
        # Optional explicit base for a one-shot manual navigation/grasp/return
        # task.  If absent, preserve the historic behavior: record the robot's
        # current map pose when the manual goal arrives.
        self._manual_base_override: Optional[PoseStamped] = None

        self._event_pub = self.create_publisher(String, self._task_event_topic, 10)
        # Latched snapshot of the mission state machine, mirroring the
        # slam_pose_jump_guard status pattern: a late `topic echo --once`
        # reader immediately sees the latest state.
        node_status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, self._node_status_topic, node_status_qos
        )
        # Latched dog -> arm target report (same transient-local pattern): a
        # late arm-stack subscriber immediately sees the last confirmed
        # target without waiting for the next one.
        self._arm_report_sequence = 0
        self._arm_report_pub = self.create_publisher(
            String,
            str(self.get_parameter("arm_target_report_topic").value),
            node_status_qos,
        )
        self._goal_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("goal_topic").value),
            10,
        )
        self._recovery_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("recovery_cmd_vel_topic").value),
            10,
        )

        self.create_subscription(
            String,
            self._task_event_topic,
            self._on_task_event,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_pose_topic").value),
            self._on_target_pose,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("manual_goal_topic").value),
            self._on_manual_grasp_goal,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("manual_base_topic").value),
            self._on_manual_grasp_base,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("target_status_topic").value),
            self._on_target_status,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("sport_status_topic").value),
            self._on_sport_status,
            10,
        )
        self.create_subscription(
            SportModeState,
            str(self.get_parameter("sport_state_topic").value),
            self._on_sport_mode_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("current_pose_topic").value),
            self._on_current_pose,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("map_topic").value),
            self._on_map,
            10,
        )
        arm_target_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("arm_target_pose_topic").value),
            self._on_arm_target_pose,
            arm_target_qos,
        )
        # TRANSIENT_LOCAL matches the guard's latched status publisher, so a
        # mission started after a latched jump immediately sees RECOVERING.
        slam_guard_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            self._slam_guard_status_topic,
            self._on_slam_guard_status,
            slam_guard_qos,
        )
        # Same latched pattern for the velocity gate (cmd_vel_safety_node):
        # a mission started while the gate is already disarmed must notice
        # within one debounce window instead of idling through goal timeouts.
        motion_gate_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            self._motion_gate_status_topic,
            self._on_motion_gate_status,
            motion_gate_qos,
        )

        self._frontier_client = self.create_client(
            SetBool,
            str(self.get_parameter("frontier_enable_service").value),
        )
        self._sport_stop_client = self.create_client(
            Trigger, str(self.get_parameter("sport_stop_service").value)
        )
        self._sport_stand_down_client = self.create_client(
            Trigger, str(self.get_parameter("sport_stand_down_service").value)
        )
        self._sport_stand_up_client = self.create_client(
            Trigger, str(self.get_parameter("sport_stand_up_service").value)
        )
        self._sport_balance_stand_client = self.create_client(
            Trigger,
            str(self.get_parameter("sport_balance_stand_service").value),
        )
        self._map_accumulation_client = self.create_client(
            SetBool, str(self.get_parameter("map_accumulation_service").value)
        )
        self._slam_jump_guard_monitoring_client = self.create_client(
            SetBool,
            str(self.get_parameter("slam_jump_guard_monitoring_service").value),
        )
        self._velocity_odom_jump_monitoring_client = self.create_client(
            SetBool,
            str(self.get_parameter("velocity_odom_jump_monitoring_service").value),
        )
        self._frontier_params_client = self.create_client(
            SetParameters,
            "/frontier_explorer_node/set_parameters",
        )
        self._fence_applied = False
        self.create_service(Trigger, "~/start", self._on_start)
        self.create_service(Trigger, "~/stop", self._on_stop)
        self.create_service(Trigger, "~/abort", self._on_abort)

        rate = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self.create_timer(1.0 / rate, self._control_step)
        self.create_timer(0.1, self._escape_step)
        self.get_logger().info(
            "mission_supervisor_node ready but IDLE; call ~/start to begin "
            "the search -> approach -> %s -> return chain (targets=%d)."
            % (
                "grasp" if self._arm_integration_enabled else "hold",
                self._target_goal_count,
            )
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_task_event(self, msg: String) -> None:
        event = msg.data.strip()
        if self._state == MissionState.IDLE and self._auto_start:
            if event == "frontier_exploration_enabled":
                self._begin_mission("auto_start")
            return
        if self._state != MissionState.SEARCHING:
            return
        if parse_target_seen(event):
            # The event is only an operator-facing notification.  Starting a
            # motion goal from it bypassed the confidence/status state machine
            # and could turn one far-away frame into a committed target.
            self.get_logger().info(
                "Target event received; waiting for status-driven candidate policy.",
                throttle_duration_sec=2.0,
            )
        elif event in ("frontier_candidate_stabilizing", "frontier_goal_started"):
            self._clear_frontier_unavailable()
        elif event == "frontier_goal_reached":
            self._clear_frontier_unavailable()
            self._frontier_goal_count += 1
            self.get_logger().info(
                "Frontier goals reached: %d/%d"
                % (self._frontier_goal_count, self._frontier_goals_before_scan)
            )
            if self._frontier_goal_count >= self._frontier_goals_before_scan:
                self._begin_scan_sequence()
        elif is_frontier_unavailable_event(event):
            self._mark_frontier_unavailable(event)

    def _on_target_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        try:
            cand_x = payload.get("candidate_x")
            cand_y = payload.get("candidate_y")
            cand_conf = payload.get("candidate_confidence")
            if cand_x is None or cand_y is None or cand_conf is None:
                self._observe_approach_confidence(None)
                return
            cand_x = float(cand_x)
            cand_y = float(cand_y)
            cand_conf = float(cand_conf)
        except (TypeError, ValueError):
            return

        if not self._pose_inside_fence(cand_x, cand_y):
            return
        if self._state in (MissionState.SEARCHING, MissionState.SCANNING):
            self._observe_visual_candidate(cand_x, cand_y, cand_conf)
            return

        if self._state in (
            MissionState.APPROACHING,
            MissionState.VERIFYING_TARGET,
        ):
            self._observe_approach_confidence(cand_conf, cand_x, cand_y)

    def _on_sport_status(self, msg: String) -> None:
        """Record firmware acknowledgement for the active posture request."""
        try:
            payload = json.loads(msg.data)
            if str(payload.get("stage", "")) != "response":
                return
            api_id = int(payload.get("api_id", 0))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if api_id != self._target_posture_expected_api:
            return
        self._target_posture_ack = bool(payload.get("succeed", False))
        self._target_posture_ack_info = str(payload.get("info", ""))

    def _on_sport_mode_state(self, msg: SportModeState) -> None:
        """Use measured posture as a fallback when firmware omits an API reply."""
        if self._target_posture_ack is not None:
            return
        api_id = int(self._target_posture_expected_api)
        mode = int(msg.mode)
        if not sport_mode_confirms_api(api_id, mode):
            return
        self._target_posture_ack = True
        self._target_posture_ack_info = "confirmed_by_sport_mode:%d" % mode
        self.get_logger().info(
            "Posture API %d confirmed by measured sport mode %d."
            % (api_id, mode)
        )

    def _clear_pending_visual_candidate(self) -> None:
        self._pending_visual_candidate = None
        self._pending_visual_candidate_frames = 0

    def _observe_visual_candidate(self, x: float, y: float, confidence: float) -> None:
        """Promote only a temporally and spatially stable visual candidate.

        Perception output can fluctuate as depth estimates change.  A one-off
        frame must not cancel a useful Frontier route, and two observations of
        different objects must not be merged into one candidate.
        """
        if confidence < self._candidate_confidence_threshold:
            self._clear_pending_visual_candidate()
            return
        if target_is_locked(x, y, self._locked_targets, self._ignored_target_radius_m):
            self._clear_pending_visual_candidate()
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if candidate_is_rejected(
            x,
            y,
            self._rejected_candidates,
            self._candidate_reject_radius_m,
            now,
            self._candidate_retry_cooldown_s,
        ):
            self._clear_pending_visual_candidate()
            return

        pending = self._pending_visual_candidate
        if pending is None or math.hypot(x - pending[0], y - pending[1]) > (
            self._visual_candidate_stability_radius_m
        ):
            self._pending_visual_candidate = (x, y, confidence)
            self._pending_visual_candidate_frames = 1
        else:
            # Light smoothing gives the later standoff computation a stable
            # location without allowing a detection to jump to another object.
            frames = self._pending_visual_candidate_frames
            weight = min(frames, 4)
            self._pending_visual_candidate = (
                (pending[0] * weight + x) / (weight + 1),
                (pending[1] * weight + y) / (weight + 1),
                max(pending[2], confidence),
            )
            self._pending_visual_candidate_frames += 1

        if self._pending_visual_candidate_frames < self._visual_candidate_required_frames:
            self._publish_event(
                "mission_visual_candidate_stabilizing:%d/%d"
                % (
                    self._pending_visual_candidate_frames,
                    self._visual_candidate_required_frames,
                )
            )
            return

        promoted = self._pending_visual_candidate
        self._clear_pending_visual_candidate()
        if self._state == MissionState.SCANNING:
            self._note_candidate_pose(promoted[0], promoted[1], promoted[2])
        elif self._state == MissionState.SEARCHING:
            self._start_visual_approach(promoted[0], promoted[1], promoted[2], "status")

    def _on_target_pose(self, msg: PoseStamped) -> None:
        if self._state not in (
            MissionState.APPROACHING,
            MissionState.VERIFYING_TARGET,
            MissionState.SCANNING,
        ):
            return
        xy = self._pose_xy_in_map(msg)
        if xy is None:
            return
        if not self._pose_inside_fence(xy[0], xy[1]):
            return
        if target_is_locked(
            xy[0],
            xy[1],
            self._locked_targets,
            self._ignored_target_radius_m,
        ):
            return
        if self._state == MissionState.SCANNING:
            # PoseStamped has neither confidence nor an object identity and
            # used to bypass the five-frame stability filter below the status
            # path.  A single bad depth sample could therefore become a fake
            # remembered target.  Scanning candidates are now accepted only
            # through _observe_visual_candidate(), which has confidence and
            # temporal/spatial stability checks.
            return
        # A PoseStamped observation has no confidence field.  During an
        # approach it must therefore never replace the fixed candidate used
        # to compute the Nav2 standoff goal: a different detection could
        # otherwise move the acceptance ring underneath the robot and make a
        # successful approach look like a failure.  Confidence-bearing status
        # messages refine an associated candidate in _observe_approach_confidence.
        if self._state in (MissionState.APPROACHING, MissionState.VERIFYING_TARGET):
            if self._target_pose is None or math.hypot(
                xy[0] - float(self._target_pose.pose.position.x),
                xy[1] - float(self._target_pose.pose.position.y),
            ) > self._target_association_radius_m:
                # PoseStamped carries no confidence or object identity.  Once
                # a target is locked, a displaced pose is most often a depth
                # jump of that same detection.  Never let it move the fixed
                # anchor or create a phantom remembered target.  Other targets
                # are deliberately collected during segmented scanning, where
                # stable confidence-bearing observations are available.
                return
            if self._state == MissionState.APPROACHING:
                # Even an associated pose must not move the locked anchor
                # while the Nav2 goal is active: that moves the arrival ring
                # after the goal was issued (field run 2026-09-27: flickering
                # depth estimates walked the anchor mid-drive and turned a
                # good stop into a measured 0.74m approach_not_close failure).
                # VERIFYING below may still refine from close-range views.
                return

        stored = PoseStamped()
        stored.header.stamp = self.get_clock().now().to_msg()
        stored.header.frame_id = self._map_frame
        stored.pose.position.x = xy[0]
        stored.pose.position.y = xy[1]
        stored.pose.position.z = msg.pose.position.z
        stored.pose.orientation = msg.pose.orientation
        self._target_pose = stored

    def _on_slam_guard_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            state = str(payload.get("state", ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if state == self._slam_guard_state:
            return
        self._slam_guard_state = state
        if state in ("RECOVERING", "READY_FOR_MANUAL_REARM"):
            # Every remembered coordinate belongs to the old map frame.  Drop
            # it immediately, before the watchdog's debounced mission abort,
            # so neither the planner nor an operator client can present stale targets as
            # valid after an official-SLAM relocalization/reset.
            self._target_pose = None
            self._candidate_xy = None
            self._pending_targets = []
            self._locked_targets = []
            self._unserviceable_targets = []
            self._base_pose = None
            self._robot_xy = None
            self._current_pose_at_s = 0.0
            self._scan_visited_points.clear()
            self._rejected_candidates.clear()
            self._clear_pending_visual_candidate()
            self.get_logger().warn(
                "SLAM pose-jump guard reported %s; mission motion is no longer "
                "safe and all old-map target memory was invalidated." % state
            )

    def _on_motion_gate_status(self, msg: String) -> None:
        """Track the velocity gate's latched enabled flag (see _check_motion_gate_abort)."""
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        enabled = payload.get("enabled")
        if enabled is None:
            return
        if not isinstance(enabled, bool):
            enabled = str(enabled).strip().lower() == "true"
        reason = str(payload.get("reason", ""))
        if enabled == self._motion_gate_enabled and reason == self._motion_gate_reason:
            return
        was_enabled = self._motion_gate_enabled
        self._motion_gate_enabled = enabled
        self._motion_gate_reason = reason
        if not enabled and was_enabled is not False:
            self.get_logger().warn(
                "Velocity gate reported enabled=false (%s)."
                % (reason or "no reason given")
            )

    def _on_current_pose(self, msg: PoseStamped) -> None:
        self._robot_xy = (float(msg.pose.position.x), float(msg.pose.position.y))
        self._robot_yaw = _yaw_from_quaternion(msg.pose.orientation)
        self._current_pose_at_s = self.get_clock().now().nanoseconds / 1e9

    def _on_scan(self, msg) -> None:
        self._latest_scan = msg
        self._latest_scan_ns = self.get_clock().now().nanoseconds

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._latest_map = msg

    def _on_arm_target_pose(self, msg: PoseStamped) -> None:
        """Start one grasp from a fresh arm-camera target after arrival."""
        if (
            not self._arm_integration_enabled
            or self._state != MissionState.WAITING_ARM_TARGET
            or self._arm_goal_pending
            or self._arm_goal_active
        ):
            return
        now = self.get_clock().now().nanoseconds / 1e9
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if not msg.header.frame_id.strip():
            self.get_logger().warn(
                "Ignoring arm target pose without a frame_id.",
                throttle_duration_sec=2.0,
            )
            return
        if stamp <= 0.0:
            self.get_logger().info(
                "Ignoring arm target pose without a valid timestamp.",
                throttle_duration_sec=2.0,
            )
            return
        age = abs(now - stamp)
        if (
            age > self._arm_target_max_age_s
            and age <= self._arm_target_clock_skew_fallback_s
        ):
            self.get_logger().info(
                "Ignoring stale arm target pose (age=%.2fs); waiting for a fresh lock."
                % age,
                throttle_duration_sec=2.0,
            )
            return
        if age > self._arm_target_clock_skew_fallback_s:
            # The Jetson dock may boot with a 1970/uptime clock while the user
            # PC has wall-clock time.  This callback is accepted only in
            # WAITING_ARM_TARGET, and the Gemini node publishes a stable-lock
            # heartbeat continuously.  Therefore receiving it after the
            # handoff is a stronger freshness signal than incomparable
            # cross-host absolute timestamps.
            self.get_logger().warn(
                "Arm-camera clock differs by %.1fs; accepting the post-handoff "
                "Gemini heartbeat using local receipt time." % age,
                throttle_duration_sec=5.0,
            )
        self._send_arm_grasp_goal(msg)

    def _on_start(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self._state in (MissionState.IDLE, MissionState.DONE):
            self._begin_mission("service_start")
            response.message = "mission started"
        else:
            response.message = "mission already running: %s" % self._state.value
        response.success = True
        return response

    def _on_manual_grasp_base(self, msg: PoseStamped) -> None:
        """Store an explicit map-frame base for the next manual grasp task."""
        if self._state not in (MissionState.IDLE, MissionState.DONE):
            self._publish_event(
                "manual_grasp_base_rejected:mission_busy:%s" % self._state.value
            )
            return
        frame = msg.header.frame_id.strip() or self._map_frame
        if frame != self._map_frame:
            self._publish_event("manual_grasp_base_rejected:frame=%s" % frame)
            self.get_logger().error(
                "Manual grasp base must use frame %s, got %s."
                % (self._map_frame, frame)
            )
            return
        self._manual_base_override = copy.deepcopy(msg)
        self._manual_base_override.header.frame_id = self._map_frame
        self._manual_base_override.header.stamp = self.get_clock().now().to_msg()
        pose = self._manual_base_override.pose
        self._publish_event(
            "manual_grasp_base_set:x=%.2f,y=%.2f" % (pose.position.x, pose.position.y)
        )
        self.get_logger().info(
            "Stored explicit manual base: (%.2f, %.2f)."
            % (pose.position.x, pose.position.y)
        )

    def _on_manual_grasp_goal(self, msg: PoseStamped) -> None:
        """Run one RViz-selected navigate -> Gemini grasp -> return mission."""
        if self._state not in (MissionState.IDLE, MissionState.DONE):
            self._publish_event(
                "manual_grasp_goal_rejected:mission_busy:%s" % self._state.value
            )
            self.get_logger().warn(
                "Ignoring RViz grasp goal while mission is %s." % self._state.value
            )
            return
        if self._robot_xy is None:
            self._publish_event("manual_grasp_goal_rejected:no_robot_pose")
            self.get_logger().error("Cannot record base: /slam/current_pose is missing.")
            return
        if not self._arm_integration_enabled:
            self._publish_event("manual_grasp_goal_rejected:arm_integration_disabled")
            self.get_logger().error("Manual grasp goal requires arm integration.")
            return
        frame = msg.header.frame_id.strip() or self._map_frame
        if frame != self._map_frame:
            self._publish_event(
                "manual_grasp_goal_rejected:frame=%s" % frame
            )
            self.get_logger().error(
                "RViz manual grasp goal must use frame %s, got %s."
                % (self._map_frame, frame)
            )
            return

        self._cancel_active_goal("manual_grasp_goal")
        self._stop_recovery("manual grasp start")
        self._disable_frontier()
        self._clear_frontier_unavailable()
        self._clear_pending_visual_candidate()
        self._clear_arm_action_state()
        self._locked_targets = []
        self._pending_target_completion = None
        self._target_posture_retry_count = 0
        self._unserviceable_targets = []
        self._pending_targets = []
        self._reached_target_count = 0
        self._target_goal_count = 1
        self._manual_mission_active = True
        manual_base_override = getattr(self, "_manual_base_override", None)
        if manual_base_override is not None:
            self._base_pose = copy.deepcopy(manual_base_override)
            self._manual_base_override = None
        else:
            self._base_pose = self._pose_message(
                self._robot_xy[0], self._robot_xy[1], self._robot_yaw
            )
        self._target_pose = copy.deepcopy(msg)
        self._target_pose.header.frame_id = self._map_frame
        self._target_pose.header.stamp = self.get_clock().now().to_msg()
        self._state = MissionState.MANUAL_NAVIGATING

        goal = NavigateToPose.Goal()
        goal.pose = copy.deepcopy(self._target_pose)
        self._publish_event(
            "manual_grasp_mission_started:base=(%.2f,%.2f),goal=(%.2f,%.2f)"
            % (
                self._base_pose.pose.position.x,
                self._base_pose.pose.position.y,
                goal.pose.pose.position.x,
                goal.pose.pose.position.y,
            )
        )
        self.get_logger().warn(
            "RViz manual grasp mission: base=(%.2f, %.2f), goal=(%.2f, %.2f)."
            % (
                self._base_pose.pose.position.x,
                self._base_pose.pose.position.y,
                goal.pose.pose.position.x,
                goal.pose.pose.position.y,
            )
        )
        self._send_goal(goal, "manual_grasp")

    def _on_stop(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        self._cancel_active_goal("mission_stopped")
        self._reset_to_idle("mission_stopped")
        response.success = True
        response.message = "mission stopped"
        return response

    def _on_abort(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        self._cancel_active_goal("mission_aborted")
        self._reset_to_idle("mission_aborted")
        response.success = True
        response.message = "mission aborted"
        return response

    # ------------------------------------------------------------------
    # Mission transitions
    # ------------------------------------------------------------------

    def _begin_mission(self, reason: str) -> None:
        self._manual_mission_active = False
        self._state = MissionState.SEARCHING
        self._base_pose = None
        self._target_pose = None
        self._reached_target_count = 0
        self._locked_targets = []
        self._pending_target_completion = None
        self._target_posture_retry_count = 0
        self._unserviceable_targets = []
        self._pending_targets = []
        self._frontier_goal_count = 0
        self._candidate_xy = None
        self._clear_pending_visual_candidate()
        self._rejected_candidates.clear()
        self._fence_applied = False
        self._scan_attempts = 0
        self._scan_rotations_at_location = 0
        self._scan_segment_index = 0
        self._scan_pause_until = 0.0
        self._scan_failed_segments_streak = 0
        self._slam_loss_since = 0.0
        self._scan_visited_points.clear()
        if self._robot_xy is not None:
            self._scan_visited_points.append(self._robot_xy)
        self._scan_best_candidate = None
        self._confirm_until = 0.0
        self._target_high_confidence_frames = 0
        self._target_low_confidence_frames = 0
        self._target_verify_until = 0.0
        self._approach_visual_loss_deadline = 0.0
        self._failed_approach_goals.clear()
        self._approach_standoff_attempts = 0
        self._last_approach_goal_xy = None
        self._approach_topup_used = 0
        self._approach_direct_topup = False
        self._approach_source = ""
        self._memory_reanchor_candidate = None
        self._memory_reanchor_frames = 0
        self._memory_reanchor_used = False
        self._goal_handle = None
        self._goal_pending = False
        self._goal_active = False
        self._goal_label = ""
        self._goal_start_time = 0.0
        self._spin_handle = None
        self._spin_active = False
        self._spin_start_time = 0.0
        self._cancel_return_alignment("mission start")
        self._clear_arm_action_state()
        self._next_goal_allowed_time = 0.0
        self._hold_until = 0.0
        self._target_posture_phase = ""
        self._clear_target_posture_ack()
        self._active_return_xy = None
        self._mission_start_time = self.get_clock().now().nanoseconds / 1e9
        self._clear_frontier_unavailable()
        self._stop_recovery("mission start")
        self._publish_event("mission_started:%s" % reason)
        self.get_logger().warn(
            "Mission started (%s); searching for %d target(s)."
            % (reason, self._target_goal_count)
        )
        # A staged launch intentionally leaves Frontier disabled.  Starting a
        # mission must transfer it into active exploration even when no visual
        # candidate is already in the head camera; otherwise SEARCHING can sit
        # idle forever with no velocity command and no task event.
        self._enable_frontier()

    def _try_start_approach(self, source: str) -> None:
        if self._target_pose is None:
            self.get_logger().warn(
                "Target event received but no /perception/target_pose yet; waiting.",
                throttle_duration_sec=5.0,
            )
            return
        self._start_visual_approach(
            float(self._target_pose.pose.position.x),
            float(self._target_pose.pose.position.y),
            self._candidate_confidence_threshold,
            source,
        )

    def _start_visual_approach(
        self, x: float, y: float, confidence: float, source: str
    ) -> None:
        """Temporarily preempt Frontier to inspect a visual candidate.

        A stable high-confidence candidate can be locked for this one
        approach.  Locking freezes its map position and prevents near-field
        camera loss from cancelling a valid Nav2 approach; the target is
        counted only after geometric arrival.
        """
        if self._state != MissionState.SEARCHING:
            return
        if target_is_locked(x, y, self._locked_targets, self._ignored_target_radius_m):
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if candidate_is_rejected(
            x,
            y,
            self._rejected_candidates,
            self._candidate_reject_radius_m,
            now,
            self._candidate_retry_cooldown_s,
        ):
            return
        stored = self._pose_message(x, y, 0.0)
        self._target_pose = stored
        self._candidate_xy = (x, y)
        self._clear_pending_visual_candidate()
        self._target_high_confidence_frames = 0
        self._target_low_confidence_frames = 0
        self._target_verify_until = 0.0
        self._approach_target_locked = bool(
            self._lock_visual_goal_after_stable_candidate
            and confidence >= self._locked_approach_min_confidence
        )
        self._approach_source = source
        self._memory_reanchor_candidate = None
        self._memory_reanchor_frames = 0
        self._memory_reanchor_used = False
        self._approach_visual_loss_deadline = (
            now + self._approach_visual_loss_grace_s
        )
        # A new candidate gets a fresh standoff-retry and top-up budget.
        self._approach_standoff_attempts = 0
        self._approach_topup_used = 0
        self._approach_direct_topup = False
        self._cancel_active_goal("visual_candidate")
        self._state = MissionState.APPROACHING
        self._next_goal_allowed_time = now + self._goal_cancel_settle_s
        self._disable_frontier()
        self._publish_event(
            "mission_visual_approach_started:conf=%.2f,x=%.2f,y=%.2f,source=%s,locked=%s"
            % (confidence, x, y, source, self._approach_target_locked)
        )
        self.get_logger().warn(
            "Visual candidate (conf=%.2f, locked=%s) preempted Frontier; "
            "approaching fixed standoff at (%.2f, %.2f)."
            % (confidence, self._approach_target_locked, x, y)
        )

    def _observe_approach_confidence(
        self,
        confidence: Optional[float],
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """Apply low-confidence abandonment and near-target confirmation."""
        if self._state not in (
            MissionState.APPROACHING,
            MissionState.VERIFYING_TARGET,
        ):
            return
        # A memory-sourced route must earn a fresh visual lock during this
        # approach.  Run this for both associated observations (memory was
        # already accurate) and displaced observations (memory needs one
        # correction).  After four stable frames the anchor is frozen again,
        # preserving the useful near-field camera-loss behavior.
        if (
            x is not None
            and y is not None
            and self._try_reanchor_memory_approach(confidence, x, y)
        ):
            return
        # Spatial association: only a detection near the selected target may
        # feed its counters.  A displaced estimate while following a locked
        # target is usually the known RGB-D depth jump, not a trustworthy new
        # object.  Ignore it and keep going to the frozen coordinate.  Other
        # targets are remembered during the segmented scan phase.
        if (
            x is not None
            and y is not None
            and self._target_pose is not None
            and math.hypot(
                x - float(self._target_pose.pose.position.x),
                y - float(self._target_pose.pose.position.y),
            ) > self._target_association_radius_m
        ):
            if not getattr(self, "_approach_target_locked", False) and (
                confidence is not None
                and confidence >= self._candidate_confidence_threshold
            ):
                self._record_pending_target(x, y, confidence)
            # This frame belongs to another object.  Treat the selected target
            # as unseen without allowing the distractor to retarget the route.
            # Locked approaches intentionally ignore this loss and continue to
            # their fixed coordinate; verification remains honest.
            self._observe_approach_confidence(None)
            return
        if (
            self._state == MissionState.VERIFYING_TARGET
            and x is not None
            and y is not None
            and self._target_pose is not None
            and math.hypot(
                x - float(self._target_pose.pose.position.x),
                y - float(self._target_pose.pose.position.y),
            ) <= self._target_association_radius_m
        ):
            # The Nav2 approach goal is already complete.  It is now safe to
            # refine the recorded target position from close RGB-D estimates.
            # Do not move it while APPROACHING: that would move the arrival
            # ring after the goal has been issued.
            self._target_pose.pose.position.x = x
            self._target_pose.pose.position.y = y

        if confidence is None or confidence < self._target_loss_confidence_threshold:
            if (
                self._state == MissionState.APPROACHING
                and getattr(self, "_approach_target_locked", False)
            ):
                # The selected target was already stable and high-confidence
                # before navigation started.  Do not discard its fixed map
                # pose merely because it has passed out of the head camera's
                # near-field view.
                return
            self._target_low_confidence_frames += 1
            self._target_high_confidence_frames = 0
            if self._state == MissionState.APPROACHING:
                now = self.get_clock().now().nanoseconds / 1e9
                if now < self._approach_visual_loss_deadline:
                    # Continue toward the already validated, fixed standoff
                    # pose.  The final confirmation is deliberately deferred
                    # until the robot is close and stationary.
                    return
            if self._target_low_confidence_frames >= self._target_loss_required_frames:
                self._abandon_visual_candidate("confidence_lost")
            return

        self._target_low_confidence_frames = 0
        if self._state == MissionState.APPROACHING:
            # Any fresh detection refreshes the grace window.  We intentionally
            # do not update _target_pose here: moving the target while a Nav2
            # goal is active makes the arrival/standoff check unstable.
            now = self.get_clock().now().nanoseconds / 1e9
            self._approach_visual_loss_deadline = (
                now + self._approach_visual_loss_grace_s
            )
            return
        if self._state != MissionState.VERIFYING_TARGET:
            return
        if confidence >= self._target_confirm_confidence_threshold:
            self._target_high_confidence_frames += 1
            if self._target_high_confidence_frames >= self._target_confirm_required_frames:
                self._confirm_visual_target(confidence)
        else:
            self._target_high_confidence_frames = 0

    def _try_reanchor_memory_approach(
        self, confidence: Optional[float], x: float, y: float
    ) -> bool:
        """Correct one stale remembered coordinate from stable fresh vision.

        The normal visual lock remains immutable.  Only an approach sourced
        from pending memory may use this one-shot correction, and only after a
        bounded cluster of high-confidence observations.  Returning ``True``
        means the observation belongs to the re-anchor decision and must not
        be recorded as another target or counted as loss of the old one.
        """
        if (
            self._state != MissionState.APPROACHING
            or getattr(self, "_approach_source", "") != "pending_memory"
            or getattr(self, "_memory_reanchor_used", False)
            or self._target_pose is None
            or confidence is None
            or confidence < self._target_confirm_confidence_threshold
        ):
            return False

        old_x = float(self._target_pose.pose.position.x)
        old_y = float(self._target_pose.pose.position.y)

        # Decide observability from the remembered target, not from the new
        # detection.  In the near blind zone the detector can lock onto a
        # different yellow patch farther away; using that patch's range here
        # would incorrectly make it eligible to overwrite the real target.
        if self._robot_xy is not None:
            camera_x = self._robot_xy[0] + (
                self._memory_reanchor_camera_forward_offset_m
                * math.cos(self._robot_yaw)
            )
            camera_y = self._robot_xy[1] + (
                self._memory_reanchor_camera_forward_offset_m
                * math.sin(self._robot_yaw)
            )
            expected_dx = old_x - camera_x
            expected_dy = old_y - camera_y
            # Horizontal projection in map XY.  Target/camera height is
            # intentionally excluded from the near-blind-zone decision.
            expected_distance = math.hypot(expected_dx, expected_dy)
            expected_world_bearing = math.atan2(expected_dy, expected_dx)
            expected_camera_bearing = math.atan2(
                math.sin(expected_world_bearing - self._robot_yaw),
                math.cos(expected_world_bearing - self._robot_yaw),
            )
            expected_visible = (
                self._memory_reanchor_min_visible_distance_m
                <= expected_distance
                <= self._memory_reanchor_max_visible_distance_m
                and abs(expected_camera_bearing)
                <= self._memory_reanchor_half_fov_rad
            )
            if not expected_visible:
                # Consume the observation without treating it as another
                # target or as evidence that the selected target was lost.
                # Navigation continues toward the frozen memory anchor.
                self._memory_reanchor_candidate = None
                self._memory_reanchor_frames = 0
                return True

        correction = math.hypot(x - old_x, y - old_y)
        if correction > self._memory_reanchor_max_correction_m:
            return False

        pending = self._memory_reanchor_candidate
        if pending is None or math.hypot(x - pending[0], y - pending[1]) > (
            self._memory_reanchor_stability_radius_m
        ):
            self._memory_reanchor_candidate = (x, y, confidence)
            self._memory_reanchor_frames = 1
        else:
            frames = self._memory_reanchor_frames
            weight = min(frames, 4)
            self._memory_reanchor_candidate = (
                (pending[0] * weight + x) / (weight + 1),
                (pending[1] * weight + y) / (weight + 1),
                max(pending[2], confidence),
            )
            self._memory_reanchor_frames += 1

        self._publish_event(
            "mission_pending_target_reanchor_stabilizing:%d/%d"
            % (self._memory_reanchor_frames, self._memory_reanchor_required_frames)
        )
        if self._memory_reanchor_frames < self._memory_reanchor_required_frames:
            return True

        corrected = self._memory_reanchor_candidate
        assert corrected is not None
        new_x, new_y = corrected[0], corrected[1]
        correction = math.hypot(new_x - old_x, new_y - old_y)
        self._cancel_nav_goal("pending_memory_reanchor")
        self._target_pose = self._pose_message(new_x, new_y, 0.0)
        self._candidate_xy = (new_x, new_y)
        self._approach_source = "pending_memory_reanchored"
        # A remembered coordinate is mutable only during this explicit,
        # multi-frame re-anchor.  Freeze it again immediately afterwards so
        # near-field target loss or a later depth outlier cannot cancel or
        # redirect the approach.
        self._approach_target_locked = True
        self._memory_reanchor_used = True
        self._memory_reanchor_candidate = None
        self._memory_reanchor_frames = 0
        self._approach_standoff_attempts = 0
        self._approach_topup_used = 0
        self._approach_direct_topup = False
        self._last_approach_goal_xy = None
        self._failed_approach_goals.clear()
        self._pending_targets = [
            item
            for item in self._pending_targets
            if math.hypot(item[0] - new_x, item[1] - new_y)
            > self._target_association_radius_m
        ]
        now = self.get_clock().now().nanoseconds / 1e9
        self._approach_visual_loss_deadline = (
            now + self._approach_visual_loss_grace_s
        )
        self._next_goal_allowed_time = now + self._goal_cancel_settle_s
        self._publish_event(
            "mission_pending_target_reanchored:old=(%.2f,%.2f),new=(%.2f,%.2f),offset=%.2f"
            % (old_x, old_y, new_x, new_y, correction)
        )
        self.get_logger().warn(
            "Remembered target corrected once from (%.2f, %.2f) to "
            "(%.2f, %.2f), offset %.2fm; corrected anchor is now locked."
            % (old_x, old_y, new_x, new_y, correction)
        )
        return True

    def _abandon_visual_candidate(self, reason: str) -> None:
        if self._state not in (
            MissionState.APPROACHING,
            MissionState.VERIFYING_TARGET,
        ):
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self._candidate_xy is not None:
            self._rejected_candidates.append(
                (self._candidate_xy[0], self._candidate_xy[1], now)
            )
        self._cancel_active_goal("visual_candidate_%s" % reason)
        self._publish_event("mission_visual_candidate_abandoned:%s" % reason)
        if getattr(self, "_stop_after_target_interaction", False):
            self.get_logger().warn(
                "Visual candidate abandoned (%s); single-target mode stops "
                "instead of resuming exploration." % reason
            )
            self._reset_to_idle(
                "mission_single_target_stopped:candidate_%s" % reason
            )
            return
        self.get_logger().warn(
            "Visual candidate abandoned (%s); resuming exploration." % reason
        )
        self._target_pose = None
        self._candidate_xy = None
        self._approach_target_locked = False
        self._approach_source = ""
        self._memory_reanchor_candidate = None
        self._memory_reanchor_frames = 0
        self._memory_reanchor_used = False
        self._clear_pending_visual_candidate()
        self._target_high_confidence_frames = 0
        self._target_low_confidence_frames = 0
        self._target_verify_until = 0.0
        self._approach_visual_loss_deadline = 0.0
        self._resume_search("visual_candidate_%s" % reason)

    def _confirm_visual_target(self, confidence: float) -> None:
        if self._state != MissionState.VERIFYING_TARGET or self._target_pose is None:
            return
        x = float(self._target_pose.pose.position.x)
        y = float(self._target_pose.pose.position.y)
        if target_is_locked(x, y, self._locked_targets, self._ignored_target_radius_m):
            self._abandon_visual_candidate("already_locked")
            return
        self._accept_confirmed_target(
            x, y, source="visual", confidence=confidence
        )

    def _accept_confirmed_target(
        self,
        x: float,
        y: float,
        *,
        source: str,
        confidence: float | None = None,
    ) -> None:
        """Lock a visual target, then either grasp it or count it directly."""
        self._locked_targets.append((x, y))
        self._publish_arm_target_report(x, y, source=source, confidence=confidence)
        if self._arm_integration_enabled:
            self._begin_arm_grasp(x, y, source=source, confidence=confidence)
            return
        # In the no-arm demonstration the crouch is the physical proof that
        # this target was serviced. Keep arrival provisional until firmware
        # acknowledges the complete crouch -> stand sequence.
        posture_enabled = bool(getattr(self, "_target_posture_enabled", False))
        get_parameter = getattr(self, "get_parameter", None)
        if callable(get_parameter):
            try:
                posture_enabled = bool(
                    get_parameter("target_posture_enabled").value
                )
            except Exception:  # noqa: BLE001 - test doubles need no ROS params
                pass
        if posture_enabled:
            self._pending_target_completion = (x, y, source, confidence)
            self._target_posture_retry_count = 0
        else:
            self._reached_target_count += 1
        self._enter_confirmed_target_hold(
            x,
            y,
            source=source,
            confidence=confidence,
        )

    def _publish_arm_target_report(
        self, x: float, y: float, *, source: str, confidence: float | None
    ) -> None:
        """Report one confirmed target to the arm stack (dog -> arm).

        The target does not need to stay in the head camera's view: besides
        the map-frame position, the report carries a transform-free bearing
        and distance measured from the current robot pose, so an arm-mounted
        camera can start searching around that direction without any
        calibrated map-to-arm transform.  bearing_rad is CCW-positive in the
        robot frame, normalized to [-pi, pi].
        """
        self._arm_report_sequence += 1
        payload = {
            "seq": self._arm_report_sequence,
            "event": "target_confirmed",
            "source": source,
            "frame": self._map_frame,
            "target_xy": [round(x, 3), round(y, 3)],
            "robot_xy": (
                None
                if self._robot_xy is None
                else [round(self._robot_xy[0], 3), round(self._robot_xy[1], 3)]
            ),
            "robot_yaw_rad": round(self._robot_yaw, 3),
            "bearing_rad": None,
            "distance_m": None,
            "confidence": confidence,
            "stamp_s": round(self.get_clock().now().nanoseconds / 1e9, 3),
        }
        if self._robot_xy is not None:
            world_yaw = math.atan2(y - self._robot_xy[1], x - self._robot_xy[0])
            bearing = math.atan2(
                math.sin(world_yaw - self._robot_yaw),
                math.cos(world_yaw - self._robot_yaw),
            )
            payload["bearing_rad"] = round(bearing, 3)
            payload["distance_m"] = round(
                math.hypot(x - self._robot_xy[0], y - self._robot_xy[1]), 3
            )
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._arm_report_pub.publish(msg)
        self.get_logger().warn(
            "Target report #%d (source=%s): map=(%.2f, %.2f), bearing=%.1fdeg, "
            "distance=%.2fm."
            % (
                self._arm_report_sequence,
                source,
                x,
                y,
                0.0
                if payload["bearing_rad"] is None
                else math.degrees(payload["bearing_rad"]),
                0.0 if payload["distance_m"] is None else payload["distance_m"],
            )
        )

    def _begin_arm_grasp(
        self,
        x: float,
        y: float,
        *,
        source: str,
        confidence: float | None,
    ) -> None:
        """Park the base and wait for a fresh arm-mounted-camera target."""
        now = self.get_clock().now().nanoseconds / 1e9
        self._cancel_nav_goal("arm_grasp_handoff")
        self._cancel_spin("arm_grasp_handoff")
        self._disable_frontier()
        # StopMove is best-effort here.  The Nav2 goal has already completed,
        # while cmd_vel safety still owns the actual velocity output.
        self._call_trigger(self._sport_stop_client, "sport StopMove before grasp")
        self._state = MissionState.WAITING_ARM_TARGET
        self._arm_wait_started_at = now
        self._arm_wait_deadline = now + self._arm_target_wait_timeout_s
        self._arm_target_xy = (x, y)
        self._arm_target_source = source
        self._arm_target_confidence = confidence
        self._arm_feedback_stage = ""
        self._publish_event(
            "mission_arm_waiting_for_target:timeout=%.1fs,x=%.2f,y=%.2f"
            % (self._arm_target_wait_timeout_s, x, y)
        )
        self.get_logger().warn(
            "Target confirmed at (%.2f, %.2f); base parked while waiting "
            "for a fresh arm-camera grasp pose." % (x, y)
        )

    def _send_arm_grasp_goal(self, pose: PoseStamped) -> None:
        if not self._arm_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn(
                "/arm/grasp_target unavailable; waiting until arm-target timeout.",
                throttle_duration_sec=2.0,
            )
            return
        self._arm_target_sequence += 1
        goal = GraspTarget.Goal()
        goal.target_pose = copy.deepcopy(pose)
        goal.target_id = "mission_target_%03d" % self._arm_target_sequence
        self._state = MissionState.GRASPING
        self._arm_goal_pending = True
        self._arm_goal_active = False
        self._arm_goal_start_time = self.get_clock().now().nanoseconds / 1e9
        self._arm_goal_generation += 1
        generation = self._arm_goal_generation
        self._publish_event("mission_arm_grasp_requested:%s" % goal.target_id)
        future = self._arm_client.send_goal_async(
            goal,
            feedback_callback=lambda feedback, gen=generation: self._on_arm_feedback(
                feedback, gen
            ),
        )
        future.add_done_callback(
            lambda done, gen=generation: self._on_arm_goal_response(done, gen)
        )

    def _on_arm_feedback(self, feedback_msg, generation: int) -> None:
        if generation != self._arm_goal_generation:
            return
        stage = str(feedback_msg.feedback.stage).strip()
        if not stage or stage == self._arm_feedback_stage:
            return
        self._arm_feedback_stage = stage
        self._publish_event(
            "mission_arm_grasp_feedback:%s,progress=%.2f"
            % (stage, float(feedback_msg.feedback.progress))
        )

    def _on_arm_goal_response(self, future, generation: int) -> None:
        if generation != self._arm_goal_generation:
            return
        self._arm_goal_pending = False
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._finish_arm_grasp(False, "send_failed:%s" % exc)
            return
        if not handle.accepted:
            self._finish_arm_grasp(False, "goal_rejected")
            return
        self._arm_goal_handle = handle
        self._arm_goal_active = True
        self._publish_event("mission_arm_grasp_started")
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda done, gen=generation: self._on_arm_goal_result(done, gen)
        )

    def _on_arm_goal_result(self, future, generation: int) -> None:
        if generation != self._arm_goal_generation:
            return
        self._arm_goal_active = False
        self._arm_goal_handle = None
        try:
            wrapped = future.result()
            succeeded = (
                wrapped.status == GoalStatus.STATUS_SUCCEEDED
                and bool(wrapped.result.success)
            )
            message = str(wrapped.result.message).strip()
            reason = message or "status_%d" % int(wrapped.status)
        except Exception as exc:  # noqa: BLE001
            succeeded = False
            reason = "result_read_failed:%s" % exc
        self._finish_arm_grasp(succeeded, reason)

    def _finish_arm_grasp(self, succeeded: bool, reason: str) -> None:
        """Complete the handshake and return ownership to navigation."""
        target_xy = self._arm_target_xy
        source = self._arm_target_source or "visual"
        confidence = self._arm_target_confidence
        self._clear_arm_action_state()
        if succeeded and target_xy is not None:
            self._reached_target_count += 1
            self._publish_event(
                "mission_arm_grasp_succeeded:%d/%d"
                % (self._reached_target_count, self._target_goal_count)
            )
            self._enter_confirmed_target_hold(
                target_xy[0],
                target_xy[1],
                source="%s+grasp" % source,
                confidence=confidence,
            )
            return
        self._publish_event("mission_arm_grasp_failed:%s" % reason)
        self.get_logger().error(
            "Arm grasp did not complete (%s); target is not counted and the "
            "mission will continue with a different target." % reason
        )
        self._target_pose = None
        self._candidate_xy = None
        if getattr(self, "_manual_mission_active", False):
            self._reset_to_idle("manual_grasp_mission_failed:%s" % reason)
            return
        self._resume_search("arm_grasp_failed")

    def _clear_arm_action_state(self) -> None:
        self._arm_goal_handle = None
        self._arm_goal_pending = False
        self._arm_goal_active = False
        self._arm_goal_start_time = 0.0
        self._arm_wait_started_at = 0.0
        self._arm_wait_deadline = 0.0
        self._arm_target_xy = None
        self._arm_target_source = ""
        self._arm_target_confidence = None
        self._arm_feedback_stage = ""

    def _enter_confirmed_target_hold(
        self,
        x: float,
        y: float,
        *,
        source: str,
        confidence: float | None = None,
    ) -> None:
        """Enter the common post-confirmation hold for every target source.

        A target can be confirmed either while approaching a visual candidate or
        after the scan-point confirmation route.  Both cases must pass through
        TARGET_HOLDING: it is where the optional crouch/stand sequence runs and
        where the decision to resume search or return to base is made.
        """
        self._target_verify_until = 0.0
        self._target_high_confidence_frames = 0
        self._target_low_confidence_frames = 0
        self._state = MissionState.TARGET_HOLDING
        self._hold_until = (
            self.get_clock().now().nanoseconds / 1e9 + self._target_seen_hold_s
        )
        pending_posture = getattr(self, "_pending_target_completion", None) is not None
        event_name = (
            "mission_target_arrival_pending_posture"
            if pending_posture
            else "mission_target_confirmed"
        )
        display_count = self._reached_target_count + (1 if pending_posture else 0)
        detail = (
            "%s:%d/%d,source=%s,x=%.2f,y=%.2f"
            % (event_name, display_count, self._target_goal_count, source, x, y)
        )
        if confidence is not None:
            detail += ",conf=%.2f" % confidence
        self._publish_event(detail)
        self.get_logger().warn(
            "Target %s from %s (%d/%d); holding %.1fs."
            % (
                "awaits posture confirmation" if pending_posture else "confirmed",
                source,
                display_count,
                self._target_goal_count,
                self._target_seen_hold_s,
            )
        )

    def _pose_inside_fence(self, x: float, y: float) -> bool:
        """Drop targets/candidates outside the usable exploration interior."""
        if not self._exploration_square_enabled or self._base_pose is None:
            return True
        inside = point_inside_square_with_margin(
            x,
            y,
            float(self._base_pose.pose.position.x),
            float(self._base_pose.pose.position.y),
            self._exploration_square_half_size_m,
            self._exploration_boundary_margin_m,
        )
        if not inside:
            self.get_logger().info(
                "Ignoring target outside usable %.0fm square or within %.2fm "
                "boundary band at (%.2f, %.2f)."
                % (
                    2.0
                    * (
                        self._exploration_square_half_size_m
                        - self._exploration_boundary_margin_m
                    ),
                    self._exploration_boundary_margin_m,
                    x,
                    y,
                ),
                throttle_duration_sec=5.0,
            )
        return inside

    def _resume_search(self, reason: str) -> None:
        self._state = MissionState.SEARCHING
        self._clear_frontier_unavailable()
        self._publish_event("mission_search_resumed:%s" % reason)
        # A remembered sighting beats blind frontier exploration.
        if self._consume_pending_target(reason):
            return
        self._enable_frontier()
        self.get_logger().info("Resuming frontier search (%s)." % reason)

    def _begin_scan_sequence(self) -> None:
        self._clear_frontier_unavailable()
        self._disable_frontier()
        self._scan_attempts = 0
        self._scan_rotations_at_location = 0
        self._scan_visited_points.clear()
        if self._robot_xy is not None:
            self._scan_visited_points.append(self._robot_xy)
        self._scan_best_candidate = None
        self._clear_pending_visual_candidate()
        self._confirm_until = 0.0
        self._state = MissionState.SCANNING
        self._publish_event("mission_scan_started:attempt=1")
        self.get_logger().warn(
            "%d/%d frontier goals reached with %d/%d targets; starting "
            "in-place scan."
            % (
                self._frontier_goal_count,
                self._frontier_goals_before_scan,
                self._reached_target_count,
                self._target_goal_count,
            )
        )
        self._start_segmented_scan()

    def _begin_return(self) -> None:
        if self._state == MissionState.RETURNING:
            return
        if self._reached_target_count < self._target_goal_count:
            # Return is the terminal success transition.  Keeping this guard
            # here prevents an old scan/recovery callback from accidentally
            # ending a mission before all required targets are confirmed.
            self.get_logger().error(
                "Refusing return with only %d/%d confirmed targets."
                % (self._reached_target_count, self._target_goal_count)
            )
            self._resume_search("return_guard")
            return
        self._state = MissionState.RETURNING
        self._publish_event("mission_return_started")
        self._start_return_alignment_or_goal()

    def _reset_to_idle(self, event_name: str) -> None:
        self._cancel_return_alignment("mission reset")
        self._state = MissionState.IDLE
        self._clear_frontier_unavailable()
        self._slam_loss_since = 0.0
        self._target_pose = None
        # Recorded positions belong to the aborted mission's map frame.
        self._pending_targets = []
        self._base_pose = None
        self._clear_pending_visual_candidate()
        self._goal_label = ""
        self._goal_start_time = 0.0
        self._spin_start_time = 0.0
        self._next_goal_allowed_time = 0.0
        self._hold_until = 0.0
        if self._target_posture_phase:
            # Do not leave the map bridge permanently paused if an operator
            # stops a mission during the optional posture marker.
            self._call_set_bool(
                self._slam_jump_guard_monitoring_client,
                True,
                "resume SLAM jump monitoring",
            )
            self._call_set_bool(
                self._velocity_odom_jump_monitoring_client,
                True,
                "resume velocity odom jump monitoring",
            )
            self._call_set_bool(
                self._map_accumulation_client, True, "resume occupancy accumulation"
            )
        self._target_posture_phase = ""
        self._pending_target_completion = None
        self._target_posture_retry_count = 0
        self._clear_target_posture_ack()
        self._target_high_confidence_frames = 0
        self._target_low_confidence_frames = 0
        self._target_verify_until = 0.0
        self._approach_source = ""
        self._memory_reanchor_candidate = None
        self._memory_reanchor_frames = 0
        self._memory_reanchor_used = False
        self._stop_recovery("mission reset")
        self._publish_event(event_name)
        self.get_logger().warn("Mission reset to IDLE (%s)." % event_name)

    # ------------------------------------------------------------------
    # Nav2 NavigateToPose goals
    # ------------------------------------------------------------------

    def _send_approach_goal(self, label: str = "approach") -> None:
        if self._target_pose is None or self._robot_xy is None:
            self.get_logger().warn(
                "Approach needs both target pose and robot pose.",
                throttle_duration_sec=5.0,
            )
            return

        target_x = float(self._target_pose.pose.position.x)
        target_y = float(self._target_pose.pose.position.y)
        robot_x, robot_y = self._robot_xy

        # A top-up walk aims straight at the frozen anchor along the
        # robot->target ray; the ring picker is deliberately bypassed so the
        # robot closes distance instead of circling to another ring point.
        direct_topup = bool(getattr(self, "_approach_direct_topup", False))
        self._approach_direct_topup = False
        if self._latest_map is not None and not direct_topup:
            goal_xy, yaw = self._map_standoff_goal(target_x, target_y, robot_x, robot_y)
            if goal_xy is None:
                if label == "confirm":
                    self._publish_event(
                        "mission_confirm_failed:no_standoff_candidates"
                    )
                    self._on_confirm_failed("no_standoff_candidates")
                else:
                    self._publish_event(
                        "mission_approach_failed:no_standoff_candidates"
                    )
                    self._on_approach_failed("no_standoff_candidates")
                return
        else:
            direct_standoff_m = (
                getattr(
                    self,
                    "_approach_topup_standoff_m",
                    self._approach_standoff_m,
                )
                if direct_topup
                else self._approach_standoff_m
            )
            goal_xy = direct_approach_goal_xy(
                robot_x,
                robot_y,
                target_x,
                target_y,
                direct_standoff_m,
            )
            yaw = yaw_to_face_target(target_x, target_y, goal_xy[0], goal_xy[1])
            self.get_logger().info(
                "Direct standoff goal (%s): (%.2f, %.2f)."
                % (
                    "top-up walk" if direct_topup else "no /navigation_map yet",
                    goal_xy[0],
                    goal_xy[1],
                )
            )

        goal = NavigateToPose.Goal()
        goal.pose = self._pose_message(goal_xy[0], goal_xy[1], yaw)
        self._last_approach_goal_xy = (goal_xy[0], goal_xy[1])
        self.get_logger().info(
            "APPROACHING: goal=(%.2f, %.2f) yaw=%.1fdeg target=(%.2f, %.2f)"
            % (
                goal_xy[0],
                goal_xy[1],
                math.degrees(yaw),
                target_x,
                target_y,
            )
        )
        self._send_goal(goal, label)

    def _map_standoff_goal(
        self,
        target_x: float,
        target_y: float,
        robot_x: float,
        robot_y: float,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Pick a safe standoff pose from the occupancy grid, or (None, 0)."""
        assert self._latest_map is not None
        explorer = FrontierExplorer(
            safety_radius_m=self._approach_safety_radius_m,
            free_threshold=65,
            occupied_threshold=90,
        )
        candidates = explorer.find_approach_standoffs(
            target_x=target_x,
            target_y=target_y,
            robot_x=robot_x,
            robot_y=robot_y,
            width=int(self._latest_map.info.width),
            height=int(self._latest_map.info.height),
            resolution=float(self._latest_map.info.resolution),
            origin_x=float(self._latest_map.info.origin.position.x),
            origin_y=float(self._latest_map.info.origin.position.y),
            data=list(self._latest_map.data),
            standoff_m=self._approach_standoff_m,
            rejected_goals=self._failed_approach_goals,
            rejected_radius_m=self._approach_failed_radius_m,
            num_angles=self._approach_num_angles,
            allow_unknown=self._approach_allow_unknown_cells,
            prefer_known=True,
            max_candidates=self._approach_max_candidates,
        )
        candidate = pick_approach_candidate(
            candidates,
            self._failed_approach_goals,
            self._approach_failed_radius_m,
        )
        if candidate is None:
            self.get_logger().error(
                "No approach standoff candidate near (%.2f, %.2f)."
                % (target_x, target_y)
            )
            return None, 0.0
        return (candidate.x, candidate.y), candidate.yaw

    def _send_return_goal(self) -> None:
        if self._base_pose is None:
            self.get_logger().error(
                "Cannot return: base pose not recorded.", throttle_duration_sec=5.0
            )
            self._publish_event("mission_return_failed:no_base_pose")
            return
        base_x = float(self._base_pose.pose.position.x)
        base_y = float(self._base_pose.pose.position.y)
        return_xy = (base_x, base_y)
        if self._latest_map is not None:
            corrected = find_nearest_free_point(
                width=int(self._latest_map.info.width),
                height=int(self._latest_map.info.height),
                resolution=float(self._latest_map.info.resolution),
                origin_x=float(self._latest_map.info.origin.position.x),
                origin_y=float(self._latest_map.info.origin.position.y),
                data=list(self._latest_map.data),
                target_x=base_x,
                target_y=base_y,
                search_radius_m=self._return_target_search_radius_m,
                free_threshold=self._return_free_threshold,
            )
            if corrected is None:
                self._publish_event(
                    "mission_return_target_unusable:no_free_cell_near_base"
                )
                self.get_logger().error(
                    "No known-free return cell within %.2fm of base (%.2f, %.2f)."
                    % (self._return_target_search_radius_m, base_x, base_y)
                )
                return
            return_xy = corrected
            correction = math.hypot(corrected[0] - base_x, corrected[1] - base_y)
            if correction > 0.5 * float(self._latest_map.info.resolution):
                self._publish_event(
                    "mission_return_target_corrected:base=(%.2f,%.2f),goal=(%.2f,%.2f),offset=%.2f"
                    % (base_x, base_y, corrected[0], corrected[1], correction)
                )
        self._active_return_xy = return_xy
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self._map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        # Keep the originally recorded base pose immutable: a corrected return
        # target is specific to the current map state and must not rewrite it.
        goal.pose.pose = copy.deepcopy(self._base_pose.pose)
        goal.pose.pose.position.x = return_xy[0]
        goal.pose.pose.position.y = return_xy[1]
        if self._robot_xy is not None:
            # Arrive facing along the approach direction instead of the
            # mission-start heading.  Deep-copying the recorded base pose
            # carried its original orientation into the goal, and Nav2 kept
            # rotating in place at the base to restore it (2026-09-27 field
            # feedback).  Approaching face-on already satisfies the goal
            # checker's yaw tolerance from any direction.
            yaw = math.atan2(
                return_xy[1] - self._robot_xy[1],
                return_xy[0] - self._robot_xy[0],
            )
            goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.get_logger().info(
            "RETURNING: base=(%.2f, %.2f)"
            % (goal.pose.pose.position.x, goal.pose.pose.position.y)
        )
        self._send_goal(goal, "return")

    def _start_return_alignment_or_goal(self) -> None:
        """Turn once toward the base before handing the route to DWB.

        A near-180-degree initial heading gives the non-holonomic controller
        two almost equivalent left/right solutions.  Frequent global-path
        updates can then flip the chosen sign and make the robot wag in place.
        This Spin goal freezes one relative turn direction before normal Nav2
        return navigation starts.
        """
        if (
            not self._return_prealign_enabled
            or self._base_pose is None
            or self._robot_xy is None
        ):
            self._send_return_goal()
            return

        base_x = float(self._base_pose.pose.position.x)
        base_y = float(self._base_pose.pose.position.y)
        yaw_to_base = math.atan2(
            base_y - self._robot_xy[1], base_x - self._robot_xy[0]
        )
        delta = (yaw_to_base - self._robot_yaw + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        # Pick one deterministic direction at the otherwise ambiguous pi
        # boundary.  Tiny localization noise must not alternate the sign.
        if math.isclose(abs(delta), math.pi, abs_tol=math.radians(0.5)):
            delta = math.pi
        if abs(delta) <= self._return_prealign_threshold_rad:
            self._publish_event(
                "mission_return_prealign_skipped:yaw_deg=%.1f"
                % math.degrees(delta)
            )
            self._send_return_goal()
            return
        if not self._return_spin_client.wait_for_server(timeout_sec=1.0):
            self._publish_event("mission_return_prealign_unavailable_fallback")
            self.get_logger().warn(
                "Return Spin action unavailable; falling back to NavigateToPose."
            )
            self._send_return_goal()
            return

        goal = Spin.Goal()
        goal.target_yaw = delta
        self._return_spin_handle = None
        self._return_alignment_active = True
        self._return_alignment_start_time = (
            self.get_clock().now().nanoseconds / 1e9
        )
        self._return_spin_generation += 1
        generation = self._return_spin_generation
        self._publish_event(
            "mission_return_prealign_started:yaw_deg=%.1f"
            % math.degrees(delta)
        )
        self.get_logger().info(
            "Pre-aligning %.1fdeg toward base before return navigation."
            % math.degrees(delta)
        )
        future = self._return_spin_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done, gen=generation: self._on_return_spin_response(done, gen)
        )

    def _on_return_spin_response(self, future, generation: int) -> None:
        if generation != self._return_spin_generation:
            return
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Sending return Spin failed: %s" % exc)
            self._finish_return_alignment(False, "send_failed")
            return
        if not handle.accepted:
            self.get_logger().warn("Nav2 rejected the return pre-alignment Spin.")
            self._finish_return_alignment(False, "rejected")
            return
        self._return_spin_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda done, gen=generation: self._on_return_spin_done(done, gen)
        )

    def _on_return_spin_done(self, future, generation: int) -> None:
        if generation != self._return_spin_generation:
            return
        try:
            status = future.result().status
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Reading return Spin result failed: %s" % exc)
            status = GoalStatus.STATUS_ABORTED
        self._finish_return_alignment(
            status == GoalStatus.STATUS_SUCCEEDED,
            "status_%d" % int(status),
        )

    def _finish_return_alignment(self, success: bool, reason: str) -> None:
        if not getattr(self, "_return_alignment_active", False):
            return
        self._return_alignment_active = False
        self._return_alignment_start_time = 0.0
        self._return_spin_handle = None
        if success:
            self._publish_event("mission_return_prealign_finished")
        else:
            self._publish_event(
                "mission_return_prealign_failed_fallback:%s" % reason
            )
            self.get_logger().warn(
                "Return pre-alignment failed (%s); continuing with normal Nav2."
                % reason
            )
        if self._state == MissionState.RETURNING:
            self._send_return_goal()

    def _send_scan_point_goal(self) -> None:
        if self._robot_xy is None:
            self._publish_event("mission_scan_point_unavailable:no_robot_pose")
            self.get_logger().warn(
                "No robot pose for scan-point navigation.",
                throttle_duration_sec=5.0,
            )
            self._on_scan_point_nav_failed("no_robot_pose")
            return
        point = self._pick_distant_scan_point()
        if point is None:
            self._publish_event("mission_scan_point_unavailable:no_distant_free_point")
            self.get_logger().error(
                "No distant free scan point found; spinning in place again.",
                throttle_duration_sec=5.0,
            )
            self._on_scan_point_nav_failed("no_distant_scan_point")
            return
        self._scan_visited_points.append(point)
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_message(point[0], point[1], self._robot_yaw)
        self.get_logger().info(
            "NAVIGATING_TO_SCAN_POINT: (%.2f, %.2f)" % (point[0], point[1])
        )
        self._send_goal(goal, "scan_point")

    def _pick_distant_scan_point(self) -> Optional[Tuple[float, float]]:
        if self._latest_map is None or self._robot_xy is None:
            return None
        max_distance = self._scan_point_max_distance_m
        workspace_bounds = None
        if self._exploration_square_enabled and self._base_pose is not None:
            center_x = float(self._base_pose.pose.position.x)
            center_y = float(self._base_pose.pose.position.y)
            usable_half_size = (
                self._exploration_square_half_size_m
                - self._exploration_boundary_margin_m
            )
            edge_distance = distance_to_square_edge(
                self._robot_xy[0],
                self._robot_xy[1],
                center_x,
                center_y,
                usable_half_size,
            )
            max_distance = min(max_distance, edge_distance)
            workspace_bounds = square_fence_bounds(
                center_x, center_y, usable_half_size
            )
        min_distance = self._scan_point_min_distance_m
        if max_distance < min_distance:
            # A compact training area can still have a useful free point just
            # inside the fence.  Prefer it over repeatedly spinning in place.
            min_distance = max(0.20, 0.5 * max_distance)
        return find_farthest_free_point(
            width=int(self._latest_map.info.width),
            height=int(self._latest_map.info.height),
            resolution=float(self._latest_map.info.resolution),
            origin_x=float(self._latest_map.info.origin.position.x),
            origin_y=float(self._latest_map.info.origin.position.y),
            data=list(self._latest_map.data),
            robot_x=self._robot_xy[0],
            robot_y=self._robot_xy[1],
            min_distance_m=min_distance,
            max_distance_m=max_distance,
            workspace_bounds=workspace_bounds,
            excluded_points=list(self._scan_visited_points),
            excluded_radius_m=self._scan_point_revisit_radius_m,
        )

    def _send_goal(self, goal: NavigateToPose.Goal, label: str) -> None:
        self._goal_label = label
        self._publish_goal_pose(goal)
        if not self._nav_client.wait_for_server(timeout_sec=1.0):
            self._publish_event("navigation_failed:nav2_unavailable")
            self._on_goal_failed("nav2_unavailable")
            return

        self._goal_pending = True
        self._goal_active = False
        self._goal_handle = None
        self._goal_start_time = self.get_clock().now().nanoseconds / 1e9
        if label == "approach" and self._target_pose is not None and self._robot_xy is not None:
            self._approach_best_distance_m = math.hypot(
                float(self._target_pose.pose.position.x) - self._robot_xy[0],
                float(self._target_pose.pose.position.y) - self._robot_xy[1],
            )
            self._approach_last_progress_time = self._goal_start_time
        self._goal_generation += 1
        generation = self._goal_generation
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done, gen=generation: self._on_goal_response(done, gen)
        )

    def _publish_goal_pose(self, goal: NavigateToPose.Goal) -> None:
        msg = PoseStamped()
        msg.header.frame_id = self._map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = goal.pose.pose
        self._goal_pub.publish(msg)

    def _on_goal_response(self, future, generation: int) -> None:
        if generation != self._goal_generation:
            return
        self._goal_pending = False
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Sending Nav2 goal failed: %s" % exc)
            self._on_goal_failed("send_failed")
            return
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected the %s goal." % self._goal_label)
            self._on_goal_failed("goal_rejected")
            return
        self._goal_handle = goal_handle
        self._goal_active = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, gen=generation: self._on_goal_result(done, gen)
        )

    def _on_goal_result(self, future, generation: int) -> None:
        if generation != self._goal_generation:
            return
        self._goal_active = False
        self._goal_handle = None
        try:
            result = future.result()
            status = result.status
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Reading Nav2 result failed: %s" % exc)
            self._on_goal_failed("result_read_failed")
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._on_goal_succeeded()
        else:
            self._on_goal_failed("status_%d" % int(status))

    def _on_goal_succeeded(self) -> None:
        label = self._goal_label
        self._goal_label = ""
        if label == "approach":
            self._on_approach_succeeded()
        elif label == "confirm":
            self._on_confirm_arrived()
        elif label == "scan_point":
            self._on_scan_point_nav_succeeded()
        elif label == "return":
            self._complete_return("nav2_succeeded")
        elif label == "manual_grasp":
            if self._target_pose is None:
                self._reset_to_idle("manual_grasp_mission_failed:no_target_pose")
                return
            x = float(self._target_pose.pose.position.x)
            y = float(self._target_pose.pose.position.y)
            self._publish_event(
                "manual_grasp_goal_arrived:x=%.2f,y=%.2f" % (x, y)
            )
            self._begin_arm_grasp(
                x,
                y,
                source="manual_rviz",
                confidence=None,
            )
        else:
            self.get_logger().warn("Unknown succeeded goal label: %r" % label)

    def _on_goal_failed(self, reason: str) -> None:
        label = self._goal_label
        self._goal_label = ""
        if label == "approach":
            if (
                reason in ("timeout", "near_field_stall")
                and self._target_pose is not None
                and self._robot_xy is not None
            ):
                # A wall-side standoff can keep Nav2 in RUNNING while the
                # robot has reached the useful near field. Reuse the strict
                # frozen-anchor distance check from the success path: it may
                # accept the arrival band or schedule one bounded top-up, but
                # cannot accept a far timeout just because vision disappeared.
                self._publish_event(
                    "mission_approach_%s_geometric_recheck" % reason
                )
                self._on_approach_succeeded()
                return
            self._on_approach_failed(reason)
        elif label == "confirm":
            self._on_confirm_failed(reason)
        elif label == "scan_point":
            self._on_scan_point_nav_failed(reason)
        elif label == "return":
            distance = self._return_distance_m()
            self._publish_event(
                "mission_return_diagnostic:reason=%s,distance=%s,robot=%s,goal=%s"
                % (
                    reason,
                    "unknown" if distance is None else "%.2f" % distance,
                    "unknown"
                    if self._robot_xy is None
                    else "(%.2f,%.2f)" % self._robot_xy,
                    "unknown"
                    if self._active_return_xy is None
                    else "(%.2f,%.2f)" % self._active_return_xy,
                )
            )
            if (
                distance is not None
                and distance <= self._return_position_tolerance_m
            ):
                self._complete_return("position_tolerance_after_%s" % reason)
                return
            if self._start_recovery("return"):
                self.get_logger().warn(
                    "Return goal failed (%s); trying lateral/backup escape "
                    "before retrying." % reason
                )
                return
            self._publish_event("mission_return_failed:%s" % reason)
            self.get_logger().error(
                "Return-to-base failed (%s); supervisor stops to avoid retry loops."
                % reason
            )
        elif label == "manual_grasp":
            self._publish_event("manual_grasp_navigation_failed:%s" % reason)
            self._reset_to_idle("manual_grasp_mission_failed:navigation_%s" % reason)
        else:
            self.get_logger().error("Mission goal failed (%s)." % reason)

    # ------------------------------------------------------------------
    # Outcome handlers
    # ------------------------------------------------------------------

    def _on_approach_succeeded(self) -> None:
        if self._target_pose is not None and self._robot_xy is not None:
            arrival_max_distance_m = getattr(
                self,
                "_approach_arrival_max_distance_m",
                self._approach_standoff_m + self._approach_arrival_tolerance_m,
            )
            # Arrival is a horizontal map-plane distance.  Never fold target
            # or camera height into this value.
            distance = math.hypot(
                float(self._target_pose.pose.position.x) - self._robot_xy[0],
                float(self._target_pose.pose.position.y) - self._robot_xy[1],
            )
            if not arrival_within_tolerance(
                distance,
                self._approach_standoff_m,
                self._approach_arrival_tolerance_m,
                arrival_max_distance_m,
            ):
                if (
                    distance
                    > arrival_max_distance_m
                    and self._approach_topup_used < self._approach_topup_max
                    and distance <= self._approach_topup_max_distance_m
                ):
                    # Nav2 reported success short of the acceptance ring
                    # (measured against the frozen anchor).  Walk one bounded
                    # direct hop toward the anchor and re-check arrival
                    # instead of abandoning a visually good target.
                    self._approach_topup_used += 1
                    self._approach_direct_topup = True
                    self._publish_event(
                        "mission_approach_topup_started:attempt=%d,dist=%.3f,max=%.3f"
                        % (
                            self._approach_topup_used,
                            distance,
                            self._approach_topup_max_distance_m,
                        )
                    )
                    self.get_logger().warn(
                        "Nav2 stopped %.3fm from the target (beyond the %.3fm "
                        "acceptance edge); walking one direct top-up hop closer."
                        % (
                            distance,
                            arrival_max_distance_m,
                        )
                    )
                    # State stays APPROACHING; _control_step re-sends the goal.
                    return
                self._publish_event(
                    "mission_approach_failed:approach_not_close:dist=%.3f,max=%.3f"
                    % (distance, arrival_max_distance_m)
                )
                self._on_approach_failed("approach_not_close")
                return

        if (
            getattr(self, "_approach_target_locked", False)
            and getattr(self, "_approach_source", "") != "pending_memory"
            and self._target_pose is not None
        ):
            x = float(self._target_pose.pose.position.x)
            y = float(self._target_pose.pose.position.y)
            self._publish_event(
                "mission_target_geometric_arrival_confirmed:locked_visual,x=%.2f,y=%.2f"
                % (x, y)
            )
            self._accept_confirmed_target(
                x,
                y,
                source="locked_visual_standoff",
                confidence=None,
            )
            return

        if self._arm_integration_enabled and self._target_pose is not None:
            # The head-mounted D435i is used to choose and approach a target,
            # but the target can leave that camera's field of view at the
            # grasp standoff.  Once Nav2 has geometrically reached the fixed
            # standoff, transfer verification to the arm-mounted Gemini
            # instead of requiring more D435i frames.  WAITING_ARM_TARGET
            # still requires a fresh, timestamped Gemini pose; a false or
            # missing near-field detection therefore times out without
            # counting the target or commanding the arm.
            x = float(self._target_pose.pose.position.x)
            y = float(self._target_pose.pose.position.y)
            self._publish_event(
                "mission_arm_handoff_started:reason=standoff_reached,x=%.2f,y=%.2f"
                % (x, y)
            )
            self._accept_confirmed_target(
                x,
                y,
                source="visual_standoff",
                confidence=None,
            )
            return

        # Reaching the geometric standoff is not enough to count a target.
        # Without arm-camera integration, keep still and require fresh
        # high-confidence observations from the head camera at close range.
        self._state = MissionState.VERIFYING_TARGET
        self._target_high_confidence_frames = 0
        self._target_low_confidence_frames = 0
        self._target_verify_until = (
            self.get_clock().now().nanoseconds / 1e9
            + self._target_verify_timeout_s
        )
        self._publish_event(
            "mission_target_verify_started:timeout=%.1fs,high=%.2f,frames=%d,target_facing_standoff=true"
            % (
                self._target_verify_timeout_s,
                self._target_confirm_confidence_threshold,
                self._target_confirm_required_frames,
            )
        )
        self.get_logger().warn(
            "Reached visual standoff; waiting for stable high-confidence "
            "confirmation."
        )

    def _on_approach_failed(self, reason: str) -> None:
        self._publish_event("mission_approach_failed:%s" % reason)
        # 把刚失败的停靠点写入黑名单：find_approach_standoffs 与
        # pick_approach_candidate 都已按 _failed_approach_goals 过滤，
        # 但此前从未有代码向它写入——24 向停靠环永远取同一个候选。
        if self._last_approach_goal_xy is not None:
            self._failed_approach_goals.append(self._last_approach_goal_xy)
            self._last_approach_goal_xy = None
        if (
            self._approach_standoff_attempts < self._approach_standoff_retry_max
            and self._target_pose is not None
            and self._robot_xy is not None
        ):
            # 换一个方向的停靠点重试一次，而不是立刻放弃整个视觉候选。
            self._approach_standoff_attempts += 1
            self.get_logger().warn(
                "Approach failed (%s); retrying from a different standoff "
                "(retry %d of %d)."
                % (reason, self._approach_standoff_attempts, self._approach_standoff_retry_max)
            )
            self._publish_event(
                "mission_approach_standoff_retry:attempt=%d,reason=%s"
                % (self._approach_standoff_attempts, reason)
            )
            self._state = MissionState.APPROACHING
            self._next_goal_allowed_time = (
                self.get_clock().now().nanoseconds / 1e9
                + self._goal_cancel_settle_s
            )
            return
        self.get_logger().error(
            "Approach failed (%s); candidate is not counted." % reason
        )
        self._abandon_visual_candidate("approach_%s" % reason)

    def _on_scan_point_nav_succeeded(self) -> None:
        self._publish_event(
            "mission_scan_point_arrived:attempt=%d" % (self._scan_attempts + 1)
        )
        self.get_logger().info("Arrived at scan point; spinning again.")
        self._state = MissionState.SCANNING
        self._scan_rotations_at_location = 0
        self._publish_event(
            "mission_scan_started:attempt=%d" % (self._scan_attempts + 1)
        )
        self._start_segmented_scan()

    def _on_scan_point_nav_failed(self, reason: str) -> None:
        self._publish_event(
            "mission_scan_point_failed:attempt=%d,reason=%s"
            % (self._scan_attempts + 1, reason)
        )
        self.get_logger().warn(
            "Scan-point navigation failed (%s); spinning in place instead."
            % reason
        )
        self._state = MissionState.SCANNING
        self._publish_event(
            "mission_scan_started:attempt=%d" % (self._scan_attempts + 1)
        )
        self._start_segmented_scan()

    def _return_distance_m(self) -> Optional[float]:
        if self._robot_xy is None:
            return None
        target = self._active_return_xy
        if target is None and self._base_pose is not None:
            target = (
                float(self._base_pose.pose.position.x),
                float(self._base_pose.pose.position.y),
            )
        if target is None:
            return None
        return math.hypot(self._robot_xy[0] - target[0], self._robot_xy[1] - target[1])

    def _complete_return(self, reason: str) -> None:
        self._state = MissionState.DONE
        self._goal_label = ""
        self._publish_event("mission_arrived_base:%s" % reason)
        self.get_logger().warn("Mission complete: arrived at base (%s)." % reason)

    def _note_candidate_pose(self, x: float, y: float, confidence: float) -> None:
        """Remember the highest-confidence candidate seen during one spin."""
        if self._state != MissionState.SCANNING:
            return
        if target_is_locked(x, y, self._locked_targets, self._ignored_target_radius_m):
            return
        near_best = self._scan_best_candidate is not None and math.hypot(
            x - self._scan_best_candidate[0],
            y - self._scan_best_candidate[1],
        ) <= self._target_association_radius_m
        if (
            self._scan_best_candidate is None
            or confidence > self._scan_best_candidate[2]
        ):
            previous = self._scan_best_candidate
            self._scan_best_candidate = (x, y, confidence)
            # The displaced runner-up is a valid target sighting — unless it
            # is the same object as the new best (a fresher estimate of it).
            if previous is not None and not near_best:
                self._record_pending_target(previous[0], previous[1], previous[2])
        elif not near_best:
            # Stable (upstream stability filter) but lower confidence than
            # the current best, and spatially separate: still a valid target
            # sighting — remember it instead of discarding it.
            self._record_pending_target(x, y, confidence)

    def _record_pending_target(
        self, x: float, y: float, confidence: float | None
    ) -> None:
        """Queue a seen-but-not-chosen target position for a later visit."""
        if not bool(self.get_parameter("pending_target_memory_enabled").value):
            return
        if target_is_locked(x, y, self._locked_targets, self._ignored_target_radius_m):
            return
        # Never re-record the target currently being approached.
        if self._candidate_xy is not None and math.hypot(
            self._candidate_xy[0] - x, self._candidate_xy[1] - y
        ) <= self._target_association_radius_m:
            return
        for index, (px, py, old_confidence) in enumerate(self._pending_targets):
            if math.hypot(px - x, py - y) <= self._target_association_radius_m:
                # PoseStamped can arrive before the confidence-bearing status.
                # Merge later evidence instead of leaving a real target at 0.0.
                new_confidence = (
                    old_confidence if confidence is None else confidence
                )
                if new_confidence > old_confidence:
                    self._pending_targets[index] = (x, y, new_confidence)
                    self._publish_event(
                        "mission_pending_target_updated:x=%.2f,y=%.2f,conf=%.2f"
                        % (x, y, new_confidence)
                    )
                return
        if len(self._pending_targets) >= 8:
            self._pending_targets.pop(0)
        conf = confidence if confidence is not None else 0.0
        self._pending_targets.append((x, y, conf))
        self._publish_event(
            "mission_pending_target_recorded:x=%.2f,y=%.2f,conf=%.2f"
            % (x, y, conf)
        )

    def _consume_pending_target(self, reason: str) -> bool:
        """Start an approach to the best remembered target, if any.

        Returns True when an approach started (frontier stays disabled).
        Entries near locked targets or under rejection cooldown are dropped.
        """
        if not bool(self.get_parameter("pending_target_memory_enabled").value):
            return False
        while self._pending_targets:
            # Prefer confidence, then distance.  A weak nearby false positive
            # must not beat a well-observed remembered target.
            if self._robot_xy is None:
                best_index = max(
                    range(len(self._pending_targets)),
                    key=lambda i: self._pending_targets[i][2],
                )
            else:
                best_index = max(
                    range(len(self._pending_targets)),
                    key=lambda i: (
                        self._pending_targets[i][2],
                        -math.hypot(
                            self._pending_targets[i][0] - self._robot_xy[0],
                            self._pending_targets[i][1] - self._robot_xy[1],
                        ),
                    ),
                )
            x, y, confidence = self._pending_targets.pop(best_index)
            if target_is_locked(
                x, y, self._locked_targets, self._ignored_target_radius_m
            ):
                continue
            now = self.get_clock().now().nanoseconds / 1e9
            if candidate_is_rejected(
                x,
                y,
                self._rejected_candidates,
                self._candidate_reject_radius_m,
                now,
                self._candidate_retry_cooldown_s,
            ):
                continue
            # Force the lock: we navigate to a remembered coordinate, so
            # near-field camera loss must not cancel the geometric arrival.
            lock_conf = max(confidence, self._locked_approach_min_confidence)
            self._publish_event(
                "mission_pending_target_goto:x=%.2f,y=%.2f,reason=%s"
                % (x, y, reason)
            )
            self.get_logger().info(
                "Resuming at remembered target (%.2f, %.2f) instead of blind"
                " frontier search." % (x, y)
            )
            self._start_visual_approach(x, y, lock_conf, "pending_memory")
            if self._candidate_xy == (x, y):
                return True
        return False

    def _advance_scan(self) -> None:
        """Go to the next scan point, or resume Frontier without returning."""
        if self._scan_attempts >= self._scan_attempts_max:
            self._publish_event("mission_scan_attempts_exhausted")
            self.get_logger().warn(
                "Scan sequence exhausted after %d spins with %d/%d confirmed "
                "targets; resuming Frontier instead of returning."
                % (
                    self._scan_attempts,
                    self._reached_target_count,
                    self._target_goal_count,
                )
            )
            self._resume_search("scan_attempts_exhausted")
            return
        if self._scan_rotations_at_location < self._scan_rotations_before_relocate:
            self._state = MissionState.SCANNING
            self._publish_event(
                "mission_scan_started:attempt=%d"
                % (self._scan_attempts + 1)
            )
            self._start_segmented_scan()
            return
        self._state = MissionState.NAVIGATING_TO_SCAN_POINT
        self._publish_event(
            "mission_scan_point_navigation_started:attempt=%d"
            % (self._scan_attempts + 1)
        )
        self._send_scan_point_goal()

    def _on_confirm_arrived(self) -> None:
        if self._state != MissionState.CONFIRMING_TARGET:
            return
        self._confirm_until = (
            self.get_clock().now().nanoseconds / 1e9
            + self._post_nav_confirm_timeout_s
        )
        self._publish_event(
            "mission_confirm_waiting:%.1fs" % self._post_nav_confirm_timeout_s
        )
        self.get_logger().warn(
            "Arrived near scan candidate; waiting %.1fs for confirmation."
            % self._post_nav_confirm_timeout_s
        )

    def _on_confirm_succeeded(self, x: float, y: float) -> None:
        if self._state != MissionState.CONFIRMING_TARGET:
            return
        if target_is_locked(x, y, self._locked_targets, self._ignored_target_radius_m):
            self._on_confirm_failed("already_locked")
            return
        self._accept_confirmed_target(x, y, source="scan")

    def _on_confirm_failed(self, reason: str) -> None:
        if self._state != MissionState.CONFIRMING_TARGET:
            return
        self._publish_event("mission_confirm_failed:%s" % reason)
        self.get_logger().warn(
            "Scan candidate did not confirm (%s); trying escape." % reason
        )
        self._confirm_until = 0.0
        if self._start_recovery("confirm"):
            return
        self._target_pose = None
        self._advance_scan()

    # ------------------------------------------------------------------
    # Supervisor escape recovery (lateral first, then reverse backup)
    # ------------------------------------------------------------------

    def _start_recovery(self, pending: str) -> bool:
        if not bool(self.get_parameter("escape_enabled").value):
            return False
        if self._recovery_active:
            return False
        now_ns = self.get_clock().now().nanoseconds
        cooldown_ns = int(
            float(self.get_parameter("escape_cooldown_s").value) * 1e9
        )
        if now_ns - self._recovery_last_end_ns < cooldown_ns:
            return False
        self._recovery_pending = pending
        self._recovery_active = True
        scan_timeout_ns = int(
            float(self.get_parameter("escape_scan_timeout_s").value) * 1e9
        )
        scan = self._latest_scan
        if scan is not None and now_ns - self._latest_scan_ns <= scan_timeout_ns:
            left, right = side_sector_clearances(
                scan.ranges, float(scan.angle_min), float(scan.angle_increment)
            )
            direction = choose_lateral_escape_direction(
                left,
                right,
                self._escape_lateral_trigger_m,
                self._escape_lateral_destination_clearance_m,
            )
            if direction != 0.0:
                self._recovery_phase = "lateral"
                self._recovery_direction = direction
                self._recovery_start_xy = self._robot_xy
                self._recovery_until_ns = now_ns + int(
                    self._escape_lateral_duration_s * 1e9
                )
                self._publish_event(
                    "escape_lateral_started:%s" % ("left" if direction > 0 else "right")
                )
                self.get_logger().warn(
                    "Escape: lateral %s for %.1fs."
                    % (
                        "left" if direction > 0 else "right",
                        self._escape_lateral_duration_s,
                    )
                )
                return True
        self._start_backup_phase()
        return True

    def _start_backup_phase(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self._recovery_phase = "backup"
        self._recovery_direction = 0.0
        self._recovery_start_xy = self._robot_xy
        self._recovery_start_ns = now_ns
        self._recovery_until_ns = now_ns + int(
            self._escape_backup_timeout_s * 1e9
        )
        self._publish_event("escape_backup_started")
        self.get_logger().warn(
            "Escape: reverse backup %.1fm at %.2fm/s."
            % (self._escape_backup_distance_m, self._escape_backup_speed_mps)
        )

    def _escape_step(self) -> None:
        if not self._recovery_active:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._recovery_phase == "lateral":
            if now_ns >= self._recovery_until_ns:
                self._finish_lateral()
                return
            command = Twist()
            command.linear.y = self._recovery_direction * self._escape_lateral_speed_mps
            self._recovery_pub.publish(command)
            return
        if self._recovery_phase == "backup":
            moved = self._recovery_moved_m()
            if (
                moved >= self._escape_backup_distance_m * 0.9
                or now_ns - self._recovery_start_ns
                > int(self._escape_backup_timeout_s * 1e9)
            ):
                self._finish_recovery(moved >= self._escape_backup_distance_m * 0.9)
                return
            command = Twist()
            command.linear.x = -self._escape_backup_speed_mps
            self._recovery_pub.publish(command)
            return

    def _finish_lateral(self) -> None:
        self._recovery_pub.publish(Twist())
        moved = self._recovery_moved_m()
        if moved >= self._escape_min_movement_m:
            self._publish_event("escape_lateral_moved:%.2fm" % moved)
            self.get_logger().warn(
                "Lateral escape moved %.2fm; retrying pending action." % moved
            )
            self._finish_recovery(True)
            return
        self.get_logger().warn(
            "Lateral escape did not move (%.2fm); falling back to reverse backup."
            % moved
        )
        self._start_backup_phase()

    def _recovery_moved_m(self) -> float:
        if self._recovery_start_xy is None or self._robot_xy is None:
            return 0.0
        return math.hypot(
            self._robot_xy[0] - self._recovery_start_xy[0],
            self._robot_xy[1] - self._recovery_start_xy[1],
        )

    def _finish_recovery(self, success: bool) -> None:
        self._recovery_pub.publish(Twist())
        self._recovery_active = False
        self._recovery_phase = ""
        self._recovery_start_xy = None
        self._recovery_last_end_ns = self.get_clock().now().nanoseconds
        pending = self._recovery_pending
        self._recovery_pending = None
        if success:
            self._publish_event("escape_finished")
            self._retry_pending_action(pending)
            return
        self._publish_event("escape_failed")
        if pending == "confirm":
            self._target_pose = None
            self._advance_scan()
        elif pending == "scan":
            self._relocate_after_scan_stall("escape_failed")
        elif pending == "return":
            self._publish_event("mission_return_failed:escape_failed")
            self.get_logger().error(
                "Return-to-base failed after escape; supervisor stops."
            )
        else:
            self.get_logger().error("Recovery finished with no pending action.")

    def _stop_recovery(self, reason: str) -> None:
        """Abort any in-progress recovery without retrying the pending action."""
        self._recovery_pub.publish(Twist())
        self._recovery_active = False
        self._recovery_phase = ""
        self._recovery_direction = 0.0
        self._recovery_start_xy = None
        self._recovery_pending = None
        self._recovery_last_end_ns = self.get_clock().now().nanoseconds
        self.get_logger().info("Recovery stopped (%s)." % reason)

    def _retry_pending_action(self, pending: Optional[str]) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if pending == "confirm":
            if self._target_pose is None:
                self._advance_scan()
                return
            self._state = MissionState.CONFIRMING_TARGET
            self._confirm_until = 0.0
            self._next_goal_allowed_time = now
            self._send_approach_goal("confirm")
        elif pending == "return":
            self._state = MissionState.RETURNING
            self._send_return_goal()
        elif pending == "scan":
            self._relocate_after_scan_stall("escape_finished")
        else:
            self.get_logger().warn("Unknown recovery pending action: %r" % pending)

    def _relocate_after_scan_stall(self, reason: str) -> None:
        """Leave a rotation-blocked location and continue visual search."""
        self._scan_failed_segments_streak = 0
        self._scan_pause_until = 0.0
        self._scan_attempts += 1
        # Force _advance_scan() down its relocation branch instead of asking
        # for another full turn at the same collision-constrained pose.
        self._scan_rotations_at_location = self._scan_rotations_before_relocate
        self._publish_event("mission_scan_relocating_after_stall:%s" % reason)
        self.get_logger().warn(
            "Scan rotation is blocked (%s); relocating to a nearby unvisited "
            "free point instead of aborting the mission." % reason
        )
        self._advance_scan()

    # ------------------------------------------------------------------
    # Spin (in-place scan rotation)
    # ------------------------------------------------------------------

    def _start_segmented_scan(self) -> None:
        """Start one full scan made of short turn-and-observe segments."""
        self._scan_best_candidate = None
        self._scan_segment_index = 0
        self._scan_pause_until = 0.0
        self._start_spin()

    def _start_spin(self) -> None:
        """Command the next 60-degree scan segment, not a full 360-degree spin."""
        if not self._spin_client.wait_for_server(timeout_sec=1.0):
            self._publish_event("mission_scan_failed:spin_unavailable")
            self.get_logger().error(
                "Spin unavailable; returning to Frontier search, not base."
            )
            self._resume_search("spin_unavailable")
            return
        goal = Spin.Goal()
        goal.target_yaw = self._scan_segment_yaw_rad
        self._spin_active = False
        self._spin_handle = None
        self._spin_start_time = self.get_clock().now().nanoseconds / 1e9
        self._spin_generation += 1
        generation = self._spin_generation
        self._publish_event(
            "mission_scan_segment_started:%d/%d,yaw_deg=%.1f"
            % (
                self._scan_segment_index + 1,
                self._scan_segments_per_rotation,
                math.degrees(self._scan_segment_yaw_rad),
            )
        )
        future = self._spin_client.send_goal_async(goal)
        future.add_done_callback(
            lambda done, gen=generation: self._on_spin_response(done, gen)
        )

    def _on_spin_response(self, future, generation: int) -> None:
        if generation != self._spin_generation:
            return
        try:
            spin_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Sending Spin goal failed: %s" % exc)
            self._on_spin_result(GoalStatus.STATUS_ABORTED)
            return
        if not spin_handle.accepted:
            self.get_logger().warn("Nav2 rejected the Spin goal.")
            self._on_spin_result(GoalStatus.STATUS_ABORTED)
            return
        self._spin_handle = spin_handle
        self._spin_active = True
        result_future = spin_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, gen=generation: self._on_spin_done(done, gen)
        )

    def _on_spin_done(self, future, generation: int) -> None:
        if generation != self._spin_generation:
            return
        self._spin_active = False
        self._spin_handle = None
        try:
            status = future.result().status
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Reading Spin result failed: %s" % exc)
            status = GoalStatus.STATUS_ABORTED
        self._on_spin_result(status)

    def _on_spin_result(self, status: int) -> None:
        if self._state != MissionState.SCANNING:
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._scan_failed_segments_streak = 0
            self.get_logger().info(
                "Scan segment %d/%d completed."
                % (self._scan_segment_index + 1, self._scan_segments_per_rotation)
            )
        else:
            self._scan_failed_segments_streak += 1
            self.get_logger().warn(
                "Scan segment %d/%d ended with status %d; continuing to observation pause."
                % (
                    self._scan_segment_index + 1,
                    self._scan_segments_per_rotation,
                    int(status),
                )
            )
            if self._scan_failed_segments_streak >= self._scan_failed_segments_abort_max:
                # The gate watchdog handles disabled motion gates separately.
                # Repeated Collision Ahead here means this pose is unsuitable
                # for an in-place turn: escape once and relocate instead of
                # ending the whole three-target mission.
                self.get_logger().warn(
                    "%d consecutive scan segments failed; escaping and "
                    "relocating the visual scan."
                    % self._scan_failed_segments_streak
                )
                self._publish_event(
                    "mission_scan_motion_stalled_recovery:streak=%d"
                    % self._scan_failed_segments_streak
                )
                if not self._start_recovery("scan"):
                    self._relocate_after_scan_stall("recovery_unavailable")
                return
        self._scan_segment_index += 1
        if self._scan_segment_index < self._scan_segments_per_rotation:
            now = self.get_clock().now().nanoseconds / 1e9
            self._scan_pause_until = now + self._scan_segment_pause_s
            self._publish_event(
                "mission_scan_pause_started:segment=%d/%d,duration=%.1fs"
                % (
                    self._scan_segment_index,
                    self._scan_segments_per_rotation,
                    self._scan_segment_pause_s,
                )
            )
            return

        self._scan_pause_until = 0.0
        self._publish_event("mission_scan_finished")
        self._scan_attempts += 1
        self._scan_rotations_at_location += 1
        best = self._scan_best_candidate
        self._scan_best_candidate = None
        if best is not None:
            self._publish_event(
                "mission_scan_candidate_selected:conf=%.2f,x=%.2f,y=%.2f"
                % (best[2], best[0], best[1])
            )
            self.get_logger().warn(
                "Spin found best candidate conf=%.2f at (%.2f, %.2f); "
                "approaching for close verification." % (best[2], best[0], best[1])
            )
            # A spin candidate follows the same candidate -> approach ->
            # close verification policy as one seen during Frontier travel.
            self._state = MissionState.SEARCHING
            self._start_visual_approach(best[0], best[1], best[2], "scan")
            return
        self._advance_scan()

    # ------------------------------------------------------------------
    # Control loop helpers
    # ------------------------------------------------------------------

    def _control_step(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        # Publish the latched status snapshot even while IDLE so an operator
        # can always see the mission node is alive and in which state.
        if now - self._last_status_pub_s >= 1.0:
            self._last_status_pub_s = now
            self._publish_node_status(now)
        self._record_base_pose_once()
        self._apply_square_fence_if_needed()
        if self._state in (MissionState.IDLE, MissionState.DONE):
            return

        self._check_slam_loss_abort(now)
        if self._state in (MissionState.IDLE, MissionState.DONE):
            # The SLAM-loss watchdog already reset the mission above.
            return

        self._check_motion_gate_abort(now)
        if self._state in (MissionState.IDLE, MissionState.DONE):
            # The motion-gate watchdog already reset the mission above.
            return

        if should_start_scan_after_frontier_unavailable(
            self._state,
            self._frontier_unavailable_since,
            now,
            self._frontier_unavailable_scan_delay_s,
            self._mission_start_time,
            self._scan_fallback_min_sec,
        ):
            waited = now - self._frontier_unavailable_since
            reason = self._frontier_unavailable_reason or "frontier_unavailable"
            self._publish_event(
                "mission_scan_fallback_triggered:reason=%s,waited=%.1fs"
                % (reason, waited)
            )
            self.get_logger().warn(
                "No safe/reachable Frontier goal for %.1fs (%s); switching "
                "from SEARCHING to in-place visual scan." % (waited, reason)
            )
            self._begin_scan_sequence()
            return

        if (
            self._state == MissionState.VERIFYING_TARGET
            and now >= self._target_verify_until
        ):
            self._abandon_visual_candidate("verification_timeout")

        if (
            self._state == MissionState.CONFIRMING_TARGET
            and self._confirm_until > 0.0
            and now >= self._confirm_until
        ):
            self._on_confirm_failed("confirm_timeout")

        if (
            self._state == MissionState.APPROACHING
            and not self._goal_pending
            and not self._goal_active
            and now >= self._next_goal_allowed_time
        ):
            self._send_approach_goal()

        if (
            self._state == MissionState.CONFIRMING_TARGET
            and self._confirm_until == 0.0
            and not self._recovery_active
            and not self._goal_pending
            and not self._goal_active
            and now >= self._next_goal_allowed_time
        ):
            self._send_approach_goal("confirm")

        if (
            self._state == MissionState.WAITING_ARM_TARGET
            and self._arm_wait_deadline > 0.0
            and now >= self._arm_wait_deadline
        ):
            self._finish_arm_grasp(False, "fresh_arm_target_timeout")
            return

        if (
            self._state == MissionState.GRASPING
            and (self._arm_goal_pending or self._arm_goal_active)
            and now - self._arm_goal_start_time >= self._arm_action_timeout_s
        ):
            self._cancel_arm_goal("timeout")
            # A timeout does not prove that the arm has finished retracting.
            # Never hand velocity ownership back to navigation while the arm
            # may still be moving; remain stopped and require an operator to
            # inspect /arm/status before starting a new mission.
            self._disable_frontier()
            self._call_trigger(self._sport_stop_client, "sport StopMove after arm timeout")
            self._publish_event("mission_arm_grasp_failed:action_timeout")
            self._reset_to_idle("mission_arm_manual_reset_required")
            return

        if (
            self._state == MissionState.SCANNING
            and not self._spin_active
            and self._scan_pause_until > 0.0
            and now >= self._scan_pause_until
        ):
            self._scan_pause_until = 0.0
            self._start_spin()

        if self._state == MissionState.TARGET_HOLDING and now >= self._hold_until:
            if self._target_posture_phase:
                self._step_target_posture(now)
            # Read the declared parameter here rather than only the startup
            # cache, so an operator can safely enable/disable the posture
            # behavior before a later confirmed target without restarting.
            elif bool(self.get_parameter("target_posture_enabled").value):
                self._start_target_posture(now)
            else:
                self._finish_target_hold()

        if self._state == MissionState.HOLDING and now >= self._hold_until:
            self._publish_event("mission_hold_finished")
            self.get_logger().info("Hold finished; returning to base.")
            self._begin_return()

        if (
            self._state == MissionState.RETURNING
            and getattr(self, "_return_alignment_active", False)
            and now - self._return_alignment_start_time
            >= self._return_prealign_timeout_s
        ):
            self._cancel_return_alignment("timeout")
            self._publish_event("mission_return_prealign_timeout_fallback")
            self.get_logger().warn(
                "Return pre-alignment timed out; continuing with normal Nav2."
            )
            self._send_return_goal()

        self._check_timeouts(now)

    def _mark_frontier_unavailable(self, reason: str) -> None:
        """Remember the first no-candidate event in one continuous interval."""
        if self._frontier_unavailable_since <= 0.0:
            self._frontier_unavailable_since = (
                self.get_clock().now().nanoseconds / 1e9
            )
            self._frontier_unavailable_reason = reason
            self._publish_event("mission_frontier_unavailable:%s" % reason)
            self.get_logger().warn(
                "Frontier unavailable (%s); starting %.1fs fallback timer."
                % (reason, self._frontier_unavailable_scan_delay_s)
            )

    def _clear_frontier_unavailable(self) -> None:
        self._frontier_unavailable_since = 0.0
        self._frontier_unavailable_reason = ""

    def _record_base_pose_once(self) -> None:
        if self._base_pose is not None or self._robot_xy is None:
            return
        if self._state in (MissionState.IDLE, MissionState.DONE):
            return
        self._base_pose = self._pose_message(
            self._robot_xy[0],
            self._robot_xy[1],
            self._robot_yaw,
        )
        self.get_logger().info(
            "Base pose recorded: (%.2f, %.2f)" % self._robot_xy
        )

    def _apply_square_fence_if_needed(self) -> None:
        """Push the exploration fence to the frontier node once per mission."""
        if self._fence_applied or not self._exploration_square_enabled:
            return
        if self._base_pose is None:
            return
        if not self._frontier_params_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(
                "frontier_explorer_node set_parameters service unavailable; "
                "retrying exploration fence.",
                throttle_duration_sec=5.0,
            )
            return
        center_x = float(self._base_pose.pose.position.x)
        center_y = float(self._base_pose.pose.position.y)
        usable_half_size = (
            self._exploration_square_half_size_m
            - self._exploration_boundary_margin_m
        )
        min_x, max_x, min_y, max_y = square_fence_bounds(
            center_x,
            center_y,
            usable_half_size,
        )
        request = SetParameters.Request()
        request.parameters = [
            Parameter("exploration_min_x", value=min_x).to_parameter_msg(),
            Parameter("exploration_max_x", value=max_x).to_parameter_msg(),
            Parameter("exploration_min_y", value=min_y).to_parameter_msg(),
            Parameter("exploration_max_y", value=max_y).to_parameter_msg(),
        ]
        future = self._frontier_params_client.call_async(request)
        future.add_done_callback(self._on_fence_params_set)

    def _on_fence_params_set(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Setting exploration fence failed: %s" % exc)
            return
        if all(result.successful for result in response.results):
            self._fence_applied = True
            self.get_logger().warn(
                "Exploration fence applied (outer half-size %.2fm, usable "
                "half-size %.2fm, boundary margin %.2fm; centered at base %.2f, %.2f)."
                % (
                    self._exploration_square_half_size_m,
                    self._exploration_square_half_size_m
                    - self._exploration_boundary_margin_m,
                    self._exploration_boundary_margin_m,
                    float(self._base_pose.pose.position.x)
                    if self._base_pose is not None
                    else 0.0,
                    float(self._base_pose.pose.position.y)
                    if self._base_pose is not None
                    else 0.0,
                )
            )
        else:
            self.get_logger().error(
                "Exploration fence rejected by frontier node: %s"
                % [result.reason for result in response.results]
            )

    def _check_slam_loss_abort(self, now: float) -> None:
        """Abort the mission when the official SLAM stream is unusable.

        Observed failure mode: the pose-jump guard latched RECOVERING (or the
        dock-side unitree_slam died outright), both motion gates closed, and
        this supervisor kept issuing goals and waiting out 60s scan-segment
        timeouts because nothing told it the map frame was gone.  Two signals
        cover the failure: the guard's latched state, and a stale
        /slam/current_pose.  Debounce for slam_loss_confirm_s so the brief
        2-3s stream hiccups seen in real runs do not abort a healthy mission.
        """
        if not self._slam_loss_abort_enabled:
            return
        if self._target_posture_phase:
            # The crouch/stand sequence drives only Sport services.  Let it
            # finish standing up and resuming the guards; the abort then fires
            # on a later tick instead of leaving the robot crouched.
            return

        reason = ""
        if self._slam_guard_state in ("RECOVERING", "READY_FOR_MANUAL_REARM"):
            reason = "guard_latched:%s" % self._slam_guard_state
        elif (
            self._current_pose_at_s > 0.0
            and now - self._current_pose_at_s > self._slam_pose_max_age_s
        ):
            reason = "pose_stale:age=%.1fs" % (now - self._current_pose_at_s)
        elif (
            self._current_pose_at_s <= 0.0
            and self._robot_xy is None
            and self._mission_start_time > 0.0
            and now - self._mission_start_time
            > self._slam_pose_max_age_s + self._slam_loss_confirm_s
        ):
            # _mission_start_time > 0 guards the manual-grasp flow, which never
            # sets it but also cannot start without a received pose.
            reason = "pose_missing"

        if not reason:
            self._slam_loss_since = 0.0
            return

        if self._slam_loss_since <= 0.0:
            self._slam_loss_since = now
            self.get_logger().warn(
                "SLAM loss suspected (%s); confirming for %.1fs before abort."
                % (reason, self._slam_loss_confirm_s)
            )
            return
        if now - self._slam_loss_since < self._slam_loss_confirm_s:
            return

        self._slam_loss_since = 0.0
        self.get_logger().error(
            "Aborting mission: official SLAM is unusable (%s). Recover with "
            "./scripts/recover_from_slam_loss.sh before starting a new mission."
            % reason
        )
        self._publish_event("mission_slam_lost_abort:%s" % reason)
        self._disable_frontier()
        self._cancel_active_goal("mission_slam_lost")
        self._call_trigger(self._sport_stop_client, "sport StopMove after SLAM loss")
        self._reset_to_idle("mission_slam_lost:%s" % reason)

    def _check_motion_gate_abort(self, now: float) -> None:
        """Abort the mission when the velocity gate is (and stays) closed.

        Observed failure mode: cmd_vel_safety was disarmed (operator action or
        the odom-jump auto-disarm) while the mission kept issuing goals; every
        Nav2 goal and every scan segment then sat motionless until its own
        timeout, worst case 4 x spin_timeout_s before the failed-segment fuse
        tripped -- the robot idled ~10 minutes in the field.  The gate's
        latched status topic makes the condition directly observable, so abort
        within motion_gate_closed_confirm_s instead.  Unlike SLAM loss the map
        frame may be perfectly healthy; this fires when motion output alone is
        disarmed.  Postponed during the crouch/stand sequence exactly like the
        SLAM-loss abort (never strand the robot crouched).
        """
        if not self._motion_gate_abort_enabled:
            return
        if self._target_posture_phase:
            # The posture sequence drives only Sport services; let it finish
            # standing up first, the abort fires on a later tick.
            return

        if self._motion_gate_enabled is not False:
            self._motion_gate_closed_since = 0.0
            return

        if self._motion_gate_closed_since <= 0.0:
            self._motion_gate_closed_since = now
            self.get_logger().warn(
                "Velocity gate closed (%s); confirming for %.1fs before abort."
                % (
                    self._motion_gate_reason or "reason unknown",
                    self._motion_gate_closed_confirm_s,
                )
            )
            return
        if now - self._motion_gate_closed_since < self._motion_gate_closed_confirm_s:
            return

        reason = self._motion_gate_reason or "reason_unknown"
        self._motion_gate_closed_since = 0.0
        self.get_logger().error(
            "Aborting mission: velocity gate is closed (%s). Re-enable the "
            "gates (蹲起流程复现.md section 4) before starting a new mission."
            % reason
        )
        self._publish_event("mission_motion_gate_closed_abort:%s" % reason)
        self._disable_frontier()
        self._cancel_active_goal("motion_gate_closed")
        self._call_trigger(
            self._sport_stop_client, "sport StopMove after motion gate closed"
        )
        self._reset_to_idle("mission_motion_gate_closed:%s" % reason)

    def _check_timeouts(self, now: float) -> None:
        if self._goal_pending or self._goal_active:
            label = self._goal_label
            if (
                label == "approach"
                and getattr(self, "_approach_target_locked", False)
                and self._target_pose is not None
                and self._robot_xy is not None
                and self._approach_topup_used < self._approach_topup_max
            ):
                distance = math.hypot(
                    float(self._target_pose.pose.position.x) - self._robot_xy[0],
                    float(self._target_pose.pose.position.y) - self._robot_xy[1],
                )
                best = self._approach_best_distance_m
                if best is None:
                    self._approach_best_distance_m = distance
                    self._approach_last_progress_time = now
                elif distance <= best - self._approach_progress_epsilon_m:
                    self._approach_best_distance_m = distance
                    self._approach_last_progress_time = now
                elif (
                    distance <= self._approach_near_stall_distance_m
                    and self._approach_last_progress_time > 0.0
                    and now - self._approach_last_progress_time
                    >= self._approach_near_stall_timeout_s
                ):
                    stalled_s = now - self._approach_last_progress_time
                    self._cancel_nav_goal("near_field_stall")
                    self._publish_event(
                        "mission_approach_near_field_stall:dist=%.3f,no_progress=%.1fs"
                        % (distance, stalled_s)
                    )
                    self._on_goal_failed("near_field_stall")
                    return
            if label == "approach":
                timeout = self._approach_timeout_s
            elif label == "confirm":
                timeout = self._approach_timeout_s
            elif label == "return":
                timeout = self._return_timeout_s
            elif label == "scan_point":
                timeout = self._scan_point_timeout_s
            else:
                timeout = self._approach_timeout_s
            if now - self._goal_start_time > timeout:
                self._cancel_nav_goal("timeout")
                self._publish_event("navigation_timeout:%s" % label)
                self._on_goal_failed("timeout")
        if self._spin_active and now - self._spin_start_time > self._spin_timeout_s:
            self._cancel_spin("timeout")
            self._on_spin_result(GoalStatus.STATUS_ABORTED)

    # ------------------------------------------------------------------
    # Frontier control
    # ------------------------------------------------------------------

    def _call_trigger(self, client, label: str) -> bool:
        if not client.wait_for_service(timeout_sec=0.2):
            self.get_logger().error("%s service unavailable" % label)
            return False
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda done: self._log_service_result(label, done))
        return True

    def _call_set_bool(self, client, enabled: bool, label: str) -> bool:
        if not client.wait_for_service(timeout_sec=0.2):
            self.get_logger().error("%s service unavailable" % label)
            return False
        request = SetBool.Request()
        request.data = bool(enabled)
        future = client.call_async(request)
        future.add_done_callback(lambda done: self._log_service_result(label, done))
        return True

    def _log_service_result(self, label: str, future) -> None:
        try:
            response = future.result()
            if not bool(getattr(response, "success", True)):
                self.get_logger().error(
                    "%s rejected: %s" % (label, getattr(response, "message", ""))
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("%s call failed: %s" % (label, exc))

    def _clear_target_posture_ack(self) -> None:
        self._target_posture_expected_api = 0
        self._target_posture_ack = None
        self._target_posture_ack_info = ""
        self._target_posture_ack_deadline = 0.0

    def _wait_for_target_posture_ack(
        self, *, api_id: int, phase: str, now: float
    ) -> None:
        self._target_posture_expected_api = int(api_id)
        self._target_posture_ack = None
        self._target_posture_ack_info = ""
        self._target_posture_ack_deadline = (
            now + self._target_posture_ack_timeout_s
        )
        self._target_posture_phase = phase
        self._hold_until = now + 0.1

    def _fail_target_posture(
        self, reason: str, *, allow_pose_retry: bool = True
    ) -> None:
        info = self._target_posture_ack_info
        detail = reason if not info else "%s:%s" % (reason, info)
        self.get_logger().error("Target posture failed: %s" % detail)
        self._publish_event("mission_target_posture_failed:%s" % detail)

        pending = getattr(self, "_pending_target_completion", None)
        retryable = reason in {
            "pre_crouch_stand_rejected",
            "pre_crouch_stand_ack_timeout",
            "stand_down_rejected",
            "stand_down_ack_timeout",
        }
        retry_count = int(getattr(self, "_target_posture_retry_count", 0))
        retry_max = int(getattr(self, "_target_posture_retry_max", 1))
        if (
            allow_pose_retry
            and pending is not None
            and retryable
            and retry_count < retry_max
        ):
            # The first crouch rejection often means the robot is in a
            # terrain-dependent locomotion mode. Normalize to BalanceStand,
            # wait for it to settle, then run StandUp -> StandDown once more.
            self._target_posture_retry_count = retry_count + 1
            self._clear_target_posture_ack()
            now = self.get_clock().now().nanoseconds / 1e9
            self._wait_for_target_posture_ack(
                api_id=SPORT_API_BALANCE_STAND,
                phase="RETRY_BALANCE_WAIT_ACK",
                now=now,
            )
            if not self._call_trigger(
                self._sport_balance_stand_client,
                "sport BalanceStand before posture retry",
            ):
                self._fail_target_posture(
                    "retry_balance_service_unavailable",
                    allow_pose_retry=False,
                )
                return
            self._publish_event(
                "mission_target_posture_retry_requested:%d/%d,reason=%s"
                % (self._target_posture_retry_count, retry_max, detail)
            )
            self.get_logger().warn(
                "Changing to BalanceStand before the one allowed target "
                "posture retry (%d/%d)."
                % (self._target_posture_retry_count, retry_max)
            )
            return

        self._call_set_bool(
            self._slam_jump_guard_monitoring_client,
            True,
            "resume SLAM jump monitoring",
        )
        self._call_set_bool(
            self._velocity_odom_jump_monitoring_client,
            True,
            "resume velocity odom jump monitoring",
        )
        self._call_set_bool(
            self._map_accumulation_client, True, "resume occupancy accumulation"
        )
        self._target_posture_phase = ""
        self._clear_target_posture_ack()
        self._pending_target_completion = None
        if pending is not None:
            x, y, _source, _confidence = pending
            # Keep this coordinate in _locked_targets: it is a permanent
            # per-mission ignore record after the bounded retry is exhausted.
            # Also remove any duplicate scene-memory entries so the pending
            # queue cannot route back to it.
            self._pending_targets = [
                item
                for item in self._pending_targets
                if math.hypot(item[0] - x, item[1] - y)
                > self._ignored_target_radius_m
            ]
            self._unserviceable_targets.append((x, y, detail))
            self._publish_event(
                "mission_target_unserviceable:x=%.2f,y=%.2f,reason=%s"
                % (x, y, detail)
            )
        self._publish_event(
            "mission_target_interaction_failed:not_counted:%s" % detail
        )
        self._target_pose = None
        self._candidate_xy = None
        if getattr(self, "_stop_after_target_interaction", False):
            self._reset_to_idle(
                "mission_single_target_stopped:posture_failed"
            )
            return
        self._resume_search("target_unserviceable")

    def _start_target_posture(self, now: float) -> None:
        """Stop, freeze map accumulation, then command one controlled crouch."""
        clients_ready = all(
            client.wait_for_service(timeout_sec=0.2)
            for client in (
                self._sport_stop_client,
                self._sport_stand_down_client,
                self._sport_stand_up_client,
                self._sport_balance_stand_client,
                self._map_accumulation_client,
                self._slam_jump_guard_monitoring_client,
                self._velocity_odom_jump_monitoring_client,
            )
        )
        if not clients_ready:
            self.get_logger().error(
                "Target posture skipped: required sport/map service is unavailable."
            )
            self._publish_event("mission_target_posture_skipped:service_unavailable")
            self._fail_target_posture("service_unavailable")
            return

        self._disable_frontier()
        self._call_trigger(self._sport_stop_client, "sport StopMove")
        if not self._call_set_bool(
            self._slam_jump_guard_monitoring_client,
            False,
            "pause SLAM jump monitoring for controlled posture",
        ):
            self._fail_target_posture("slam_jump_guard_pause_failed")
            return
        if not self._call_set_bool(
            self._velocity_odom_jump_monitoring_client,
            False,
            "pause velocity odom jump monitoring for controlled posture",
        ):
            self._fail_target_posture("velocity_jump_guard_pause_failed")
            return
        self._call_set_bool(
            self._map_accumulation_client, False, "pause occupancy accumulation"
        )
        self._target_posture_phase = "POSTURE_GUARDS_SETTLE"
        self._hold_until = now + self._target_posture_guard_settle_s
        self._publish_event("mission_target_posture_guards_paused")

    def _step_target_posture(self, now: float) -> None:
        if self._target_posture_phase == "RETRY_BALANCE_WAIT_ACK":
            if self._target_posture_ack is True:
                self._clear_target_posture_ack()
                self._target_posture_phase = "RETRY_BALANCE_SETTLE"
                self._hold_until = now + getattr(
                    self, "_target_posture_retry_settle_s", 2.0
                )
                self._publish_event("mission_target_posture_retry_balance_started")
                return
            if self._target_posture_ack is False:
                self._fail_target_posture(
                    "retry_balance_rejected", allow_pose_retry=False
                )
                return
            if now >= self._target_posture_ack_deadline:
                self._fail_target_posture(
                    "retry_balance_ack_timeout", allow_pose_retry=False
                )
                return
            self._hold_until = min(
                self._target_posture_ack_deadline, now + 0.1
            )
            return
        if self._target_posture_phase == "RETRY_BALANCE_SETTLE":
            self._target_posture_phase = "POSTURE_GUARDS_SETTLE"
            self._hold_until = now + self._target_posture_guard_settle_s
            self._publish_event("mission_target_posture_retry_started")
            return
        if self._target_posture_phase == "POSTURE_GUARDS_SETTLE":
            # B2W locomotion normally uses BalanceStand (mode 1), while
            # StandDown must be requested from regular StandUp (mode 6).
            self._wait_for_target_posture_ack(
                api_id=SPORT_API_STAND_UP,
                phase="PRE_CROUCH_STAND_WAIT_ACK",
                now=now,
            )
            if not self._call_trigger(
                self._sport_stand_up_client, "sport StandUp"
            ):
                self._fail_target_posture(
                    "pre_crouch_stand_service_unavailable"
                )
                return
            self._publish_event("mission_target_pre_crouch_stand_requested")
            return
        if self._target_posture_phase == "PRE_CROUCH_STAND_WAIT_ACK":
            if self._target_posture_ack is True:
                self._clear_target_posture_ack()
                self._target_posture_phase = "PRE_CROUCH_STAND_SETTLE"
                self._hold_until = now + self._target_pre_crouch_stand_settle_s
                self._publish_event("mission_target_pre_crouch_stand_started")
                return
            if self._target_posture_ack is False:
                self._fail_target_posture("pre_crouch_stand_rejected")
                return
            if now >= self._target_posture_ack_deadline:
                self._fail_target_posture("pre_crouch_stand_ack_timeout")
                return
            self._hold_until = min(
                self._target_posture_ack_deadline, now + 0.1
            )
            return
        if self._target_posture_phase == "PRE_CROUCH_STAND_SETTLE":
            self._wait_for_target_posture_ack(
                api_id=SPORT_API_STAND_DOWN,
                phase="CROUCH_WAIT_ACK",
                now=now,
            )
            if not self._call_trigger(
                self._sport_stand_down_client, "sport StandDown"
            ):
                self._fail_target_posture("stand_down_service_unavailable")
                return
            self._publish_event("mission_target_crouch_requested")
            return
        if self._target_posture_phase == "CROUCH_WAIT_ACK":
            if self._target_posture_ack is True:
                self._clear_target_posture_ack()
                self._target_posture_phase = "CROUCH_TRANSITION"
                self._hold_until = now + self._target_crouch_transition_s
                self._publish_event("mission_target_crouch_started")
                return
            if self._target_posture_ack is False:
                self._fail_target_posture("stand_down_rejected")
                return
            if now >= self._target_posture_ack_deadline:
                self._fail_target_posture("stand_down_ack_timeout")
                return
            self._hold_until = min(
                self._target_posture_ack_deadline, now + 0.1
            )
            return
        if self._target_posture_phase == "CROUCH_TRANSITION":
            self._target_posture_phase = "CROUCH_HOLD"
            self._hold_until = now + self._target_crouch_hold_s
            self._publish_event("mission_target_crouch_holding")
            return
        if self._target_posture_phase == "CROUCH_HOLD":
            self._wait_for_target_posture_ack(
                api_id=SPORT_API_STAND_UP,
                phase="STAND_WAIT_ACK",
                now=now,
            )
            if not self._call_trigger(self._sport_stand_up_client, "sport StandUp"):
                self._fail_target_posture("stand_up_service_unavailable")
                return
            self._publish_event("mission_target_stand_requested")
            return
        if self._target_posture_phase == "STAND_WAIT_ACK":
            if self._target_posture_ack is True:
                self._clear_target_posture_ack()
                self._publish_event("mission_target_stand_started")
                self._wait_for_target_posture_ack(
                    api_id=SPORT_API_BALANCE_STAND,
                    phase="BALANCE_WAIT_ACK",
                    now=now,
                )
                if not self._call_trigger(
                    self._sport_balance_stand_client,
                    "sport BalanceStand",
                ):
                    self._fail_target_posture(
                        "balance_stand_service_unavailable"
                    )
                    return
                self._publish_event("mission_target_balance_stand_requested")
                return
            if self._target_posture_ack is False:
                self._fail_target_posture("stand_up_rejected")
                return
            if now >= self._target_posture_ack_deadline:
                self._fail_target_posture("stand_up_ack_timeout")
                return
            self._hold_until = min(
                self._target_posture_ack_deadline, now + 0.1
            )
            return
        if self._target_posture_phase == "BALANCE_WAIT_ACK":
            if self._target_posture_ack is True:
                self._clear_target_posture_ack()
                self._target_posture_phase = "STAND_SETTLE"
                self._hold_until = now + self._target_stand_settle_s
                self._publish_event("mission_target_balance_stand_started")
                return
            if self._target_posture_ack is False:
                self._fail_target_posture("balance_stand_rejected")
                return
            if now >= self._target_posture_ack_deadline:
                self._fail_target_posture("balance_stand_ack_timeout")
                return
            self._hold_until = min(
                self._target_posture_ack_deadline, now + 0.1
            )
            return
        if self._target_posture_phase == "STAND_SETTLE":
            self._call_set_bool(
                self._slam_jump_guard_monitoring_client,
                True,
                "resume SLAM jump monitoring",
            )
            self._call_set_bool(
                self._velocity_odom_jump_monitoring_client,
                True,
                "resume velocity odom jump monitoring",
            )
            self._call_set_bool(
                self._map_accumulation_client, True, "resume occupancy accumulation"
            )
            self._publish_event("mission_target_posture_guards_resumed")
            self._target_posture_phase = ""
            self._clear_target_posture_ack()
            pending = getattr(self, "_pending_target_completion", None)
            if pending is not None:
                x, y, source, confidence = pending
                self._pending_target_completion = None
                self._reached_target_count += 1
                detail = (
                    "mission_target_confirmed:%d/%d,source=%s,x=%.2f,y=%.2f"
                    % (
                        self._reached_target_count,
                        self._target_goal_count,
                        source,
                        x,
                        y,
                    )
                )
                if confidence is not None:
                    detail += ",conf=%.2f" % confidence
                self._publish_event(detail)
            self._publish_event("mission_target_posture_finished")
            self._finish_target_hold()

    def _finish_target_hold(self) -> None:
        self._publish_event(
            "mission_target_hold_finished:%d/%d"
            % (self._reached_target_count, self._target_goal_count)
        )
        if self._stop_after_target_interaction:
            self._disable_frontier()
            self._target_pose = None
            self._candidate_xy = None
            self._state = MissionState.IDLE
            self._publish_event("mission_stopped_after_target_interaction")
            self.get_logger().info(
                "Target interaction completed; staged mission is now idle."
            )
            return
        if self._reached_target_count >= self._target_goal_count:
            self.get_logger().info("All targets confirmed; returning to base.")
            self._begin_return()
        else:
            self._frontier_goal_count = 0
            self._target_pose = None
            self._candidate_xy = None
            self._resume_search("target_confirmed")

    def _disable_frontier(self) -> None:
        self._set_frontier_enabled(False)

    def _enable_frontier(self) -> None:
        self._set_frontier_enabled(True)

    def _set_frontier_enabled(self, enabled: bool) -> None:
        if not self._frontier_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(
                "frontier_explorer_node enable service unavailable; "
                "cannot %s exploration." % ("enable" if enabled else "disable")
            )
            return
        request = SetBool.Request()
        request.data = enabled
        future = self._frontier_client.call_async(request)
        future.add_done_callback(self._on_frontier_set)

    def _on_frontier_set(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("Setting frontier enable failed: %s" % exc)
            return
        if not response.success:
            self.get_logger().warn(
                "Frontier enable rejected: %s" % response.message
            )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def _cancel_active_goal(self, reason: str) -> None:
        self._cancel_nav_goal(reason)
        self._cancel_spin(reason)
        self._cancel_return_alignment(reason)
        self._cancel_arm_goal(reason)

    def _cancel_nav_goal(self, reason: str) -> None:
        handle = self._goal_handle
        self._goal_handle = None
        self._goal_pending = False
        self._goal_active = False
        self._goal_generation += 1
        if handle is not None:
            handle.cancel_goal_async()
            self.get_logger().info("Canceled active Nav2 goal: %s" % reason)

    def _cancel_spin(self, reason: str) -> None:
        handle = self._spin_handle
        self._spin_handle = None
        self._spin_active = False
        self._spin_generation += 1
        if handle is not None:
            handle.cancel_goal_async()
            self.get_logger().info("Canceled active Spin goal: %s" % reason)

    def _cancel_return_alignment(self, reason: str) -> None:
        handle = getattr(self, "_return_spin_handle", None)
        was_active = bool(getattr(self, "_return_alignment_active", False))
        self._return_spin_handle = None
        self._return_alignment_active = False
        self._return_alignment_start_time = 0.0
        self._return_spin_generation = (
            int(getattr(self, "_return_spin_generation", 0)) + 1
        )
        if handle is not None:
            handle.cancel_goal_async()
        if was_active:
            self.get_logger().info(
                "Canceled return pre-alignment Spin: %s" % reason
            )

    def _cancel_arm_goal(self, reason: str) -> None:
        handle = self._arm_goal_handle
        was_active = self._arm_goal_pending or self._arm_goal_active
        self._arm_goal_generation += 1
        self._clear_arm_action_state()
        if handle is not None:
            handle.cancel_goal_async()
        if was_active:
            self.get_logger().info("Canceled active arm grasp: %s" % reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pose_xy_in_map(self, pose: PoseStamped) -> Optional[Tuple[float, float]]:
        if pose.header.frame_id in ("", self._map_frame):
            return float(pose.pose.position.x), float(pose.pose.position.y)
        if self._robot_xy is None:
            return None
        cos_yaw = math.cos(self._robot_yaw)
        sin_yaw = math.sin(self._robot_yaw)
        x = float(pose.pose.position.x)
        y = float(pose.pose.position.y)
        return (
            self._robot_xy[0] + cos_yaw * x - sin_yaw * y,
            self._robot_xy[1] + sin_yaw * x + cos_yaw * y,
        )

    def _pose_message(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def _publish_event(self, event_name: str) -> None:
        self._last_event = event_name
        msg = String()
        msg.data = event_name
        self._event_pub.publish(msg)
        self.get_logger().info("Published task event: %s" % event_name)

    def _build_status_payload(self, now: float) -> dict:
        """One JSON snapshot of the mission state machine for operators."""
        unserviceable = list(getattr(self, "_unserviceable_targets", []))
        confirmed_locked = [
            (x, y)
            for x, y in self._locked_targets
            if not any(
                math.hypot(x - failed_x, y - failed_y) <= 1e-6
                for failed_x, failed_y, _reason in unserviceable
            )
        ]
        robot_xy = None
        if self._robot_xy is not None:
            robot_xy = [round(self._robot_xy[0], 3), round(self._robot_xy[1], 3)]
        base_xy = None
        if self._base_pose is not None:
            base_xy = [
                round(float(self._base_pose.pose.position.x), 3),
                round(float(self._base_pose.pose.position.y), 3),
            ]
        return {
            "state": self._state.value,
            "reached_targets": self._reached_target_count,
            "target_goal_count": self._target_goal_count,
            "candidate_xy": (
                None
                if getattr(self, "_candidate_xy", None) is None
                else [
                    round(self._candidate_xy[0], 3),
                    round(self._candidate_xy[1], 3),
                ]
            ),
            "recovery_active": bool(getattr(self, "_recovery_active", False)),
            "recovery_phase": str(getattr(self, "_recovery_phase", "")),
            "recovery_pending": getattr(self, "_recovery_pending", None),
            "manual_mission": self._manual_mission_active,
            "guard_state": self._slam_guard_state,
            "motion_gate_enabled": self._motion_gate_enabled,
            "pending_targets": len(self._pending_targets),
            "pending_target_positions": [
                [round(x, 3), round(y, 3), round(confidence, 3)]
                for x, y, confidence in self._pending_targets
            ],
            "confirmed_target_positions": [
                [round(x, 3), round(y, 3)]
                for x, y in confirmed_locked
            ],
            "unserviceable_targets": len(unserviceable),
            "unserviceable_target_records": [
                {"xy": [round(x, 3), round(y, 3)], "reason": reason}
                for x, y, reason in unserviceable
            ],
            "posture_retry_count": int(
                getattr(self, "_target_posture_retry_count", 0)
            ),
            "slam_loss_suspected_s": (
                0.0
                if self._slam_loss_since <= 0.0
                else round(now - self._slam_loss_since, 1)
            ),
            "active_goal": self._goal_label,
            "spin_active": self._spin_active,
            "posture_phase": self._target_posture_phase,
            "robot_xy": robot_xy,
            "base_xy": base_xy,
            "last_event": self._last_event,
        }

    def _publish_node_status(self, now: float) -> None:
        msg = String()
        msg.data = json.dumps(
            self._build_status_payload(now), separators=(",", ":")
        )
        self._status_pub.publish(msg)


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z)),
    )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
