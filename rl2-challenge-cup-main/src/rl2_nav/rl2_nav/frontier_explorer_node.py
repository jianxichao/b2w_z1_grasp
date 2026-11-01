"""Standalone frontier exploration node for Nav2 debugging and demos."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Deque, Optional, Tuple

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import SetBool
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rl2_nav.frontier_explorer import (
    FrontierCandidate,
    FrontierExplorer,
    choose_lateral_escape_direction,
    side_sector_clearances,
    world_to_cell,
)


class FrontierExplorerNode(Node):
    """Send Nav2 goals toward frontiers in `/map`."""

    def __init__(self) -> None:
        super().__init__("frontier_explorer_node")

        # Motion must be armed explicitly after map, TF, Nav2, safety filter,
        # and the real controller have all been checked.
        self.declare_parameter("enabled", False)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("navigate_action", "/navigate_to_pose")
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("require_global_costmap_reachability", True)
        # OccupancyGrid values 99/100 correspond to inscribed/lethal cells;
        # inflated costs up to 98 remain traversable for the global planner.
        self.declare_parameter("global_costmap_traversable_cost_max", 98)
        # Frontier points lie on known/unknown boundaries. Snap each one to a
        # connected traversable cell within the planner's 0.5m tolerance.
        self.declare_parameter("global_costmap_goal_tolerance_m", 0.50)
        # A merely traversable center cell is insufficient for a wheeled robot:
        # require a low-cost disk around every snapped goal so a frontier cannot
        # terminate against a wall or inside the inflation band.
        self.declare_parameter("global_costmap_goal_clearance_m", 0.32)
        self.declare_parameter("global_costmap_goal_clearance_cost_max", 65)
        # Optional axis-aligned geofence in map coordinates. Equal or reversed
        # limits disable that axis, preserving automatic wall-based exploration.
        self.declare_parameter("exploration_min_x", 0.0)
        self.declare_parameter("exploration_max_x", 0.0)
        self.declare_parameter("exploration_min_y", 0.0)
        self.declare_parameter("exploration_max_y", 0.0)
        self.declare_parameter("frontier_min_size", 8)
        self.declare_parameter("frontier_max_candidates", 10)
        self.declare_parameter("cluster_connect_distance", 2)
        self.declare_parameter("free_threshold", 65)
        self.declare_parameter("occupied_threshold", 90)
        self.declare_parameter("goal_safety_radius_m", 0.35)
        # 候选点到障碍的最小通道宽度（FrontierExplorer 的窄道过滤阈值）。
        # 默认 0.28 会让 ~0.6m 门洞的候选整批被静默丢弃；现场窄门多时
        # 可降到 ~0.20（footprint 半宽 0.22 + 少量裕度）。
        # 0.25 m quantizes to 5 whole 5 cm cells: one cell of margin over
        # the 0.22 m rectangular footprint half-width (0.22 would round down
        # to 4 cells = 0.20 m, below the half-width itself).
        self.declare_parameter("path_clearance_m", 0.25)
        self.declare_parameter("failed_goal_radius_m", 0.65)
        self.declare_parameter("failed_goal_memory", 20)
        self.declare_parameter("replan_interval_sec", 2.5)
        self.declare_parameter("prefer_nearest_until_first_success", False)
        # Require the best post-costmap candidate to persist before committing
        # the robot.  Two samples remove one-frame wall holes without adding a
        # long stationary planning phase.
        self.declare_parameter("candidate_stability_required_cycles", 2)
        self.declare_parameter("candidate_stability_radius_m", 0.60)
        # Zero means continuous exploration.  A positive value makes one
        # enable cycle self-disarm after that many goal attempts.
        self.declare_parameter("max_goals_per_enable", 0)
        # 看门狗：连续这么久没有取得位置进展才取消目标。它不是目标总时长；
        # 远距离 frontier 只要仍在接近目标就允许继续运行。
        self.declare_parameter("goal_timeout_sec", 12.0)
        self.declare_parameter("goal_progress_distance_m", 0.08)
        # 另一条看门狗：goal 发出后一直没等到 action server 的接收回执
        # （服务器进程挂掉时 send future 可能永不完成），_goal_pending 会
        # 把探索状态机永久锁死。超过该时长就当作发送失败处理（不计连败，
        # 避免服务器重启期间加速熔断）。0 关闭。
        self.declare_parameter("goal_send_timeout_s", 10.0)
        # 连续失败多少次后，扩大该区域的黑名单半径，强制换方向
        self.declare_parameter("max_consecutive_failures", 3)
        # 连续失败达到该值后自动撤防。0 表示关闭熔断，但真机不建议关闭。
        self.declare_parameter("failure_stop_threshold", 8)
        self.declare_parameter("escalated_failed_goal_radius_m", 2.0)
        self.declare_parameter("auto_wall_escape_enabled", False)
        self.declare_parameter("wall_escape_scan_topic", "/scan")
        # Keep recovery motion separate from Nav2's normal velocity stream.
        self.declare_parameter("wall_escape_cmd_vel_topic", "/cmd_vel_recovery")
        self.declare_parameter("wall_escape_trigger_m", 0.55)
        self.declare_parameter("wall_escape_destination_clearance_m", 0.60)
        self.declare_parameter("wall_escape_stop_clearance_m", 0.40)
        self.declare_parameter("wall_escape_speed_mps", 0.25)
        self.declare_parameter("wall_escape_duration_sec", 1.20)
        self.declare_parameter("wall_escape_cooldown_sec", 3.0)
        self.declare_parameter("wall_escape_max_attempts", 2)
        self.declare_parameter("wall_escape_scan_timeout_sec", 0.50)
        # 横移结束后判定"是否真的挪动了"，没挪动就回退到倒车脱困。
        self.declare_parameter("wall_escape_min_movement_m", 0.25)
        # 倒车脱困：当起点被代价地图误判为 lethal 时，沿来路倒车脱困。
        # 不走 Nav2 规划（规划器拒绝从 lethal 起点规划），直接通过
        # /cmd_vel_recovery 发布反向速度。
        self.declare_parameter("backup_escape_enabled", False)
        self.declare_parameter("backup_distance_m", 1.0)
        self.declare_parameter("backup_speed_mps", 0.20)
        self.declare_parameter("backup_timeout_sec", 15.0)
        self.declare_parameter("backup_cooldown_sec", 3.0)
        self.declare_parameter("backup_max_attempts", 3)
        # 连续失败多少次后触发倒车脱困（即使起点格不是 lethal，只要
        # 控制器连续动不了，就说明被局部代价地图"围住"了）。
        self.declare_parameter("backup_failure_threshold", 2)

        self._map_topic = str(self.get_parameter("map_topic").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._failed_goal_radius_m = float(self.get_parameter("failed_goal_radius_m").value)
        self._failed_goals: Deque[Tuple[float, float]] = deque(
            maxlen=int(self.get_parameter("failed_goal_memory").value)
        )

        self._map: Optional[OccupancyGrid] = None
        self._global_costmap: Optional[OccupancyGrid] = None
        self._active_goal = None
        # Every asynchronous action callback carries this generation.  A late
        # result from an old/preempted goal must never clear or replace the
        # state of a newer goal.
        self._goal_generation = 0
        self._active_goal_generation: Optional[int] = None
        self._goal_pending = False
        self._active_candidate: Optional[FrontierCandidate] = None
        self._last_goal: Optional[PoseStamped] = None
        self._exhausted_reported = False
        self._costmap_filter_reason: Optional[str] = None
        self._goal_sent_time: Optional[Time] = None
        self._goal_last_progress_time: Optional[Time] = None
        self._goal_best_distance_m: Optional[float] = None
        self._consecutive_failures = 0
        self._successful_goal_count = 0
        self._goals_attempted_this_enable = 0
        self._max_goals_per_enable = max(
            0, int(self.get_parameter("max_goals_per_enable").value)
        )
        self._prefer_nearest_until_first_success = bool(
            self.get_parameter("prefer_nearest_until_first_success").value
        )
        self._watchdog_timer = None
        self._replan_requested = False  # 事件驱动标志：目标完成后立即重新规划
        self._last_goal_xy: Optional[Tuple[float, float]] = None
        self._pending_candidate: Optional[FrontierCandidate] = None
        self._pending_candidate_cycles = 0
        self._enabled = bool(self.get_parameter("enabled").value)
        self._latest_scan: Optional[LaserScan] = None
        self._latest_scan_ns = 0
        self._wall_escape_direction = 0.0
        self._wall_escape_until_ns = 0
        self._wall_escape_last_end_ns = 0
        self._wall_escape_attempts = 0
        self._wall_escape_start_xy: Optional[Tuple[float, float]] = None
        self._backup_active = False
        self._backup_start_xy: Optional[Tuple[float, float]] = None
        self._backup_start_ns = 0
        self._backup_attempts = 0
        self._backup_last_end_ns = 0

        # Frontier extraction over a 500x500 map is CPU-heavy and may take
        # several seconds.  Keep the safety scan callback in a distinct group
        # so a MultiThreadedExecutor can refresh it while map search is running.
        # All navigation/action state remains in the node's mutually-exclusive
        # default callback group.
        self._scan_callback_group = MutuallyExclusiveCallbackGroup()

        self._explorer = FrontierExplorer(
            min_frontier_size=int(self.get_parameter("frontier_min_size").value),
            max_frontier_candidates=int(self.get_parameter("frontier_max_candidates").value),
            cluster_connect_distance=int(self.get_parameter("cluster_connect_distance").value),
            free_threshold=int(self.get_parameter("free_threshold").value),
            occupied_threshold=int(self.get_parameter("occupied_threshold").value),
            safety_radius_m=float(self.get_parameter("goal_safety_radius_m").value),
            path_clearance_m=float(self.get_parameter("path_clearance_m").value),
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._nav_client = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter("navigate_action").value),
        )

        self.create_subscription(OccupancyGrid, self._map_topic, self._on_map, 10)
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("global_costmap_topic").value),
            self._on_global_costmap,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("wall_escape_scan_topic").value),
            self._on_scan,
            qos_profile_sensor_data,
            callback_group=self._scan_callback_group,
        )
        self._frontier_pub = self.create_publisher(MarkerArray, "/frontiers", 10)
        self._goal_pub = self.create_publisher(PoseStamped, "/frontier_goal", 10)
        self._event_pub = self.create_publisher(String, "/task_event", 10)
        self._wall_escape_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("wall_escape_cmd_vel_topic").value),
            10,
        )
        self.create_service(SetBool, "~/enable", self._on_enable)

        self.create_timer(
            float(self.get_parameter("replan_interval_sec").value),
            self._exploration_step,
        )
        # 看门狗定时器：独立于重规划周期，专门检测"目标发出后卡死不返回"的情况
        self._watchdog_timer = self.create_timer(2.0, self._check_goal_watchdog)
        self.create_timer(0.05, self._wall_escape_step)
        self.get_logger().info(
            "frontier_explorer_node ready but %s: map_topic=%s base_frame=%s"
            % ("ENABLED" if self._enabled else "DISABLED", self._map_topic, self._base_frame)
        )

    def _on_enable(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self._enabled = bool(request.data)
        if not self._enabled:
            self._stop_wall_escape("frontier disabled")
            self._stop_backup_escape("frontier disabled")
            if self._active_goal is not None:
                try:
                    self._active_goal.cancel_goal_async()
                except Exception as exc:
                    self.get_logger().warn("Cancel goal while disabling failed: %s" % exc)
            self._goal_generation += 1
            self._active_goal_generation = None
            self._goal_pending = False
            self._active_goal = None
            self._active_candidate = None
            self._goal_sent_time = None
            self._goal_last_progress_time = None
            self._goal_best_distance_m = None
            self._reset_pending_candidate()
            self._publish_event("frontier_exploration_disabled")
        else:
            self._goals_attempted_this_enable = 0
            # Each explicit enable starts a new exploration interval.  A
            # previous no-candidate interval may have latched this flag, and
            # keeping it set would suppress the next
            # frontier_waiting_for_reachable_goal/frontier_exhausted event.
            # The mission supervisor relies on that event to start its
            # no-Frontier -> visual-scan fallback timer.
            self._exhausted_reported = False
            self._reset_pending_candidate()
            # Re-read the parameter at every explicit arm so a staged test can
            # change its goal limit without silently retaining the old value.
            self._max_goals_per_enable = max(
                0, int(self.get_parameter("max_goals_per_enable").value)
            )
            self._consecutive_failures = 0
            # A new explicit arm is a new supervised recovery attempt.  The
            # old implementation retained this count forever after two tries.
            self._wall_escape_attempts = 0
            self._backup_attempts = 0
            self._wall_escape_start_xy = None
            self._publish_event("frontier_exploration_enabled")
        response.success = True
        response.message = (
            "frontier exploration enabled (goal limit: %d)"
            % self._max_goals_per_enable
            if self._enabled
            else "frontier exploration disabled"
        )
        self.get_logger().warn(response.message)
        return response

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        self._global_costmap = msg

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._latest_scan_ns = self.get_clock().now().nanoseconds

    def _check_goal_watchdog(self) -> None:
        """Cancel only when a goal has made no positional progress for too long."""
        # A pending-but-unaccepted goal has no _active_goal, so the progress
        # watchdog below never fires for it — and _goal_pending blocks
        # _exploration_step.  Give pending sends their own shorter watchdog.
        if self._goal_pending:
            self._check_pending_goal_staleness()
            return
        if self._goal_last_progress_time is None or self._active_goal is None:
            return

        now = self.get_clock().now()
        if self._active_candidate is not None:
            robot_xy = self._robot_xy_in_map()
            if robot_xy is not None:
                distance = math.hypot(
                    self._active_candidate.x - robot_xy[0],
                    self._active_candidate.y - robot_xy[1],
                )
                progress_step = float(
                    self.get_parameter("goal_progress_distance_m").value
                )
                if (
                    self._goal_best_distance_m is None
                    or self._goal_best_distance_m - distance >= progress_step
                ):
                    self._goal_best_distance_m = distance
                    self._goal_last_progress_time = now

        elapsed = (now - self._goal_last_progress_time).nanoseconds / 1e9
        timeout = float(self.get_parameter("goal_timeout_sec").value)

        if elapsed > timeout:
            self.get_logger().error(
                f"Goal made no positional progress for {elapsed:.1f}s! "
                "Canceling and replanning..."
            )
            # 主动取消当前目标
            try:
                self._active_goal.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f"Cancel goal failed: {e}")

            # 先计入连败再拉黑，使升级半径对本次失败立即生效（此前日志
            # 声称已用升级半径拉黑，实际要等下一轮且连败数达标才生效）。
            self._consecutive_failures += 1
            if self._active_candidate is not None:
                rejected_radius = self._effective_rejected_radius()
                self._failed_goals.append((self._active_candidate.x, self._active_candidate.y))
                self.get_logger().warn(
                    f"Blacklisting area around ({self._active_candidate.x:.2f}, {self._active_candidate.y:.2f}) "
                    f"with radius {rejected_radius}m due to timeout"
                )

            self._goal_generation += 1
            self._active_goal_generation = None
            self._active_goal = None
            self._active_candidate = None
            self._goal_sent_time = None
            self._goal_last_progress_time = None
            self._goal_best_distance_m = None
            self._publish_event("frontier_goal_timeout")
            if self._failure_stop_limit_reached():
                self._self_disable_after_failures()
                return
            if self._goal_attempt_limit_reached():
                self._self_disable_at_goal_limit()
                return
            if self._consecutive_failures >= int(
                self.get_parameter("backup_failure_threshold").value
            ):
                self._maybe_start_escape()
            self._trigger_immediate_replan()

    def _check_pending_goal_staleness(self) -> None:
        """Clear a goal send whose future never resolved (dead action server).

        Not counted as a goal failure: the goal itself was never judged, and
        counting server outages would accelerate the failure fuse while the
        server is restarting.
        """
        if not self._goal_pending or self._goal_sent_time is None:
            return
        timeout = float(self.get_parameter("goal_send_timeout_s").value)
        if timeout <= 0.0:
            return
        elapsed = (self.get_clock().now() - self._goal_sent_time).nanoseconds / 1e9
        if elapsed <= timeout:
            return
        self.get_logger().error(
            "Frontier goal send stayed pending for %.1fs (action server gone?);"
            " clearing and replanning." % elapsed
        )
        # Bump the generation so a late response for this send is ignored.
        self._goal_generation += 1
        self._active_goal_generation = None
        self._goal_pending = False
        self._active_goal = None
        self._active_candidate = None
        self._goal_sent_time = None
        self._goal_last_progress_time = None
        self._goal_best_distance_m = None
        self._publish_event("frontier_goal_send_stale")
        self._trigger_immediate_replan()

    def _trigger_immediate_replan(self) -> None:
        """Request replanning on the regular timer instead of re-entering here.

        Calling ``_exploration_step`` directly from an action result callback
        allowed nested/stale callbacks to create a rapid preemption cascade.
        The timer provides a bounded cooldown and keeps action state changes
        serialized.
        """
        self._replan_requested = True

    def _exploration_step(self) -> None:
        if not self._enabled:
            return
        if self._backup_active:
            return
        if self._wall_escape_direction != 0.0:
            return
        if self._active_goal is not None or self._goal_pending:
            return
        self._replan_requested = False
        if self._map is None:
            self.get_logger().warn("No navigation map received yet.", throttle_duration_sec=5.0)
            return

        robot_pose = self._robot_xy_in_map()
        if robot_pose is None:
            self.get_logger().warn("No map->base transform yet.", throttle_duration_sec=5.0)
            return

        compute_started = time.monotonic()
        candidates = self._find_candidates(robot_pose)
        candidates = self._filter_candidates_in_geofence(candidates)
        candidates = self._filter_candidates_on_global_costmap(candidates, robot_pose)
        compute_elapsed = time.monotonic() - compute_started
        if compute_elapsed >= 0.50:
            self.get_logger().warn(
                "Frontier candidate computation took %.2fs; safety scan callbacks "
                "remain isolated on the sensor executor thread." % compute_elapsed,
                throttle_duration_sec=5.0,
            )
        self._publish_frontier_markers(candidates)
        if not candidates:
            self._reset_pending_candidate()
            if not self._exhausted_reported:
                self._publish_event(
                    "frontier_waiting_for_reachable_goal"
                    if self._costmap_filter_reason
                    else "frontier_exhausted"
                )
                self._exhausted_reported = True
            self.get_logger().warn(
                self._costmap_filter_reason or "No safe frontier candidates found.",
                throttle_duration_sec=5.0,
            )
            return

        self._exhausted_reported = False
        candidate = candidates[0]
        if self._prefer_nearest_until_first_success and self._successful_goal_count == 0:
            candidate = min(candidates, key=lambda item: item.distance)
            self.get_logger().info(
                "First frontier safety policy selected nearest candidate at %.2fm."
                % candidate.distance
            )
        if not self._candidate_is_stable(candidate):
            return
        self._send_goal(candidate, robot_pose)

    def _reset_pending_candidate(self) -> None:
        self._pending_candidate = None
        self._pending_candidate_cycles = 0

    def _candidate_is_stable(self, candidate: FrontierCandidate) -> bool:
        """Return true only after one spatially consistent candidate persists."""
        required = max(
            1, int(self.get_parameter("candidate_stability_required_cycles").value)
        )
        radius = max(
            0.0, float(self.get_parameter("candidate_stability_radius_m").value)
        )
        previous = self._pending_candidate
        if previous is not None and math.hypot(
            candidate.x - previous.x, candidate.y - previous.y
        ) <= radius:
            self._pending_candidate_cycles += 1
        else:
            self._pending_candidate_cycles = 1
        self._pending_candidate = candidate

        if self._pending_candidate_cycles < required:
            # A viable candidate has returned.  Tell the supervisor to cancel
            # any older no-candidate fallback timer while we perform the short
            # temporal-stability check.
            if self._pending_candidate_cycles == 1:
                self._publish_event("frontier_candidate_stabilizing")
            self.get_logger().info(
                "Frontier candidate stabilizing: x=%.2f y=%.2f cycle=%d/%d"
                % (
                    candidate.x,
                    candidate.y,
                    self._pending_candidate_cycles,
                    required,
                ),
                throttle_duration_sec=2.0,
            )
            return False
        self._reset_pending_candidate()
        return True

    def _filter_candidates_in_geofence(
        self, candidates: list[FrontierCandidate]
    ) -> list[FrontierCandidate]:
        min_x = float(self.get_parameter("exploration_min_x").value)
        max_x = float(self.get_parameter("exploration_max_x").value)
        min_y = float(self.get_parameter("exploration_min_y").value)
        max_y = float(self.get_parameter("exploration_max_y").value)
        use_x = max_x > min_x
        use_y = max_y > min_y
        if not use_x and not use_y:
            return candidates
        filtered = [
            candidate
            for candidate in candidates
            if (not use_x or min_x <= candidate.x <= max_x)
            and (not use_y or min_y <= candidate.y <= max_y)
        ]
        if len(filtered) != len(candidates):
            self.get_logger().info(
                "Geofence rejected %d/%d frontier candidates."
                % (len(candidates) - len(filtered), len(candidates))
            )
        return filtered

    def _filter_candidates_on_global_costmap(
        self,
        candidates: list[FrontierCandidate],
        robot_pose: Tuple[float, float],
    ) -> list[FrontierCandidate]:
        self._costmap_filter_reason = None
        require = bool(
            self.get_parameter("require_global_costmap_reachability").value
        )
        if self._global_costmap is None:
            if require:
                self._costmap_filter_reason = "No global costmap received; withholding frontier goals."
                self.get_logger().warn(
                    self._costmap_filter_reason,
                    throttle_duration_sec=5.0,
                )
                return []
            return candidates

        costmap = self._global_costmap
        robot_col, robot_row = world_to_cell(
            robot_pose[0],
            robot_pose[1],
            float(costmap.info.resolution),
            float(costmap.info.origin.position.x),
            float(costmap.info.origin.position.y),
        )
        threshold = int(
            self.get_parameter("global_costmap_traversable_cost_max").value
        )
        if not (
            0 <= robot_col < int(costmap.info.width)
            and 0 <= robot_row < int(costmap.info.height)
        ):
            self._costmap_filter_reason = (
                "Robot is outside the global costmap; withholding frontier goals."
            )
            return []
        robot_cost = int(costmap.data[robot_row * int(costmap.info.width) + robot_col])
        if robot_cost < 0 or robot_cost > threshold:
            self._costmap_filter_reason = (
                "Robot start cell is not traversable on global costmap "
                f"(cost={robot_cost}); waiting instead of sending doomed goals."
            )
            self._publish_event("frontier_start_blocked:cost=%d" % robot_cost)
            if robot_cost >= 99:
                self._maybe_start_escape()
            return []
        self._wall_escape_attempts = 0
        filtered = self._explorer.filter_reachable_on_costmap(
            candidates=candidates,
            width=int(costmap.info.width),
            height=int(costmap.info.height),
            resolution=float(costmap.info.resolution),
            origin_x=float(costmap.info.origin.position.x),
            origin_y=float(costmap.info.origin.position.y),
            data=list(costmap.data),
            robot_x=robot_pose[0],
            robot_y=robot_pose[1],
            traversable_cost_max=threshold,
            goal_tolerance_m=float(
                self.get_parameter("global_costmap_goal_tolerance_m").value
            ),
            goal_clearance_m=float(
                self.get_parameter("global_costmap_goal_clearance_m").value
            ),
            goal_clearance_cost_max=int(
                self.get_parameter("global_costmap_goal_clearance_cost_max").value
            ),
        )
        rejected_count = len(candidates) - len(filtered)
        if rejected_count:
            self.get_logger().info(
                "Global-costmap reachability rejected %d/%d frontier candidates."
                % (rejected_count, len(candidates))
            )
        if candidates and not filtered:
            self._costmap_filter_reason = (
                "No frontier candidate is connected on the global costmap; waiting for map updates."
            )
        return filtered

    def _maybe_start_escape(self) -> None:
        """Try lateral wall escape first, then fall back to reverse backup."""
        if self._maybe_start_wall_escape():
            return
        self._maybe_start_backup_escape()

    def _maybe_start_backup_escape(self) -> bool:
        if not bool(self.get_parameter("backup_escape_enabled").value):
            return False
        if self._backup_active or self._active_goal is not None or self._goal_pending:
            return False
        now_ns = self.get_clock().now().nanoseconds
        cooldown_ns = int(
            float(self.get_parameter("backup_cooldown_sec").value) * 1e9
        )
        if now_ns - self._backup_last_end_ns < cooldown_ns:
            return False
        if self._backup_attempts >= int(
            self.get_parameter("backup_max_attempts").value
        ):
            self.get_logger().warn(
                "Backup escape attempt limit reached; waiting for operator.",
                throttle_duration_sec=5.0,
            )
            return False
        robot = self._robot_xy_in_map()
        if robot is None:
            self.get_logger().warn(
                "No robot pose for backup escape; trying lateral escape.",
                throttle_duration_sec=5.0,
            )
            return False
        self._backup_active = True
        self._backup_start_xy = robot
        self._backup_start_ns = now_ns
        self._backup_attempts += 1
        self._publish_event(
            "backup_escape_started:attempt=%d" % self._backup_attempts
        )
        self.get_logger().warn(
            "Backing up %.1fm at %.2fm/s (attempt %d)."
            % (
                float(self.get_parameter("backup_distance_m").value),
                float(self.get_parameter("backup_speed_mps").value),
                self._backup_attempts,
            )
        )
        return True

    def _backup_escape_step(self) -> None:
        if not self._backup_active:
            return
        now_ns = self.get_clock().now().nanoseconds
        robot = self._robot_xy_in_map()
        stop_reason = ""
        if not self._enabled:
            stop_reason = "frontier disabled"
        elif self._backup_start_xy is None or robot is None:
            stop_reason = "no robot pose"
        else:
            moved = math.hypot(
                robot[0] - self._backup_start_xy[0],
                robot[1] - self._backup_start_xy[1],
            )
            if moved >= float(self.get_parameter("backup_distance_m").value) * 0.9:
                stop_reason = "reached"
            elif (
                now_ns - self._backup_start_ns
                > float(self.get_parameter("backup_timeout_sec").value) * 1e9
            ):
                stop_reason = "timeout"
        if stop_reason:
            self._stop_backup_escape(stop_reason)
            return
        command = Twist()
        command.linear.x = -float(self.get_parameter("backup_speed_mps").value)
        self._wall_escape_pub.publish(command)

    def _stop_backup_escape(self, reason: str) -> None:
        if not self._backup_active:
            return
        self._wall_escape_pub.publish(Twist())
        self._backup_active = False
        self._backup_start_xy = None
        self._backup_last_end_ns = self.get_clock().now().nanoseconds
        if reason == "reached":
            self._publish_event("backup_escape_finished")
        else:
            self._publish_event("backup_escape_failed:%s" % reason)
        self.get_logger().warn("Backup escape stopped: %s." % reason)
        self._trigger_immediate_replan()

    def _maybe_start_wall_escape(self) -> bool:
        if not bool(self.get_parameter("auto_wall_escape_enabled").value):
            return False
        if self._active_goal is not None or self._goal_pending:
            return False
        now_ns = self.get_clock().now().nanoseconds
        cooldown_ns = int(
            float(self.get_parameter("wall_escape_cooldown_sec").value) * 1e9
        )
        if now_ns - self._wall_escape_last_end_ns < cooldown_ns:
            return False
        if self._wall_escape_attempts >= int(
            self.get_parameter("wall_escape_max_attempts").value
        ):
            self.get_logger().warn(
                "Wall escape attempt limit reached; disable and re-enable Frontier "
                "after checking the surroundings to authorize a new attempt.",
                throttle_duration_sec=5.0,
            )
            return False
        scan_timeout_ns = int(
            float(self.get_parameter("wall_escape_scan_timeout_sec").value) * 1e9
        )
        scan = self._latest_scan
        if scan is None or now_ns - self._latest_scan_ns > scan_timeout_ns:
            self.get_logger().warn(
                "Start cell blocked, but /scan is missing or stale; lateral escape withheld.",
                throttle_duration_sec=5.0,
            )
            return False
        left, right = side_sector_clearances(
            scan.ranges, float(scan.angle_min), float(scan.angle_increment)
        )
        direction = choose_lateral_escape_direction(
            left,
            right,
            float(self.get_parameter("wall_escape_trigger_m").value),
            float(self.get_parameter("wall_escape_destination_clearance_m").value),
        )
        if direction == 0.0:
            self.get_logger().warn(
                "Start cell blocked, but neither lateral destination is scan-safe "
                f"(left={left:.2f}m right={right:.2f}m)."
                ,
                throttle_duration_sec=5.0,
            )
            return False
        self._wall_escape_direction = direction
        self._wall_escape_start_xy = self._robot_xy_in_map()
        self._wall_escape_until_ns = now_ns + int(
            float(self.get_parameter("wall_escape_duration_sec").value) * 1e9
        )
        self._wall_escape_attempts += 1
        self._publish_event(
            "wall_escape_left" if direction > 0.0 else "wall_escape_right"
        )
        self.get_logger().warn(
            "Starting bounded lateral wall escape: direction=%s attempt=%d."
            % ("left" if direction > 0.0 else "right", self._wall_escape_attempts)
        )
        return True

    def _wall_escape_step(self) -> None:
        if self._backup_active:
            self._backup_escape_step()
            return
        if self._wall_escape_direction == 0.0:
            return
        now_ns = self.get_clock().now().nanoseconds
        scan = self._latest_scan
        scan_timeout_ns = int(
            float(self.get_parameter("wall_escape_scan_timeout_sec").value) * 1e9
        )
        if (
            not self._enabled
            or scan is None
            or now_ns - self._latest_scan_ns > scan_timeout_ns
            or now_ns >= self._wall_escape_until_ns
        ):
            self._stop_wall_escape("completed or safety input unavailable")
            return
        left, right = side_sector_clearances(
            scan.ranges, float(scan.angle_min), float(scan.angle_increment)
        )
        destination = left if self._wall_escape_direction > 0.0 else right
        if destination <= float(
            self.get_parameter("wall_escape_stop_clearance_m").value
        ):
            self._stop_wall_escape("destination side became obstructed")
            return
        command = Twist()
        command.linear.y = self._wall_escape_direction * float(
            self.get_parameter("wall_escape_speed_mps").value
        )
        self._wall_escape_pub.publish(command)

    def _stop_wall_escape(self, reason: str) -> None:
        if self._wall_escape_direction == 0.0:
            return
        self._wall_escape_pub.publish(Twist())
        self._wall_escape_direction = 0.0
        self._wall_escape_until_ns = 0
        self._wall_escape_last_end_ns = self.get_clock().now().nanoseconds
        self._publish_event("wall_escape_stopped")
        self.get_logger().warn("Wall escape stopped: %s." % reason)
        moved = True
        if self._wall_escape_start_xy is not None:
            robot = self._robot_xy_in_map()
            if robot is not None:
                moved = math.hypot(
                    robot[0] - self._wall_escape_start_xy[0],
                    robot[1] - self._wall_escape_start_xy[1],
                ) >= float(self.get_parameter("wall_escape_min_movement_m").value)
            else:
                moved = False
            self._wall_escape_start_xy = None
        if self._enabled and not moved:
            self.get_logger().warn(
                "Lateral escape did not move the robot; trying reverse backup."
            )
            self._maybe_start_backup_escape()
        self._trigger_immediate_replan()

    def _effective_rejected_radius(self) -> float:
        """当前应使用的失败目标黑名单半径。

        连败数达到 max_consecutive_failures 时立即升级到
        escalated_failed_goal_radius_m（包括刚被计数的这一次失败），
        保证超时处理里"以升级半径拉黑"的日志与实际过滤行为一致。
        """
        max_failures = int(self.get_parameter("max_consecutive_failures").value)
        if self._consecutive_failures >= max_failures:
            return float(self.get_parameter("escalated_failed_goal_radius_m").value)
        return self._failed_goal_radius_m

    def _find_candidates(self, robot_pose: Tuple[float, float]) -> list[FrontierCandidate]:
        assert self._map is not None
        # 连续失败多次后，扩大黑名单半径，强制换一个更远的方向，避免死角死循环
        rejected_radius = self._effective_rejected_radius()
        if rejected_radius > self._failed_goal_radius_m:
            self.get_logger().warn(
                f"{self._consecutive_failures} consecutive failures, "
                f"escalating avoidance radius to {rejected_radius}m"
            )
        return self._explorer.find_frontiers(
            width=int(self._map.info.width),
            height=int(self._map.info.height),
            resolution=float(self._map.info.resolution),
            origin_x=float(self._map.info.origin.position.x),
            origin_y=float(self._map.info.origin.position.y),
            data=list(self._map.data),
            robot_x=robot_pose[0],
            robot_y=robot_pose[1],
            rejected_goals=self._failed_goals,
            rejected_radius_m=rejected_radius,
            previous_goal_xy=self._last_goal_xy,
        )

    def _robot_xy_in_map(self) -> Optional[Tuple[float, float]]:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException:
            return None
        return transform.transform.translation.x, transform.transform.translation.y

    def _send_goal(
        self,
        candidate: FrontierCandidate,
        robot_pose: Tuple[float, float],
    ) -> None:
        if not self._nav_client.wait_for_server(timeout_sec=1.0):
            self._publish_event("navigation_failed")
            self.get_logger().error("Nav2 /navigate_to_pose action server unavailable.")
            return

        self._reset_pending_candidate()

        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = candidate.x
        pose.pose.position.y = candidate.y
        goal_yaw = math.atan2(
            candidate.y - robot_pose[1], candidate.x - robot_pose[0]
        )
        pose.pose.orientation.z = math.sin(0.5 * goal_yaw)
        pose.pose.orientation.w = math.cos(0.5 * goal_yaw)

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goals_attempted_this_enable += 1
        self._last_goal = pose
        self._active_candidate = candidate
        self._last_goal_xy = (candidate.x, candidate.y)
        self._goal_sent_time = self.get_clock().now()
        self._goal_last_progress_time = self._goal_sent_time
        self._goal_best_distance_m = candidate.distance
        self._goal_pub.publish(pose)
        self.get_logger().info(
            "Exploring frontier: x=%.2f y=%.2f size=%d distance=%.2f score=%.2f"
            % (candidate.x, candidate.y, candidate.size, candidate.distance, candidate.score)
        )

        future = self._nav_client.send_goal_async(goal)
        self._goal_generation += 1
        generation = self._goal_generation
        self._active_goal_generation = generation
        self._goal_pending = True
        future.add_done_callback(
            lambda done, goal_generation=generation: self._on_goal_response(
                done, goal_generation
            )
        )

    def _on_goal_response(self, future, generation: int) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            if generation != self._active_goal_generation:
                return
            self._goal_pending = False
            self._consecutive_failures += 1
            self._active_goal_generation = None
            self._active_candidate = None
            self._goal_sent_time = None
            self._goal_last_progress_time = None
            self._goal_best_distance_m = None
            self._publish_event("frontier_goal_send_failed")
            self.get_logger().error("Sending frontier goal failed: %s" % exc)
            if self._failure_stop_limit_reached():
                self._self_disable_after_failures()
            else:
                self._trigger_immediate_replan()
            return

        if generation != self._active_goal_generation or not self._enabled:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        self._goal_pending = False
        if not goal_handle.accepted:
            self._remember_failed_goal()
            self._consecutive_failures += 1
            self._goal_sent_time = None
            self._goal_last_progress_time = None
            self._goal_best_distance_m = None
            self._active_candidate = None
            self._publish_event("frontier_goal_rejected")
            self._active_goal = None
            self._active_goal_generation = None
            if self._failure_stop_limit_reached():
                self._self_disable_after_failures()
                return
            if self._goal_attempt_limit_reached():
                self._self_disable_at_goal_limit()
                return
            self._trigger_immediate_replan()
            return

        self._active_goal = goal_handle
        # Tell the mission supervisor that a real Frontier goal now exists so
        # it can cancel any pending no-candidate -> scan fallback timer.
        self._publish_event("frontier_goal_started")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done, goal_generation=generation: self._on_goal_result(
                done, goal_generation
            )
        )

    def _on_goal_result(self, future, generation: int) -> None:
        if generation != self._active_goal_generation:
            self.get_logger().debug(
                "Ignoring stale frontier result for generation %d." % generation
            )
            return
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error("Reading frontier result failed: %s" % exc)
            result = None
        status = result.status if result is not None else GoalStatus.STATUS_UNKNOWN
        self._goal_sent_time = None  # 清除计时器，目标已返回结果
        self._goal_last_progress_time = None
        self._goal_best_distance_m = None
        if not self._enabled:
            self._active_goal = None
            self._active_goal_generation = None
            self._active_candidate = None
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._publish_event("frontier_goal_reached")
            self._successful_goal_count += 1
            self._consecutive_failures = 0  # 成功就清零失败计数
            self.get_logger().info("Frontier goal reached, continuing exploration...")
        elif status == GoalStatus.STATUS_ABORTED:
            # ABORTED状态（通常是progress timeout或避障失败）记录为失败
            self._remember_failed_goal()
            self._consecutive_failures += 1
            self.get_logger().warn(
                f"Goal aborted (consecutive failures: {self._consecutive_failures}), "
                "will try different frontier"
            )
            self._publish_event("frontier_goal_aborted")
        elif status == GoalStatus.STATUS_CANCELED:
            # 外部取消（supervisor 切换到扫描/视觉停靠抢占）是层间协作，
            # 不是导航失败：不拉黑该 frontier、不计入连败，否则一次正常的
            # 扫描切换会白白积累失败计数甚至触发自熔断。
            # 自己看门狗取消的目标在到达这里之前已被 generation 守卫丢弃。
            self.get_logger().info(
                "Frontier goal canceled externally; not counted as a failure."
            )
            self._publish_event("frontier_goal_canceled")
        else:
            self._remember_failed_goal()
            self._consecutive_failures += 1
            self.get_logger().warn(
                f"Goal failed with status {status} (consecutive failures: {self._consecutive_failures})"
            )
            self._publish_event("frontier_goal_failed")
        self._active_goal = None
        self._active_goal_generation = None
        self._active_candidate = None
        if self._failure_stop_limit_reached():
            self._self_disable_after_failures()
            return
        if self._goal_attempt_limit_reached():
            self._self_disable_at_goal_limit()
            return
        if self._consecutive_failures >= int(
            self.get_parameter("backup_failure_threshold").value
        ):
            self._maybe_start_escape()
        self._trigger_immediate_replan()  # 无论成功失败，目标结束后立即重新规划下一个

    def _goal_attempt_limit_reached(self) -> bool:
        return (
            self._max_goals_per_enable > 0
            and self._goals_attempted_this_enable >= self._max_goals_per_enable
        )

    def _self_disable_at_goal_limit(self) -> None:
        self._stop_wall_escape("goal limit reached")
        self._enabled = False
        self._publish_event("frontier_goal_limit_reached")
        self._publish_event("frontier_exploration_disabled")
        self.get_logger().warn(
            "Frontier goal-attempt limit reached (%d); exploration self-disabled."
            % self._max_goals_per_enable
        )

    def _failure_stop_limit_reached(self) -> bool:
        threshold = int(self.get_parameter("failure_stop_threshold").value)
        return threshold > 0 and self._consecutive_failures >= threshold

    def _self_disable_after_failures(self) -> None:
        self._stop_wall_escape("failure limit reached")
        self._enabled = False
        self._goal_generation += 1
        self._active_goal_generation = None
        self._goal_pending = False
        self._active_goal = None
        self._active_candidate = None
        self._goal_sent_time = None
        self._goal_last_progress_time = None
        self._goal_best_distance_m = None
        self._publish_event("frontier_failure_limit_reached")
        self._publish_event("frontier_exploration_disabled")
        self.get_logger().error(
            "Frontier exploration self-disabled after %d consecutive failures."
            % self._consecutive_failures
        )

    def _remember_failed_goal(self) -> None:
        if self._active_candidate is not None:
            self._failed_goals.append((self._active_candidate.x, self._active_candidate.y))

    def _publish_frontier_markers(self, candidates: list[FrontierCandidate]) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._map_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for idx, candidate in enumerate(candidates):
            marker = Marker()
            marker.header.frame_id = self._map_frame
            marker.header.stamp = clear.header.stamp
            marker.ns = "frontier_candidates"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = candidate.x
            marker.pose.position.y = candidate.y
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.22
            marker.scale.y = 0.22
            marker.scale.z = 0.10
            marker.color.r = 0.1
            marker.color.g = 0.7
            marker.color.b = 1.0
            marker.color.a = 0.85
            markers.markers.append(marker)

        self._frontier_pub.publish(markers)

    def _publish_event(self, name: str) -> None:
        msg = String()
        msg.data = name
        self._event_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrontierExplorerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
