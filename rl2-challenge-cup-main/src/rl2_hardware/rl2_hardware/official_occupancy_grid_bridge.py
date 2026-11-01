"""Project official SLAM point clouds into the occupancy grid used by frontier exploration."""

from __future__ import annotations

import array
import json
import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from rl2_hardware.pointcloud_utils import pointcloud_xyz, yaw_from_quaternion


def horizontal_range_mask(
    xyz: np.ndarray, max_range: float, origin_x: float = 0.0, origin_y: float = 0.0
) -> np.ndarray:
    """Select points within ``max_range`` of an explicit XY origin."""
    if max_range <= 0.0:
        return np.ones(len(xyz), dtype=bool)
    return np.hypot(xyz[:, 0] - origin_x, xyz[:, 1] - origin_y) <= max_range


def official_slam_inputs_stale(
    last_pose_s: float, last_cloud_s: float, now_s: float, timeout_s: float
) -> bool:
    """Return true only after a previously live SLAM input pair goes stale."""
    if timeout_s <= 0.0 or last_pose_s <= 0.0 or last_cloud_s <= 0.0:
        return False
    return now_s - last_pose_s > timeout_s or now_s - last_cloud_s > timeout_s


def official_slam_stale_reason(
    last_pose_s: float, last_cloud_s: float, now_s: float, timeout_s: float
) -> str:
    """Describe which previously live official SLAM input timed out."""
    if timeout_s <= 0.0 or last_pose_s <= 0.0 or last_cloud_s <= 0.0:
        return "not_live_yet"
    pose_stale = now_s - last_pose_s > timeout_s
    cloud_stale = now_s - last_cloud_s > timeout_s
    if pose_stale and cloud_stale:
        return "pose_and_cloud_stale"
    if pose_stale:
        return "pose_stale"
    if cloud_stale:
        return "cloud_stale"
    return "healthy"


def clear_oriented_footprint(
    grid: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    half_length_m: float,
    half_width_m: float,
) -> int:
    """Clear only cells physically occupied by the robot body footprint."""
    if half_length_m <= 0.0 or half_width_m <= 0.0 or resolution <= 0.0:
        return 0
    radius = math.hypot(half_length_m, half_width_m)
    min_col = max(0, int(math.floor((robot_x - radius - origin_x) / resolution)))
    max_col = min(grid.shape[1] - 1, int(math.floor((robot_x + radius - origin_x) / resolution)))
    min_row = max(0, int(math.floor((robot_y - radius - origin_y) / resolution)))
    max_row = min(grid.shape[0] - 1, int(math.floor((robot_y + radius - origin_y) / resolution)))
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)
    cleared = 0
    for row in range(min_row, max_row + 1):
        world_y = origin_y + (row + 0.5) * resolution
        for col in range(min_col, max_col + 1):
            world_x = origin_x + (col + 0.5) * resolution
            dx, dy = world_x - robot_x, world_y - robot_y
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            if abs(local_x) <= half_length_m and abs(local_y) <= half_width_m:
                grid[row, col] = 0
                cleared += 1
    return cleared


class OfficialOccupancyGridBridge(Node):
    """Build a growing 2D map from official SLAM point clouds."""

    def __init__(self) -> None:
        super().__init__("official_occupancy_grid_bridge")

        self.declare_parameter("pointcloud_topic", "/slam/points")
        self.declare_parameter("clearing_pointcloud_topic", "")
        self.declare_parameter("current_pose_topic", "/slam/current_pose")
        self.declare_parameter("navigation_map_topic", "/navigation_map")
        self.declare_parameter("command_status_topic", "/unitree_slam/command_status")
        self.declare_parameter("reset_on_start_mapping", True)
        # Unitree SLAM may occasionally apply a large map-frame correction.
        # Keeping cells accumulated in the old frame would make Nav2's robot
        # pose land in an unrelated lethal/unknown region.
        self.declare_parameter("reset_on_pose_jump", True)
        self.declare_parameter("pose_jump_distance_m", 1.0)
        self.declare_parameter("map_frame", "slam_map")
        self.declare_parameter("map_size_m", 25.0)
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("max_range_m", 8.0)
        self.declare_parameter("min_height_m", -0.30)
        self.declare_parameter("max_height_m", 1.20)
        self.declare_parameter("ground_free_enabled", True)
        self.declare_parameter("ground_min_height_m", -0.65)
        self.declare_parameter("ground_max_height_m", -0.32)
        self.declare_parameter("ground_free_radius_m", 0.10)
        self.declare_parameter("live_clearing_enabled", True)
        self.declare_parameter("max_clearing_rays", 2500)
        # One noisy return (insect, multipath) used to mark a cell lethal
        # until manually reset.  Require this many distinct-cloud hits before
        # writing 100 (1 restores instant marking).  At ~8.7 Hz two hits delay
        # the global map by only ~0.23 s; local avoidance still reacts through
        # the /scan obstacle layer, which is uncounted.
        self.declare_parameter("obstacle_confirm_hits", 2)
        # Dock-CPU kindness: rate-limit the live raycast clearing.  The primary
        # cloud arrives at ~8.7 Hz but re-clearing every frame is unnecessary;
        # 0.25 s still removes ghost walls quickly.  0 disables the limit.
        self.declare_parameter("min_clearing_period_s", 0.25)
        self.declare_parameter("max_points", 20000)
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("pointcloud_in_map", True)
        self.declare_parameter("clear_robot_footprint", True)
        self.declare_parameter("footprint_half_length_m", 0.34)
        self.declare_parameter("footprint_half_width_m", 0.22)
        # A retained OccupancyGrid can make RViz look healthy long after the
        # official SLAM cloud/pose stream died. Clear it once on input loss.
        self.declare_parameter("clear_map_on_input_stale", True)
        self.declare_parameter("input_stale_timeout_s", 2.0)

        map_size = float(self.get_parameter("map_size_m").value)
        resolution = float(self.get_parameter("resolution").value)
        self._width = int(math.floor(map_size / resolution))
        self._height = self._width
        self._resolution = resolution
        self._origin_x = -self._width * resolution / 2.0
        self._origin_y = -self._height * resolution / 2.0
        self._map_frame = str(self.get_parameter("map_frame").value)

        self._grid = np.full((self._height, self._width), -1, dtype=np.int8)
        # Per-cell obstacle hit counter backing obstacle_confirm_hits.
        self._hits = np.zeros((self._height, self._width), dtype=np.int8)
        self._last_clearing_apply_s = 0.0
        self._latest_points: Optional[np.ndarray] = None
        self._latest_pose: Optional[PoseStamped] = None
        self._last_pose_time_s = 0.0
        self._last_cloud_time_s = 0.0
        self._stale_input_latched = False
        self._accumulation_enabled = True

        primary_topic = str(self.get_parameter("pointcloud_topic").value)
        clearing_topic = str(self.get_parameter("clearing_pointcloud_topic").value)
        self._primary_is_clearing = bool(clearing_topic and clearing_topic == primary_topic)
        self.create_subscription(
            PointCloud2,
            primary_topic,
            self._on_pointcloud,
            qos_profile_sensor_data,
        )
        if clearing_topic and clearing_topic != primary_topic:
            self.create_subscription(
                PointCloud2,
                clearing_topic,
                self._on_clearing_pointcloud,
                qos_profile_sensor_data,
            )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("current_pose_topic").value),
            self._on_current_pose,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("command_status_topic").value),
            self._on_command_status,
            10,
        )
        self.create_service(
            Trigger,
            "/official_occupancy_grid_bridge/reset",
            self._on_reset,
        )
        self.create_service(
            SetBool,
            "/official_occupancy_grid_bridge/set_accumulation",
            self._on_set_accumulation,
        )

        self._map_pub = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter("navigation_map_topic").value),
            10,
        )
        self._health_pub = self.create_publisher(
            String, "/rl2_nav/official_slam_stream_health", 10
        )
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish_map)
        self.create_timer(0.5, self._clear_map_if_inputs_stale)
        self.create_timer(1.0, self._publish_stream_health)

        self.get_logger().info(
            "official_occupancy_grid_bridge ready: %dx%d @ %.3fm"
            % (self._width, self._height, self._resolution)
        )

    def _on_current_pose(self, msg: PoseStamped) -> None:
        if self._latest_pose is not None and _as_bool(
            self.get_parameter("reset_on_pose_jump").value
        ):
            previous = self._latest_pose.pose.position
            current = msg.pose.position
            distance = math.hypot(current.x - previous.x, current.y - previous.y)
            threshold = float(self.get_parameter("pose_jump_distance_m").value)
            if threshold > 0.0 and distance > threshold:
                self._reset_map(
                    "official SLAM pose jump %.2fm exceeded %.2fm"
                    % (distance, threshold)
                )
        self._latest_pose = msg
        self._last_pose_time_s = self.get_clock().now().nanoseconds / 1e9
        self._rearm_stale_input_latch()

    def _reset_map(self, reason: str) -> None:
        self._grid.fill(-1)
        self._hits.fill(0)
        self._latest_points = None
        # Do not apply a cloud using a pose left over from the previous mapping
        # session.  The next pose message re-arms point processing naturally.
        self._latest_pose = None
        self.get_logger().warn("Cleared accumulated occupancy grid: %s" % reason)

    def _rearm_stale_input_latch(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if not official_slam_inputs_stale(
            self._last_pose_time_s,
            self._last_cloud_time_s,
            now,
            float(self.get_parameter("input_stale_timeout_s").value),
        ):
            self._stale_input_latched = False

    def _clear_map_if_inputs_stale(self) -> None:
        if not _as_bool(self.get_parameter("clear_map_on_input_stale").value):
            return
        if self._stale_input_latched or not np.any(self._grid >= 0):
            return
        now = self.get_clock().now().nanoseconds / 1e9
        timeout = float(self.get_parameter("input_stale_timeout_s").value)
        if not official_slam_inputs_stale(
            self._last_pose_time_s, self._last_cloud_time_s, now, timeout
        ):
            return
        self._stale_input_latched = True
        reason = official_slam_stale_reason(
            self._last_pose_time_s, self._last_cloud_time_s, now, timeout
        )
        self._reset_map(
            "official SLAM input stream stale (%s; pose_age=%.2fs cloud_age=%.2fs)"
            % (
                reason,
                now - self._last_pose_time_s,
                now - self._last_cloud_time_s,
            )
        )

    def _publish_stream_health(self) -> None:
        """Expose receipt-time health; header stamps are not trusted here."""
        now = self.get_clock().now().nanoseconds / 1e9
        timeout = float(self.get_parameter("input_stale_timeout_s").value)
        payload = {
            "pose_age_s": round(max(0.0, now - self._last_pose_time_s), 3)
            if self._last_pose_time_s > 0.0
            else None,
            "cloud_age_s": round(max(0.0, now - self._last_cloud_time_s), 3)
            if self._last_cloud_time_s > 0.0
            else None,
            "timeout_s": timeout,
            "state": official_slam_stale_reason(
                self._last_pose_time_s, self._last_cloud_time_s, now, timeout
            ),
            "stale_latched": self._stale_input_latched,
        }
        self._health_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _on_reset(self, request, response):
        del request
        self._reset_map("manual reset service")
        response.success = True
        response.message = "occupancy grid cleared"
        return response

    def _on_set_accumulation(self, request, response):
        self._accumulation_enabled = bool(request.data)
        response.success = True
        response.message = (
            "occupancy accumulation enabled"
            if self._accumulation_enabled
            else "occupancy accumulation paused; stream health remains monitored"
        )
        self.get_logger().warn(response.message)
        return response

    def _on_command_status(self, msg: String) -> None:
        if not _as_bool(self.get_parameter("reset_on_start_mapping").value):
            return
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        # API 1801 is Unitree's start-mapping command.  Only clear after the
        # official service reports success, never merely when it is queued.
        if int(payload.get("api_id", 0)) == 1801 and _as_bool(
            payload.get("succeed", False)
        ):
            self._reset_map("official start-mapping API 1801 succeeded")

    def _on_pointcloud(self, msg: PointCloud2) -> None:
        self._last_cloud_time_s = self.get_clock().now().nanoseconds / 1e9
        self._rearm_stale_input_latch()
        if not self._accumulation_enabled:
            return
        xyz = pointcloud_xyz(msg)
        if xyz is None:
            return
        max_points = int(self.get_parameter("max_points").value)
        if len(xyz) > max_points:
            indices = np.linspace(0, len(xyz) - 1, max_points, dtype=int)
            xyz = xyz[indices]

        min_h = float(self.get_parameter("min_height_m").value)
        max_h = float(self.get_parameter("max_height_m").value)
        max_range = float(self.get_parameter("max_range_m").value)
        finite = np.isfinite(xyz).all(axis=1)
        pointcloud_in_map = _as_bool(self.get_parameter("pointcloud_in_map").value)
        if pointcloud_in_map:
            pose = self._latest_pose
            if pose is None:
                self.get_logger().warn(
                    "Map-frame cloud received before current pose; skipping range filter frame.",
                    throttle_duration_sec=5.0,
                )
                return
            finite &= horizontal_range_mask(
                xyz,
                max_range,
                float(pose.pose.position.x),
                float(pose.pose.position.y),
            )
        else:
            finite &= horizontal_range_mask(xyz, max_range)
        xyz = xyz[finite]
        if len(xyz) == 0:
            return

        if not pointcloud_in_map:
            xyz = self._transform_points_to_map(xyz)
            if xyz is None:
                return

        if self._primary_is_clearing:
            # Raycast-only here: the shared marking block below already counts
            # this cloud's obstacles, and counting it twice would defeat
            # obstacle_confirm_hits.
            self._apply_live_clearing_xyz(xyz, mark_obstacles=False)

        if _as_bool(self.get_parameter("ground_free_enabled").value):
            ground_min = float(self.get_parameter("ground_min_height_m").value)
            ground_max = float(self.get_parameter("ground_max_height_m").value)
            ground = xyz[(xyz[:, 2] >= ground_min) & (xyz[:, 2] <= ground_max)]
            self._mark_ground_free(ground)

        obstacles = xyz[(xyz[:, 2] >= min_h) & (xyz[:, 2] <= max_h)]
        if len(obstacles) == 0:
            if _as_bool(self.get_parameter("clear_robot_footprint").value):
                self._clear_current_robot_footprint()
            return

        cols = np.floor((obstacles[:, 0] - self._origin_x) / self._resolution).astype(int)
        rows = np.floor((obstacles[:, 1] - self._origin_y) / self._resolution).astype(int)
        valid = (
            (cols >= 0)
            & (cols < self._width)
            & (rows >= 0)
            & (rows < self._height)
        )
        cols = cols[valid]
        rows = rows[valid]
        if len(cols) == 0:
            return

        # A saved global PCD contains returns accumulated from many historical
        # sensor poses. Raycasting all of them from the robot's *current* pose
        # is geometrically invalid and can erase walls. Free space is instead
        # obtained from ground evidence above and the live clearing cloud.
        # Free-space evidence wipes hit credit: a cleared cell must re-earn
        # lethal status from scratch instead of re-confirming off stale hits.
        self._hits[self._grid == 0] = 0
        self._mark_obstacle_cells(rows, cols)
        if _as_bool(self.get_parameter("clear_robot_footprint").value):
            self._clear_current_robot_footprint()

    def _mark_ground_free(self, xyz: np.ndarray) -> None:
        if len(xyz) == 0:
            return
        cols = np.floor((xyz[:, 0] - self._origin_x) / self._resolution).astype(int)
        rows = np.floor((xyz[:, 1] - self._origin_y) / self._resolution).astype(int)
        valid = (
            (cols >= 0) & (cols < self._width) &
            (rows >= 0) & (rows < self._height)
        )
        cols, rows = cols[valid], rows[valid]
        radius = max(
            0,
            int(math.ceil(
                float(self.get_parameter("ground_free_radius_m").value)
                / self._resolution
            )),
        )
        for dy in range(-radius, radius + 1):
            rr = rows + dy
            row_ok = (rr >= 0) & (rr < self._height)
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                cc = cols + dx
                ok = row_ok & (cc >= 0) & (cc < self._width)
                self._grid[rr[ok], cc[ok]] = 0

    def _on_clearing_pointcloud(self, msg: PointCloud2) -> None:
        if not self._accumulation_enabled:
            return
        if not _as_bool(self.get_parameter("live_clearing_enabled").value):
            return
        xyz = pointcloud_xyz(msg)
        pose = self._latest_pose
        robot_cell = self._robot_cell()
        if xyz is None or pose is None or robot_cell is None:
            return
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) == 0:
            return
        self._apply_live_clearing_xyz(xyz)

    def _apply_live_clearing_xyz(
        self, xyz: np.ndarray, mark_obstacles: bool = True
    ) -> None:
        pose = self._latest_pose
        robot_cell = self._robot_cell()
        if pose is None or robot_cell is None or len(xyz) == 0:
            return
        now_s = self.get_clock().now().nanoseconds / 1e9
        min_period = max(
            0.0, float(self.get_parameter("min_clearing_period_s").value)
        )
        if min_period > 0.0 and now_s - self._last_clearing_apply_s < min_period:
            return
        self._last_clearing_apply_s = now_s
        max_range = float(self.get_parameter("max_range_m").value)
        xyz = xyz[horizontal_range_mask(
            xyz,
            max_range,
            float(pose.pose.position.x),
            float(pose.pose.position.y),
        )]
        max_rays = max(1, int(self.get_parameter("max_clearing_rays").value))
        if len(xyz) > max_rays:
            xyz = xyz[np.linspace(0, len(xyz) - 1, max_rays, dtype=int)]
        cols = np.floor((xyz[:, 0] - self._origin_x) / self._resolution).astype(int)
        rows = np.floor((xyz[:, 1] - self._origin_y) / self._resolution).astype(int)
        valid = (
            (cols >= 0) & (cols < self._width) &
            (rows >= 0) & (rows < self._height)
        )
        cols, rows = cols[valid], rows[valid]
        if len(cols) == 0:
            return
        self._raycast_free(robot_cell, cols, rows)

        if not mark_obstacles:
            return
        min_h = float(self.get_parameter("min_height_m").value)
        max_h = float(self.get_parameter("max_height_m").value)
        obstacle = valid & (xyz[:, 2] >= min_h) & (xyz[:, 2] <= max_h)
        # Recompute from the unfiltered arrays selected by the same valid mask.
        obstacle_cols = np.floor(
            (xyz[obstacle, 0] - self._origin_x) / self._resolution
        ).astype(int)
        obstacle_rows = np.floor(
            (xyz[obstacle, 1] - self._origin_y) / self._resolution
        ).astype(int)
        # Same credit-wipe rule as the primary path: cells just cleared by the
        # raycast must re-earn lethal status.
        self._hits[self._grid == 0] = 0
        self._mark_obstacle_cells(obstacle_rows, obstacle_cols)
        if _as_bool(self.get_parameter("clear_robot_footprint").value):
            self._clear_current_robot_footprint()

    def _mark_obstacle_cells(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Count one hit per cell per cloud; write 100 only once confirmed.

        ``rows``/``cols`` may contain the same cell many times within a single
        cloud, so cells are deduplicated first: one cloud always means exactly
        one hit, keeping obstacle_confirm_hits meaningful in cloud units.
        """
        required = max(1, int(self.get_parameter("obstacle_confirm_hits").value))
        if len(rows) == 0:
            return
        flat = np.unique(
            rows.astype(np.int64) * self._width + cols.astype(np.int64)
        )
        unique_rows = (flat // self._width).astype(np.int64)
        unique_cols = (flat % self._width).astype(np.int64)
        if required <= 1:
            self._grid[unique_rows, unique_cols] = 100
            return
        counts = self._hits[unique_rows, unique_cols].astype(np.int16) + 1
        self._hits[unique_rows, unique_cols] = np.minimum(counts, 127).astype(
            np.int8
        )
        confirmed = counts >= required
        self._grid[unique_rows[confirmed], unique_cols[confirmed]] = 100

    def _clear_current_robot_footprint(self) -> None:
        pose = self._latest_pose
        if pose is None:
            return
        clear_oriented_footprint(
            self._grid,
            self._resolution,
            self._origin_x,
            self._origin_y,
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            yaw_from_quaternion(pose.pose.orientation),
            float(self.get_parameter("footprint_half_length_m").value),
            float(self.get_parameter("footprint_half_width_m").value),
        )

    def _transform_points_to_map(self, xyz: np.ndarray) -> Optional[np.ndarray]:
        pose = self._latest_pose
        if pose is None:
            self.get_logger().warn(
                "pointcloud_in_map=false but no current pose received yet.",
                throttle_duration_sec=5.0,
            )
            return None
        yaw = yaw_from_quaternion(pose.pose.orientation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x = xyz[:, 0]
        y = xyz[:, 1]
        map_x = pose.pose.position.x + cos_yaw * x - sin_yaw * y
        map_y = pose.pose.position.y + sin_yaw * x + cos_yaw * y
        return np.column_stack((map_x, map_y, xyz[:, 2]))

    def _robot_cell(self) -> Optional[tuple[int, int]]:
        pose = self._latest_pose
        if pose is None:
            return None
        col = int(math.floor((pose.pose.position.x - self._origin_x) / self._resolution))
        row = int(math.floor((pose.pose.position.y - self._origin_y) / self._resolution))
        if 0 <= col < self._width and 0 <= row < self._height:
            return (row, col)
        return None

    def _raycast_free(self, robot_cell: tuple[int, int], cols: np.ndarray, rows: np.ndarray) -> None:
        try:
            import cv2
        except ImportError:
            cv2 = None

        robot_row, robot_col = robot_cell
        if cv2 is not None:
            endpoint_values = self._grid[rows, cols].copy()
            for col, row in zip(cols.tolist(), rows.tolist()):
                cv2.line(self._grid, (robot_col, robot_row), (int(col), int(row)), 0, 1)
            # cv2.line includes the endpoint, unlike the NumPy fallback below.
            # Preserve it so a clearing ray never erases a mapped obstacle merely
            # because the endpoint classification is handled by the caller.
            self._grid[rows, cols] = endpoint_values
            return

        for col, row in zip(cols.tolist(), rows.tolist()):
            steps = max(abs(int(col) - robot_col), abs(int(row) - robot_row))
            if steps <= 0:
                continue
            line_cols = np.linspace(robot_col, int(col), steps + 1).round().astype(int)
            line_rows = np.linspace(robot_row, int(row), steps + 1).round().astype(int)
            self._grid[line_rows[:-1], line_cols[:-1]] = 0

    def _publish_map(self) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.info.resolution = self._resolution
        msg.info.width = self._width
        msg.info.height = self._height
        msg.info.origin.position.x = self._origin_x
        msg.info.origin.position.y = self._origin_y
        msg.info.origin.orientation.w = 1.0
        # rclpy Humble rejects a raw numpy array here (element-wise int
        # assertion) and tolist() costs ~2.1 ms per 250k-cell grid; array('b')
        # from the raw bytes round-trips correctly in ~0.2 ms.
        msg.data = array.array("b", self._grid.tobytes())
        self._map_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OfficialOccupancyGridBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


if __name__ == "__main__":
    main()
