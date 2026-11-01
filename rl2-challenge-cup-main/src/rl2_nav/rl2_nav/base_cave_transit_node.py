"""Safe, staged transit from a configured base waypoint to a cave entrance."""

from __future__ import annotations

import json
import math
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import ComputePathToPose, NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


class TransitState(str, Enum):
    IDLE = "IDLE"
    PREVIEW_READY = "PREVIEW_READY"
    GOING_TO_BASE = "GOING_TO_BASE"
    AT_BASE = "AT_BASE"
    GOING_TO_CAVE = "GOING_TO_CAVE"
    AT_CAVE = "AT_CAVE"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def yaw_from_quaternion(q: Quaternion) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class BaseCaveTransitNode(Node):
    """Capture or input two map waypoints and execute base -> cave transit."""

    def __init__(self) -> None:
        super().__init__("base_cave_transit_node")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("current_pose_topic", "/slam/current_pose")
        self.declare_parameter("navigate_action", "/navigate_to_pose")
        self.declare_parameter("compute_path_action", "/compute_path_to_pose")
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("go_to_base_first", True)
        self.declare_parameter("pose_timeout_s", 1.0)
        self.declare_parameter("base_valid", False)
        self.declare_parameter("base_x", 0.0)
        self.declare_parameter("base_y", 0.0)
        self.declare_parameter("base_yaw", 0.0)
        self.declare_parameter("cave_valid", False)
        self.declare_parameter("cave_x", 0.0)
        self.declare_parameter("cave_y", 0.0)
        self.declare_parameter("cave_yaw", 0.0)
        self.declare_parameter("cave_a_valid", False)
        self.declare_parameter("cave_a_x", 0.0)
        self.declare_parameter("cave_a_y", 0.0)
        self.declare_parameter("cave_b_valid", False)
        self.declare_parameter("cave_b_x", 0.0)
        self.declare_parameter("cave_b_y", 0.0)
        self.declare_parameter("cave_entry_yaw", 0.0)
        self.declare_parameter("waypoint_output_path", "")
        self.declare_parameter("costmap_timeout_s", 2.0)
        self.declare_parameter("traversable_cost_max", 98)
        self.declare_parameter("target_search_radius_m", 1.0)
        self.declare_parameter("start_search_radius_m", 1.0)
        self.declare_parameter("max_plan_candidates", 40)
        self.declare_parameter("waypoint_reached_tolerance_m", 0.25)

        self._frame = str(self.get_parameter("map_frame").value)
        self._current_pose: Optional[PoseStamped] = None
        self._pose_received_monotonic = 0.0
        self._captured_base: Optional[tuple[float, float, float]] = None
        self._captured_cave: Optional[tuple[float, float, float]] = None
        self._captured_cave_a: Optional[tuple[float, float]] = None
        self._captured_cave_b: Optional[tuple[float, float]] = None
        self._state = TransitState.IDLE
        self._goal_handle = None
        self._generation = 0
        self._costmap: Optional[OccupancyGrid] = None
        self._costmap_received_monotonic = 0.0
        self._preflight_segments = []
        self._preflight_execute = False
        self._preflight_generation = 0
        self._corrected_base: Optional[tuple[float, float, float]] = None
        self._corrected_cave: Optional[tuple[float, float, float]] = None

        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, "/base_cave_mission/status", status_qos
        )
        self._preview_pub = self.create_publisher(
            PoseStamped, "/base_cave_mission/goal_preview", 10
        )
        self._gate_pub = self.create_publisher(
            PoseArray, "/base_cave_mission/cave_gate", status_qos
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/base_cave_mission/markers", status_qos
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("current_pose_topic").value),
            self._on_pose,
            10,
        )
        costmap_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("global_costmap_topic").value),
            self._on_costmap,
            costmap_qos,
        )
        self._nav = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter("navigate_action").value),
        )
        self._planner = ActionClient(
            self,
            ComputePathToPose,
            str(self.get_parameter("compute_path_action").value),
        )
        self.create_service(Trigger, "~/capture_base", self._capture_base)
        self.create_service(Trigger, "~/capture_cave", self._capture_cave)
        self.create_service(Trigger, "~/capture_cave_a", self._capture_cave_a_endpoint)
        self.create_service(Trigger, "~/capture_cave_b", self._capture_cave_b_endpoint)
        self.create_service(Trigger, "~/show_waypoints", self._show_waypoints)
        self.create_service(Trigger, "~/save_waypoints", self._save_waypoints)
        self.create_service(Trigger, "~/start", self._start)
        self.create_service(Trigger, "~/abort", self._abort)
        self._publish_status("ready")
        self.get_logger().info(
            "Base/cave transit ready; dry_run=%s. Capture or configure both "
            "waypoints before calling ~/start."
            % bool(self.get_parameter("dry_run").value)
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        self._current_pose = msg
        self._pose_received_monotonic = time.monotonic()

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._costmap = msg
        self._costmap_received_monotonic = time.monotonic()

    def _fresh_costmap(self) -> Optional[OccupancyGrid]:
        if self._costmap is None:
            return None
        if time.monotonic() - self._costmap_received_monotonic > float(
            self.get_parameter("costmap_timeout_s").value
        ):
            return None
        if self._costmap.header.frame_id not in ("", self._frame):
            return None
        return self._costmap

    @staticmethod
    def _cell_for_xy(grid: OccupancyGrid, x: float, y: float):
        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return None
        col = math.floor((x - float(grid.info.origin.position.x)) / resolution)
        row = math.floor((y - float(grid.info.origin.position.y)) / resolution)
        if not (0 <= col < grid.info.width and 0 <= row < grid.info.height):
            return None
        return int(col), int(row)

    @staticmethod
    def _xy_for_cell(grid: OccupancyGrid, col: int, row: int):
        resolution = float(grid.info.resolution)
        return (
            float(grid.info.origin.position.x) + (col + 0.5) * resolution,
            float(grid.info.origin.position.y) + (row + 0.5) * resolution,
        )

    def _cell_traversable(self, value: int, *, allow_unknown: bool = False) -> bool:
        if value < 0:
            return allow_unknown
        return value <= int(self.get_parameter("traversable_cost_max").value)

    def _snap_candidates(
        self,
        grid: OccupancyGrid,
        waypoint: tuple[float, float, float],
        radius_m: float,
    ) -> list[tuple[float, float, float]]:
        center = self._cell_for_xy(grid, waypoint[0], waypoint[1])
        if center is None:
            return []
        resolution = float(grid.info.resolution)
        radius_cells = max(0, math.ceil(radius_m / resolution))
        cells = []
        for row in range(max(0, center[1] - radius_cells), min(grid.info.height, center[1] + radius_cells + 1)):
            for col in range(max(0, center[0] - radius_cells), min(grid.info.width, center[0] + radius_cells + 1)):
                value = int(grid.data[row * grid.info.width + col])
                if not self._cell_traversable(value):
                    continue
                x, y = self._xy_for_cell(grid, col, row)
                distance = math.hypot(x - waypoint[0], y - waypoint[1])
                if distance <= radius_m + resolution:
                    cells.append((distance, value, x, y))
        cells.sort(key=lambda item: (item[0], item[1]))
        limit = max(1, int(self.get_parameter("max_plan_candidates").value))
        return [(x, y, waypoint[2]) for _, _, x, y in cells[:limit]]

    def _fresh_pose(self) -> Optional[PoseStamped]:
        if self._current_pose is None:
            return None
        age = time.monotonic() - self._pose_received_monotonic
        if age > float(self.get_parameter("pose_timeout_s").value):
            return None
        if self._current_pose.header.frame_id not in ("", self._frame):
            return None
        return self._current_pose

    @staticmethod
    def _xyz_yaw(msg: PoseStamped) -> tuple[float, float, float]:
        return (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            yaw_from_quaternion(msg.pose.orientation),
        )

    @staticmethod
    def _format_waypoint(name: str, value: tuple[float, float, float]) -> str:
        return "%s: x=%.6f y=%.6f yaw=%.6f" % ((name,) + value)

    def _capture_base(self, _request, response):
        pose = self._fresh_pose()
        if pose is None:
            response.success = False
            response.message = "fresh map-frame pose unavailable"
            return response
        self._captured_base = self._xyz_yaw(pose)
        response.success = True
        response.message = self._format_waypoint("base", self._captured_base)
        self._publish_status("base_captured")
        return response

    def _capture_cave(self, _request, response):
        pose = self._fresh_pose()
        if pose is None:
            response.success = False
            response.message = "fresh map-frame pose unavailable"
            return response
        self._captured_cave = self._xyz_yaw(pose)
        response.success = True
        response.message = self._format_waypoint("cave", self._captured_cave)
        self._publish_status("cave_captured")
        return response

    def _capture_cave_a_endpoint(self, _request, response):
        return self._capture_gate_endpoint("a", response)

    def _capture_cave_b_endpoint(self, _request, response):
        return self._capture_gate_endpoint("b", response)

    def _capture_gate_endpoint(self, label: str, response):
        pose = self._fresh_pose()
        if pose is None:
            response.success = False
            response.message = "fresh map-frame pose unavailable"
            return response
        point = (float(pose.pose.position.x), float(pose.pose.position.y))
        if label == "a":
            self._captured_cave_a = point
        else:
            self._captured_cave_b = point
        response.success = True
        response.message = "cave_%s: x=%.6f y=%.6f" % ((label,) + point)
        self._publish_gate()
        self._publish_status(f"cave_{label}_captured")
        return response

    def _gate_endpoint(self, label: str) -> Optional[tuple[float, float]]:
        captured = self._captured_cave_a if label == "a" else self._captured_cave_b
        if captured is not None:
            return captured
        if not bool(self.get_parameter(f"cave_{label}_valid").value):
            return None
        value = (
            float(self.get_parameter(f"cave_{label}_x").value),
            float(self.get_parameter(f"cave_{label}_y").value),
        )
        return value if all(math.isfinite(v) for v in value) else None

    def _cave_midpoint(self) -> Optional[tuple[float, float, float]]:
        a, b = self._gate_endpoint("a"), self._gate_endpoint("b")
        if a is not None and b is not None and math.dist(a, b) >= 0.20:
            return (
                0.5 * (a[0] + b[0]),
                0.5 * (a[1] + b[1]),
                float(self.get_parameter("cave_entry_yaw").value),
            )
        # Backward-compatible single cave waypoint.
        if self._captured_cave is not None:
            return self._captured_cave
        if bool(self.get_parameter("cave_valid").value):
            value = tuple(
                float(self.get_parameter(f"cave_{field}").value)
                for field in ("x", "y", "yaw")
            )
            return value if all(math.isfinite(v) for v in value) else None
        return None

    def _publish_gate(self) -> None:
        a, b = self._gate_endpoint("a"), self._gate_endpoint("b")
        if a is None or b is None or math.dist(a, b) < 0.20:
            return
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        for point in (a, b):
            pose = Pose()
            pose.position.x, pose.position.y = point
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._gate_pub.publish(msg)

    def _publish_markers(self) -> None:
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        def marker(marker_id, name, waypoint, color, marker_type=Marker.SPHERE):
            msg = Marker()
            msg.header.frame_id = self._frame
            msg.header.stamp = now
            msg.ns = "base_cave_waypoints"
            msg.id = marker_id
            msg.type = marker_type
            msg.action = Marker.ADD
            msg.pose.position.x = waypoint[0]
            msg.pose.position.y = waypoint[1]
            msg.pose.position.z = 0.12
            msg.pose.orientation = quaternion_from_yaw(
                waypoint[2] if len(waypoint) > 2 else 0.0
            )
            msg.scale.x = 0.28
            msg.scale.y = 0.28
            msg.scale.z = 0.28
            msg.color.r, msg.color.g, msg.color.b, msg.color.a = (*color, 0.95)
            markers.markers.append(msg)

            text = Marker()
            text.header = msg.header
            text.ns = "base_cave_labels"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = waypoint[0]
            text.pose.position.y = waypoint[1]
            text.pose.position.z = 0.55
            text.pose.orientation.w = 1.0
            text.scale.z = 0.24
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = "%s\n(%.2f, %.2f)" % (name, waypoint[0], waypoint[1])
            markers.markers.append(text)

        base = self._configured_waypoint("base")
        a, b = self._gate_endpoint("a"), self._gate_endpoint("b")
        cave = self._configured_waypoint("cave")
        if base is not None:
            marker(0, "BASE", base, (0.1, 1.0, 0.1))
        if a is not None:
            marker(1, "CAVE_A", (*a, 0.0), (0.1, 0.5, 1.0))
        if b is not None:
            marker(2, "CAVE_B", (*b, 0.0), (0.1, 0.5, 1.0))
        if cave is not None:
            marker(3, "CAVE_MID", cave, (1.0, 0.55, 0.0))
        if a is not None and b is not None:
            line = Marker()
            line.header.frame_id = self._frame
            line.header.stamp = now
            line.ns = "base_cave_gate"
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.08
            line.color.r, line.color.g, line.color.b, line.color.a = (0.0, 0.8, 1.0, 1.0)
            line.points = [Point(x=a[0], y=a[1], z=0.08), Point(x=b[0], y=b[1], z=0.08)]
            markers.markers.append(line)
        if self._corrected_base is not None:
            marker(10, "BASE_PLAN", self._corrected_base, (1.0, 0.1, 1.0))
        if self._corrected_cave is not None:
            marker(11, "CAVE_PLAN", self._corrected_cave, (1.0, 0.1, 1.0))
        self._marker_pub.publish(markers)

    def _configured_waypoint(self, name: str) -> Optional[tuple[float, float, float]]:
        if name == "cave":
            return self._cave_midpoint()
        captured = self._captured_base
        if captured is not None:
            return captured
        if not bool(self.get_parameter(f"{name}_valid").value):
            return None
        value = tuple(
            float(self.get_parameter(f"{name}_{field}").value)
            for field in ("x", "y", "yaw")
        )
        return value if all(math.isfinite(v) for v in value) else None

    def _show_waypoints(self, _request, response):
        base = self._configured_waypoint("base")
        cave = self._configured_waypoint("cave")
        self._publish_gate()
        response.success = base is not None and cave is not None
        response.message = json.dumps(
            {
                "frame_id": self._frame,
                "base": base,
                "cave_a": self._gate_endpoint("a"),
                "cave_b": self._gate_endpoint("b"),
                "cave_midpoint": cave,
            },
            separators=(",", ":"),
        )
        return response

    def _save_waypoints(self, _request, response):
        output = str(self.get_parameter("waypoint_output_path").value).strip()
        if not output:
            response.success = False
            response.message = "waypoint_output_path parameter is empty"
            return response
        base = self._configured_waypoint("base")
        a, b = self._gate_endpoint("a"), self._gate_endpoint("b")
        if base is None or a is None or b is None:
            response.success = False
            response.message = "base and cave A/B must all be captured/configured"
            return response
        if math.dist(a, b) < 0.20:
            response.success = False
            response.message = "cave A/B are too close"
            return response
        yaw = float(self.get_parameter("cave_entry_yaw").value)
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "base_cave_transit_node:\n"
            "  ros__parameters:\n"
            f"    map_frame: {self._frame}\n"
            "    base_valid: true\n"
            f"    base_x: {base[0]:.12g}\n"
            f"    base_y: {base[1]:.12g}\n"
            f"    base_yaw: {base[2]:.12g}\n"
            "    cave_a_valid: true\n"
            f"    cave_a_x: {a[0]:.12g}\n"
            f"    cave_a_y: {a[1]:.12g}\n"
            "    cave_b_valid: true\n"
            f"    cave_b_x: {b[0]:.12g}\n"
            f"    cave_b_y: {b[1]:.12g}\n"
            f"    cave_entry_yaw: {yaw:.12g}\n"
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
        response.success = True
        response.message = f"saved waypoint profile: {path}"
        self._publish_status("waypoints_saved")
        return response

    def _start(self, _request, response):
        base = self._configured_waypoint("base")
        cave = self._configured_waypoint("cave")
        if base is None or cave is None:
            response.success = False
            response.message = "base and cave waypoints must both be configured"
            return response
        current_pose = self._fresh_pose()
        if current_pose is None:
            response.success = False
            response.message = "fresh map-frame pose unavailable"
            return response
        grid = self._fresh_costmap()
        if grid is None:
            response.success = False
            response.message = "fresh global costmap unavailable"
            return response
        if self._state in (TransitState.GOING_TO_BASE, TransitState.GOING_TO_CAVE):
            response.success = False
            response.message = "transit already active"
            return response

        current = self._xyz_yaw(current_pose)
        start_cell = self._cell_for_xy(grid, current[0], current[1])
        start_value = None if start_cell is None else int(
            grid.data[start_cell[1] * grid.info.width + start_cell[0]]
        )
        if start_value is None or not self._cell_traversable(start_value):
            nearest = self._snap_candidates(
                grid,
                current,
                float(self.get_parameter("start_search_radius_m").value),
            )
            self._state = TransitState.FAILED
            detail = "none"
            if nearest:
                distance = math.hypot(nearest[0][0] - current[0], nearest[0][1] - current[1])
                detail = "x=%.3f,y=%.3f,distance=%.3fm" % (
                    nearest[0][0], nearest[0][1], distance
                )
            reason = f"start_not_traversable:cost={start_value},nearest={detail}"
            self._publish_status(reason)
            response.success = False
            response.message = reason
            return response

        self._generation += 1
        self._begin_preflight(
            current,
            base,
            cave,
            execute=not bool(self.get_parameter("dry_run").value),
            generation=self._generation,
        )
        response.success = True
        response.message = "preflight planning queued; watch /base_cave_mission/status"
        return response

    def _begin_preflight(self, current, base, cave, *, execute: bool, generation: int):
        grid = self._fresh_costmap()
        if grid is None:
            self._preflight_failed("costmap_lost", generation)
            return
        radius = float(self.get_parameter("target_search_radius_m").value)
        segments = []
        start = current
        if bool(self.get_parameter("go_to_base_first").value):
            if math.hypot(current[0] - base[0], current[1] - base[1]) <= float(
                self.get_parameter("waypoint_reached_tolerance_m").value
            ):
                self._corrected_base = current
            else:
                segments.append(("base", start, self._snap_candidates(grid, base, radius)))
                start = base
        segments.append(("cave", start, self._snap_candidates(grid, cave, radius)))
        if any(not candidates for _, _, candidates in segments):
            missing = next(label for label, _, candidates in segments if not candidates)
            self._preflight_failed(f"no_traversable_candidate_{missing}", generation)
            return
        self._preflight_segments = segments
        self._preflight_execute = execute
        self._preflight_generation = generation
        if self._corrected_base is None or math.hypot(
            current[0] - base[0], current[1] - base[1]
        ) > float(self.get_parameter("waypoint_reached_tolerance_m").value):
            self._corrected_base = None
        self._corrected_cave = None
        self._publish_status("preflight_started")
        self._plan_next_candidate()

    def _plan_next_candidate(self):
        if self._preflight_generation != self._generation:
            return
        if not self._preflight_segments:
            self._state = TransitState.PREVIEW_READY
            self._publish_status("preflight_passed")
            if not self._preflight_execute:
                return
            if bool(self.get_parameter("go_to_base_first").value):
                self._send_goal("base", self._corrected_base, self._generation)
            else:
                self._state = TransitState.AT_BASE
                self._send_goal("cave", self._corrected_cave, self._generation)
            return
        label, start, candidates = self._preflight_segments[0]
        if not candidates:
            self._preflight_failed(f"no_path_to_{label}", self._generation)
            return
        if not self._planner.wait_for_server(timeout_sec=1.0):
            self._preflight_failed("compute_path_to_pose_unavailable", self._generation)
            return
        candidate = candidates.pop(0)
        goal = ComputePathToPose.Goal()
        goal.start = self._make_pose(start)
        goal.goal = self._make_pose(candidate)
        goal.use_start = True
        future = self._planner.send_goal_async(goal)
        future.add_done_callback(
            lambda done, lab=label, cand=candidate, gen=self._generation:
            self._on_plan_goal_response(done, lab, cand, gen)
        )

    def _on_plan_goal_response(self, future, label, candidate, generation):
        if generation != self._generation:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self._preflight_failed(f"planner_request_failed:{exc}", generation)
            return
        if not handle.accepted:
            self._plan_next_candidate()
            return
        handle.get_result_async().add_done_callback(
            lambda done, lab=label, cand=candidate, gen=generation:
            self._on_plan_result(done, lab, cand, gen)
        )

    def _on_plan_result(self, future, label, candidate, generation):
        if generation != self._generation:
            return
        try:
            wrapped = future.result()
            valid = (
                int(wrapped.status) == GoalStatus.STATUS_SUCCEEDED
                and len(wrapped.result.path.poses) >= 2
            )
        except Exception:
            valid = False
        if not valid:
            self._plan_next_candidate()
            return
        if label == "base":
            self._corrected_base = candidate
        else:
            self._corrected_cave = candidate
        self._publish_preview(label, candidate)
        self._preflight_segments.pop(0)
        if self._preflight_segments:
            next_label, _, next_candidates = self._preflight_segments[0]
            corrected_start = candidate
            self._preflight_segments[0] = (next_label, corrected_start, next_candidates)
        self._publish_status(f"preflight_path_ok_{label}")
        self._plan_next_candidate()

    def _preflight_failed(self, reason: str, generation: int):
        if generation != self._generation:
            return
        self._preflight_segments = []
        self._state = TransitState.FAILED
        self._publish_status(f"preflight_failed:{reason}")

    def _make_pose(self, waypoint: tuple[float, float, float]) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        msg.pose.position.x = waypoint[0]
        msg.pose.position.y = waypoint[1]
        msg.pose.orientation = quaternion_from_yaw(waypoint[2])
        return msg

    def _publish_preview(self, label: str, waypoint: tuple[float, float, float]) -> None:
        self._preview_pub.publish(self._make_pose(waypoint))
        self._publish_status(f"preview_{label}")

    def _send_goal(
        self, label: str, waypoint: tuple[float, float, float], generation: int
    ) -> None:
        if not self._nav.wait_for_server(timeout_sec=1.0):
            self._state = TransitState.FAILED
            self._publish_status("navigate_to_pose unavailable")
            return
        self._state = (
            TransitState.GOING_TO_BASE
            if label == "base"
            else TransitState.GOING_TO_CAVE
        )
        pose = self._make_pose(waypoint)
        self._preview_pub.publish(pose)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self._nav.send_goal_async(goal)
        future.add_done_callback(
            lambda done, lab=label, gen=generation: self._on_goal_response(
                done, lab, gen
            )
        )
        self._publish_status(f"goal_sent_{label}")

    def _on_goal_response(self, future, label: str, generation: int) -> None:
        if generation != self._generation:
            return
        try:
            handle = future.result()
        except Exception as exc:  # pragma: no cover - ROS transport failure
            self._state = TransitState.FAILED
            self._publish_status(f"goal_send_failed:{exc}")
            return
        if not handle.accepted:
            self._state = TransitState.FAILED
            self._publish_status(f"goal_rejected_{label}")
            return
        self._goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda done, lab=label, gen=generation: self._on_goal_result(
                done, lab, gen
            )
        )

    def _on_goal_result(self, future, label: str, generation: int) -> None:
        if generation != self._generation:
            return
        self._goal_handle = None
        try:
            status = int(future.result().status)
        except Exception as exc:  # pragma: no cover
            self._state = TransitState.FAILED
            self._publish_status(f"goal_result_failed:{exc}")
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._state = TransitState.FAILED
            self._publish_status(f"goal_failed_{label}:status={status}")
            return
        if label == "base":
            self._state = TransitState.AT_BASE
            self._publish_status("arrived_base")
            cave = self._corrected_cave or self._configured_waypoint("cave")
            if cave is None:
                self._state = TransitState.FAILED
                self._publish_status("cave_waypoint_lost")
                return
            self._send_goal("cave", cave, generation)
        else:
            self._state = TransitState.AT_CAVE
            self._publish_status("arrived_cave")

    def _abort(self, _request, response):
        self._generation += 1
        handle = self._goal_handle
        self._goal_handle = None
        if handle is not None:
            handle.cancel_goal_async()
        self._state = TransitState.ABORTED
        self._publish_status("operator_abort")
        response.success = True
        response.message = "transit aborted; disable motion gates as well"
        return response

    def _publish_status(self, reason: str) -> None:
        base = self._configured_waypoint("base")
        cave = self._configured_waypoint("cave")
        payload = {
            "state": self._state.value,
            "reason": reason,
            "dry_run": bool(self.get_parameter("dry_run").value),
            "frame_id": self._frame,
            "base": base,
            "cave_a": self._gate_endpoint("a"),
            "cave_b": self._gate_endpoint("b"),
            "cave_midpoint": cave,
            "corrected_base": self._corrected_base,
            "corrected_cave": self._corrected_cave,
            "pose_fresh": self._fresh_pose() is not None,
        }
        self._status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self._publish_markers()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BaseCaveTransitNode()
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
