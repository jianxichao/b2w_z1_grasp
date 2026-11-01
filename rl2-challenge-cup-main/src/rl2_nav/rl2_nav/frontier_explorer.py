"""Frontier extraction and ranking for occupancy-grid exploration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Iterable, List, Sequence, Tuple

FREE_MAX = 49
CAUTION_MAX = 65
HARD_OCCUPIED_MIN = 90
UNKNOWN = -1
UNKNOWN_UINT = 255


@dataclass(frozen=True)
class FrontierCandidate:
    x: float
    y: float
    size: int
    distance: float
    score: float


@dataclass(frozen=True)
class ApproachCandidate:
    x: float
    y: float
    yaw: float
    score: float
    standoff_m: float
    angle_rad: float


@dataclass
class _RegionHistoryEntry:
    size: int
    stall_count: int
    last_seen_call: int


@dataclass
class FrontierExplorer:
    min_frontier_size: int = 5
    max_frontier_candidates: int = 8
    cluster_connect_distance: int = 2
    free_threshold: int = CAUTION_MAX
    occupied_threshold: int = HARD_OCCUPIED_MIN
    safety_radius_m: float = 0.55
    distance_weight: float = 0.35
    size_weight: float = 0.08
    # A long wall or the outside rim of a partially observed map can contain
    # hundreds of frontier cells.  A linear, unbounded size reward made those
    # artifacts dominate every nearby reachable opening.  Keep rewarding a
    # useful-sized boundary, but saturate the reward once it is large enough.
    size_reward_cap: float = 3.0
    min_frontier_distance_m: float = 1.5
    min_frontier_size_nearby: int = 3
    nearby_radius_m: float = 2.5
    path_clearance_m: float = 0.28
    large_cluster_size_multiplier: float = 4.0
    cluster_split_sectors: int = 8
    cluster_split_max_anchors: int = 4
    info_gain_weight: float = 2.0
    info_gain_cap_cells: int = 3000
    hysteresis_bonus: float = 0.5
    hysteresis_radius_m: float = 1.5
    region_stall_bucket_m: float = 1.0
    region_stall_min_calls: int = 3
    region_stall_discount: float = 0.2
    region_stall_zero_calls: int = 8
    region_history_stale_calls: int = 40
    region_history_max_entries: int = 2000
    region_stall_shrink_ratio: float = 0.95
    _region_history: dict = field(default_factory=dict, repr=False, compare=False)
    _region_call_counter: int = field(default=0, repr=False, compare=False)

    def find_frontiers(
        self,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        data: Sequence[int],
        robot_x: float,
        robot_y: float,
        rejected_goals: Iterable[Tuple[float, float]] | None = None,
        rejected_radius_m: float = 0.6,
        previous_goal_xy: Tuple[float, float] | None = None,
    ) -> List[FrontierCandidate]:
        if not data or width <= 0 or height <= 0 or resolution <= 0:
            return []

        rejected = list(rejected_goals or [])
        robot_col, robot_row = world_to_cell(robot_x, robot_y, resolution, origin_x, origin_y)
        robot_cell = self._nearest_free_cell(data, width, height, robot_col, robot_row)
        if robot_cell is None:
            return []

        occupied_dist = self._distance_transform(data, width, height, self._is_occupied)
        unknown_dist = self._distance_transform(data, width, height, self._is_unknown)
        region_id, region_size, region_bbox = self._label_unknown_regions(
            data, width, height, self.info_gain_cap_cells
        )
        region_discount = self._update_region_stall_history(
            region_size, region_bbox, resolution, origin_x, origin_y
        )

        clearance_cells = max(0, int(round(self.path_clearance_m / resolution)))
        narrow_cells = (
            self._compute_narrow_cells(width, height, occupied_dist, clearance_cells)
            if clearance_cells > 0
            else set()
        )
        reachable_cells = self._reachable_free_cells(data, width, height, robot_cell, narrow_cells)
        if len(reachable_cells) <= 1:
            reachable_cells = self._reachable_free_cells(data, width, height, robot_cell, set())
        if not reachable_cells:
            return []

        frontier_cells = self._detect_frontier_cells(data, width, height)
        clusters = self._cluster_cells(frontier_cells)
        if not clusters:
            return []

        common_kwargs = dict(
            clusters=clusters,
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            occupied_dist=occupied_dist,
            unknown_dist=unknown_dist,
            region_id=region_id,
            region_size=region_size,
            region_discount=region_discount,
            reachable_cells=reachable_cells,
            robot_cell=robot_cell,
            robot_x=robot_x,
            robot_y=robot_y,
            previous_goal_xy=previous_goal_xy,
        )

        candidates = self._rank_clusters(
            **common_kwargs,
            rejected=rejected,
            rejected_radius_m=rejected_radius_m,
            min_frontier_distance_m=self.min_frontier_distance_m,
            min_size_override=None,
        )
        if candidates:
            return candidates

        candidates = self._rank_clusters(
            **common_kwargs,
            rejected=rejected,
            rejected_radius_m=rejected_radius_m,
            min_frontier_distance_m=0.0,
            min_size_override=1,
        )
        if candidates:
            return candidates

        # Do not erase the failed-goal blacklist as a final fallback.  Reusing
        # already failed wall holes/isolated pockets creates an A -> B -> A
        # loop.  Returning no candidate lets the supervisor perform a visual
        # scan and wait for genuinely new map information instead.
        return []

    def filter_reachable_on_costmap(
        self,
        candidates: Sequence[FrontierCandidate],
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        data: Sequence[int],
        robot_x: float,
        robot_y: float,
        traversable_cost_max: int = 98,
        goal_tolerance_m: float = 0.50,
        goal_clearance_m: float = 0.0,
        goal_clearance_cost_max: int = 65,
    ) -> List[FrontierCandidate]:
        """Keep goals connected to the robot on Nav2's inflated costmap.

        Frontier extraction uses the mapping grid because unknown cells carry
        information gain.  Nav2, however, plans on a different grid containing
        live obstacles and inflation.  This second connectivity check prevents
        goals that look reachable on the mapping grid but are disconnected on
        the grid the planner actually uses.
        """
        if not candidates or not data or width <= 0 or height <= 0 or resolution <= 0:
            return []

        def traversable(value: int) -> bool:
            return 0 <= int(value) <= traversable_cost_max

        robot_col, robot_row = world_to_cell(
            robot_x, robot_y, resolution, origin_x, origin_y
        )
        # SmacPlanner starts from the robot's exact grid cell.  Substituting a
        # nearby free cell here would produce a false positive: our filter
        # would accept a candidate while the planner reports "Starting point
        # in lethal space".
        if not (0 <= robot_col < width and 0 <= robot_row < height):
            return []
        if not traversable(data[robot_row * width + robot_col]):
            return []
        start = (robot_col, robot_row)

        reachable = self._reachable_cells_matching(
            data, width, height, start, traversable
        )
        tolerance_cells = max(0, int(math.ceil(goal_tolerance_m / resolution)))
        clearance_cells = max(0, int(math.ceil(goal_clearance_m / resolution)))

        def has_goal_clearance(col: int, row: int) -> bool:
            """Reject goal cells whose robot-sized neighborhood hits inflation.

            Unknown cells are intentionally ignored here: a frontier goal is
            expected to border unknown space.  Known cells inside the clearance
            disk must nevertheless stay below the configured caution cost.
            """
            for check_row in range(
                max(0, row - clearance_cells), min(height, row + clearance_cells + 1)
            ):
                for check_col in range(
                    max(0, col - clearance_cells), min(width, col + clearance_cells + 1)
                ):
                    if math.hypot(check_col - col, check_row - row) > clearance_cells:
                        continue
                    value = int(data[check_row * width + check_col])
                    if value >= 0 and value > goal_clearance_cost_max:
                        return False
            return True

        filtered: List[FrontierCandidate] = []
        for candidate in candidates:
            goal_col, goal_row = world_to_cell(
                candidate.x, candidate.y, resolution, origin_x, origin_y
            )
            best_cell: Tuple[int, int] | None = None
            best_distance_cells = float("inf")
            for row in range(goal_row - tolerance_cells, goal_row + tolerance_cells + 1):
                for col in range(goal_col - tolerance_cells, goal_col + tolerance_cells + 1):
                    distance_cells = math.hypot(col - goal_col, row - goal_row)
                    if distance_cells > tolerance_cells:
                        continue
                    if (col, row) in reachable and has_goal_clearance(col, row):
                        if distance_cells < best_distance_cells:
                            best_distance_cells = distance_cells
                            best_cell = (col, row)
            if best_cell is None:
                continue
            if best_cell == (goal_col, goal_row):
                filtered.append(candidate)
                continue

            snapped_x, snapped_y = cell_to_world(
                best_cell[0], best_cell[1], resolution, origin_x, origin_y
            )
            snapped_distance = math.hypot(snapped_x - robot_x, snapped_y - robot_y)
            # Preserve the information-gain ordering while accounting for the
            # small change in travel distance caused by snapping.
            snapped_score = candidate.score - self.distance_weight * (
                snapped_distance - candidate.distance
            )
            filtered.append(
                FrontierCandidate(
                    x=snapped_x,
                    y=snapped_y,
                    size=candidate.size,
                    distance=snapped_distance,
                    score=snapped_score,
                )
            )
        return filtered

    def _nearest_cell_matching(
        self,
        data: Sequence[int],
        width: int,
        height: int,
        col: int,
        row: int,
        predicate,
        max_radius: int = 15,
    ) -> Tuple[int, int] | None:
        if 0 <= col < width and 0 <= row < height and predicate(data[row * width + col]):
            return (col, row)
        for radius in range(1, max_radius + 1):
            for check_row in range(max(0, row - radius), min(height, row + radius + 1)):
                for check_col in range(max(0, col - radius), min(width, col + radius + 1)):
                    if abs(check_col - col) != radius and abs(check_row - row) != radius:
                        continue
                    if predicate(data[check_row * width + check_col]):
                        return (check_col, check_row)
        return None

    @staticmethod
    def _reachable_cells_matching(
        data: Sequence[int],
        width: int,
        height: int,
        start: Tuple[int, int],
        predicate,
    ) -> set[Tuple[int, int]]:
        visited = {start}
        queue = deque([start])
        while queue:
            col, row = queue.popleft()
            for dcol, drow in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_col, next_row = col + dcol, row + drow
                if not (0 <= next_col < width and 0 <= next_row < height):
                    continue
                neighbor = (next_col, next_row)
                if neighbor in visited or not predicate(data[next_row * width + next_col]):
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return visited

    def _rank_clusters(
        self,
        clusters: List[List[Tuple[int, int]]],
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        occupied_dist: List[int],
        unknown_dist: List[int],
        region_id: List[int],
        region_size: List[int],
        region_discount: List[float],
        reachable_cells: set[Tuple[int, int]],
        robot_cell: Tuple[int, int],
        robot_x: float,
        robot_y: float,
        rejected: Sequence[Tuple[float, float]],
        rejected_radius_m: float,
        min_frontier_distance_m: float,
        min_size_override: int | None,
        previous_goal_xy: Tuple[float, float] | None,
    ) -> List[FrontierCandidate]:
        candidates: List[FrontierCandidate] = []
        large_cluster_threshold = self.min_frontier_size * self.large_cluster_size_multiplier
        for cluster in clusters:
            cluster_size = len(cluster)
            if cluster_size >= large_cluster_threshold:
                anchors = self._split_cluster_by_sector(cluster, robot_cell)
            else:
                anchors = [
                    min(
                        cluster,
                        key=lambda cell: (cell[0] - robot_cell[0]) ** 2
                        + (cell[1] - robot_cell[1]) ** 2,
                    )
                ]

            for anchor_cell in anchors:
                candidate = self._build_candidate(
                    anchor_cell=anchor_cell,
                    cluster_size=cluster_size,
                    width=width,
                    height=height,
                    resolution=resolution,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    occupied_dist=occupied_dist,
                    unknown_dist=unknown_dist,
                    region_id=region_id,
                    region_size=region_size,
                    region_discount=region_discount,
                    reachable_cells=reachable_cells,
                    robot_cell=robot_cell,
                    robot_x=robot_x,
                    robot_y=robot_y,
                    rejected=rejected,
                    rejected_radius_m=rejected_radius_m,
                    min_frontier_distance_m=min_frontier_distance_m,
                    min_size_override=min_size_override,
                    previous_goal_xy=previous_goal_xy,
                )
                if candidate is not None:
                    candidates.append(candidate)

        candidates.sort(key=lambda c: (-c.score, c.distance))
        return candidates[: self.max_frontier_candidates]

    def _build_candidate(
        self,
        anchor_cell: Tuple[int, int],
        cluster_size: int,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        occupied_dist: List[int],
        unknown_dist: List[int],
        region_id: List[int],
        region_size: List[int],
        region_discount: List[float],
        reachable_cells: set[Tuple[int, int]],
        robot_cell: Tuple[int, int],
        robot_x: float,
        robot_y: float,
        rejected: Sequence[Tuple[float, float]],
        rejected_radius_m: float,
        min_frontier_distance_m: float,
        min_size_override: int | None,
        previous_goal_xy: Tuple[float, float] | None,
    ) -> FrontierCandidate | None:
        robot_distance_m = math.hypot(
            anchor_cell[0] - robot_cell[0], anchor_cell[1] - robot_cell[1]
        ) * resolution
        if robot_distance_m < min_frontier_distance_m:
            return None

        if min_size_override is not None:
            min_size = min_size_override
        else:
            min_size = self.min_frontier_size
            if robot_distance_m <= self.nearby_radius_m:
                min_size = self.min_frontier_size_nearby
        if cluster_size < min_size:
            return None

        goal_cell = self._select_goal_cell(
            anchor_cell=anchor_cell,
            width=width,
            height=height,
            resolution=resolution,
            occupied_dist=occupied_dist,
            unknown_dist=unknown_dist,
            reachable_cells=reachable_cells,
            robot_cell=robot_cell,
        )
        if goal_cell is None:
            return None

        col, row = goal_cell
        wx, wy = cell_to_world(col, row, resolution, origin_x, origin_y)
        if self._near_rejected(wx, wy, rejected, rejected_radius_m):
            return None

        distance = math.hypot(wx - robot_x, wy - robot_y)
        info_gain = self._nearby_region_size(
            anchor_cell, width, height, region_id, region_size, region_discount
        )
        size_reward = self._frontier_size_reward(cluster_size)
        score = (
            -self.distance_weight * distance
            + size_reward
            + self.info_gain_weight * (info_gain / max(1, self.info_gain_cap_cells))
        )
        if previous_goal_xy is not None and math.hypot(
            wx - previous_goal_xy[0], wy - previous_goal_xy[1]
        ) <= self.hysteresis_radius_m:
            # Hysteresis 的目的是让探索保持同一方向推进；它不应把选择器
            # 拽回刚刚失败的区域。在黑名单半径 2 倍范围内的候选不加分，
            # 否则 hysteresis(+0.5@1.5m) 与黑名单(0.65m) 会在死角反复拉锯。
            if not self._near_rejected(wx, wy, rejected, rejected_radius_m * 2.0):
                score += self.hysteresis_bonus

        return FrontierCandidate(x=wx, y=wy, size=cluster_size, distance=distance, score=score)

    def _frontier_size_reward(self, cluster_size: int) -> float:
        return min(self.size_reward_cap, self.size_weight * max(0, cluster_size))

    def _split_cluster_by_sector(
        self,
        cluster: List[Tuple[int, int]],
        robot_cell: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        sectors = max(1, self.cluster_split_sectors)
        best_per_sector: dict[int, Tuple[Tuple[int, int], float]] = {}
        for cell in cluster:
            dx = cell[0] - robot_cell[0]
            dy = cell[1] - robot_cell[1]
            dist_sq = dx * dx + dy * dy
            angle = math.atan2(dy, dx)
            sector = int((angle + math.pi) / (2 * math.pi) * sectors) % sectors
            current = best_per_sector.get(sector)
            if current is None or dist_sq < current[1]:
                best_per_sector[sector] = (cell, dist_sq)

        anchors = [cell for cell, _ in best_per_sector.values()]
        anchors.sort(
            key=lambda cell: (cell[0] - robot_cell[0]) ** 2
            + (cell[1] - robot_cell[1]) ** 2
        )
        return anchors[: max(1, self.cluster_split_max_anchors)]

    def _nearby_region_size(
        self,
        anchor_cell: Tuple[int, int],
        width: int,
        height: int,
        region_id: List[int],
        region_size: List[int],
        region_discount: List[float],
    ) -> float:
        col, row = anchor_cell
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                ncol, nrow = col + dcol, row + drow
                if not (0 <= ncol < width and 0 <= nrow < height):
                    continue
                idx = nrow * width + ncol
                rid = region_id[idx]
                if rid != -1:
                    return region_size[rid] * region_discount[rid]
        return 0.0

    def _detect_frontier_cells(
        self, data: Sequence[int], width: int, height: int
    ) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                idx = row * width + col
                if not self._is_free(data[idx]):
                    continue
                if any(
                    self._is_unknown(data[(row + drow) * width + (col + dcol)])
                    for drow, dcol in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ]
                ):
                    cells.append((col, row))
        return cells

    def _cluster_cells(self, cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        cell_set = set(cells)
        visited: set[Tuple[int, int]] = set()
        clusters: List[List[Tuple[int, int]]] = []

        for seed in cells:
            if seed in visited:
                continue
            queue = [seed]
            visited.add(seed)
            cluster: List[Tuple[int, int]] = []
            while queue:
                col, row = queue.pop()
                cluster.append((col, row))
                for dcol in range(-self.cluster_connect_distance, self.cluster_connect_distance + 1):
                    for drow in range(-self.cluster_connect_distance, self.cluster_connect_distance + 1):
                        if dcol == 0 and drow == 0:
                            continue
                        neighbor = (col + dcol, row + drow)
                        if neighbor in cell_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            clusters.append(cluster)

        return clusters

    def _nearest_free_cell(
        self,
        data: Sequence[int],
        width: int,
        height: int,
        col: int,
        row: int,
        max_radius: int = 15,
    ) -> Tuple[int, int] | None:
        if 0 <= col < width and 0 <= row < height and self._is_free(data[row * width + col]):
            return (col, row)

        for radius in range(1, max_radius + 1):
            for check_row in range(max(0, row - radius), min(height, row + radius + 1)):
                for check_col in range(max(0, col - radius), min(width, col + radius + 1)):
                    if abs(check_col - col) != radius and abs(check_row - row) != radius:
                        continue
                    if self._is_free(data[check_row * width + check_col]):
                        return (check_col, check_row)
        return None

    def _distance_transform(
        self,
        data: Sequence[int],
        width: int,
        height: int,
        predicate,
    ) -> List[int]:
        dist = [-1] * (width * height)
        queue: Deque[Tuple[int, int]] = deque()
        for row in range(height):
            base = row * width
            for col in range(width):
                if predicate(data[base + col]):
                    dist[base + col] = 0
                    queue.append((col, row))

        while queue:
            col, row = queue.popleft()
            d = dist[row * width + col]
            for dcol, drow in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_col = col + dcol
                next_row = row + drow
                if not (0 <= next_col < width and 0 <= next_row < height):
                    continue
                idx = next_row * width + next_col
                if dist[idx] != -1:
                    continue
                dist[idx] = d + 1
                queue.append((next_col, next_row))
        return dist

    def _label_unknown_regions(
        self,
        data: Sequence[int],
        width: int,
        height: int,
        cap_cells: int,
    ) -> Tuple[List[int], List[int], List[Tuple[int, int, int, int]]]:
        region_id = [-1] * (width * height)
        region_size: List[int] = []
        region_bbox: List[Tuple[int, int, int, int]] = []
        cap = max(1, cap_cells)

        for start_row in range(height):
            base = start_row * width
            for start_col in range(width):
                start_idx = base + start_col
                if region_id[start_idx] != -1 or not self._is_unknown(data[start_idx]):
                    continue

                rid = len(region_size)
                count = 0
                min_col = max_col = start_col
                min_row = max_row = start_row
                queue: Deque[Tuple[int, int]] = deque([(start_col, start_row)])
                region_id[start_idx] = rid
                while queue:
                    col, row = queue.popleft()
                    count += 1
                    if col < min_col:
                        min_col = col
                    if col > max_col:
                        max_col = col
                    if row < min_row:
                        min_row = row
                    if row > max_row:
                        max_row = row
                    if count >= cap:
                        continue
                    for drow in (-1, 0, 1):
                        for dcol in (-1, 0, 1):
                            if drow == 0 and dcol == 0:
                                continue
                            ncol, nrow = col + dcol, row + drow
                            if not (0 <= ncol < width and 0 <= nrow < height):
                                continue
                            nidx = nrow * width + ncol
                            if region_id[nidx] != -1 or not self._is_unknown(data[nidx]):
                                continue
                            region_id[nidx] = rid
                            queue.append((ncol, nrow))
                region_size.append(min(count, cap))
                region_bbox.append((min_col, min_row, max_col, max_row))

        return region_id, region_size, region_bbox

    def _update_region_stall_history(
        self,
        region_size: List[int],
        region_bbox: List[Tuple[int, int, int, int]],
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> List[float]:
        self._region_call_counter += 1
        call = self._region_call_counter
        bucket_m = max(1e-3, self.region_stall_bucket_m)

        discount = [1.0] * len(region_size)
        seen_keys: set[Tuple[int, int]] = set()

        for rid, size in enumerate(region_size):
            min_col, min_row, max_col, max_row = region_bbox[rid]
            center_x = origin_x + 0.5 * (min_col + max_col + 1) * resolution
            center_y = origin_y + 0.5 * (min_row + max_row + 1) * resolution
            key = (
                int(math.floor(center_x / bucket_m)),
                int(math.floor(center_y / bucket_m)),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            entry = self._region_history.get(key)
            if entry is None or size < entry.size * self.region_stall_shrink_ratio:
                self._region_history[key] = _RegionHistoryEntry(
                    size=size, stall_count=0, last_seen_call=call
                )
                continue

            stall_count = entry.stall_count + 1
            self._region_history[key] = _RegionHistoryEntry(
                size=min(size, entry.size), stall_count=stall_count, last_seen_call=call
            )

            if stall_count < self.region_stall_min_calls:
                continue
            if stall_count >= self.region_stall_zero_calls:
                discount[rid] = 0.0
                continue
            span = max(1, self.region_stall_zero_calls - self.region_stall_min_calls)
            progress = (stall_count - self.region_stall_min_calls) / span
            discount[rid] = 1.0 - progress * (1.0 - self.region_stall_discount)

        self._prune_region_history(call)
        return discount

    def _prune_region_history(self, current_call: int) -> None:
        stale_before = current_call - self.region_history_stale_calls
        if stale_before > 0:
            stale_keys = [
                key
                for key, entry in self._region_history.items()
                if entry.last_seen_call < stale_before
            ]
            for key in stale_keys:
                del self._region_history[key]

        overflow = len(self._region_history) - self.region_history_max_entries
        if overflow > 0:
            oldest = sorted(
                self._region_history.items(), key=lambda item: item[1].last_seen_call
            )[:overflow]
            for key, _ in oldest:
                del self._region_history[key]

    def _compute_narrow_cells(
        self,
        width: int,
        height: int,
        occupied_dist: List[int],
        clearance_cells: int,
    ) -> set[Tuple[int, int]]:
        if clearance_cells <= 0:
            return set()

        narrow: set[Tuple[int, int]] = set()
        for row in range(height):
            base = row * width
            for col in range(width):
                d = occupied_dist[base + col]
                if d == -1 or d >= clearance_cells:
                    continue
                narrow.add((col, row))
        return narrow

    def _reachable_free_cells(
        self,
        data: Sequence[int],
        width: int,
        height: int,
        start_cell: Tuple[int, int],
        narrow_cells: set[Tuple[int, int]],
    ) -> set[Tuple[int, int]]:
        visited = {start_cell}
        queue = deque([start_cell])
        while queue:
            col, row = queue.popleft()
            for dcol, drow in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_col = col + dcol
                next_row = row + drow
                if not (0 <= next_col < width and 0 <= next_row < height):
                    continue
                neighbor = (next_col, next_row)
                if neighbor in visited:
                    continue
                if not self._is_free(data[next_row * width + next_col]):
                    continue
                if neighbor in narrow_cells:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return visited

    def _select_goal_cell(
        self,
        anchor_cell: Tuple[int, int],
        width: int,
        height: int,
        resolution: float,
        occupied_dist: List[int],
        unknown_dist: List[int],
        reachable_cells: set[Tuple[int, int]],
        robot_cell: Tuple[int, int],
    ) -> Tuple[int, int] | None:
        anchor_col, anchor_row = anchor_cell
        safety_cells = max(1, int(math.ceil(self.safety_radius_m / resolution)))
        search_radius = max(safety_cells + 15, int(math.ceil(3.5 / resolution)))

        best_cell: Tuple[int, int] | None = None
        best_score = float("inf")
        best_cell_relaxed: Tuple[int, int] | None = None
        best_score_relaxed = float("inf")
        for row in range(max(0, anchor_row - search_radius), min(height, anchor_row + search_radius + 1)):
            for col in range(max(0, anchor_col - search_radius), min(width, anchor_col + search_radius + 1)):
                cell = (col, row)
                if cell not in reachable_cells:
                    continue
                if not self._is_goal_safe(
                    col, row, width, height, occupied_dist, unknown_dist, safety_cells
                ):
                    continue

                frontier_distance = math.hypot(col - anchor_col, row - anchor_row)
                robot_distance = math.hypot(col - robot_cell[0], row - robot_cell[1])
                score = frontier_distance + 0.02 * robot_distance

                if self._is_goal_safe(
                    col, row, width, height, occupied_dist, unknown_dist, safety_cells, True
                ):
                    if score < best_score:
                        best_score = score
                        best_cell = cell
                elif score < best_score_relaxed:
                    best_score_relaxed = score
                    best_cell_relaxed = cell

        return best_cell if best_cell is not None else best_cell_relaxed

    @staticmethod
    def _is_goal_safe(
        col: int,
        row: int,
        width: int,
        height: int,
        occupied_dist: List[int],
        unknown_dist: List[int],
        safety_cells: int,
        require_known: bool = False,
    ) -> bool:
        if col < safety_cells or row < safety_cells:
            return False
        if col >= width - safety_cells or row >= height - safety_cells:
            return False

        idx = row * width + col
        occ_d = occupied_dist[idx]
        if occ_d != -1 and occ_d < safety_cells:
            return False
        if require_known:
            unk_d = unknown_dist[idx]
            if unk_d != -1 and unk_d < safety_cells:
                return False
        return True

    def _is_free(self, value: int) -> bool:
        return 0 <= int(value) <= min(self.free_threshold, self.occupied_threshold - 1)

    @staticmethod
    def _is_unknown(value: int) -> bool:
        return int(value) in (UNKNOWN, UNKNOWN_UINT)

    def _is_occupied(self, value: int) -> bool:
        return int(value) >= self.occupied_threshold

    @staticmethod
    def _near_rejected(
        x: float,
        y: float,
        rejected_goals: Sequence[Tuple[float, float]],
        radius_m: float,
    ) -> bool:
        return any(math.hypot(x - rx, y - ry) <= radius_m for rx, ry in rejected_goals)

    def find_approach_standoffs(
        self,
        target_x: float,
        target_y: float,
        robot_x: float,
        robot_y: float,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        data: Sequence[int],
        standoff_m: float = 0.45,
        rejected_goals: Iterable[Tuple[float, float]] | None = None,
        rejected_radius_m: float = 0.30,
        num_angles: int = 12,
        allow_unknown: bool = False,
        prefer_known: bool = True,
        max_candidates: int = 8,
    ) -> List[ApproachCandidate]:
        if not data or width <= 0 or height <= 0 or resolution <= 0 or standoff_m <= 0:
            return []

        rejected = list(rejected_goals or [])
        preferred = math.atan2(target_y - robot_y, target_x - robot_x)
        angle_step = 2.0 * math.pi / max(num_angles, 1)
        ordered_angles = [
            preferred + index * angle_step for index in range(max(num_angles, 1))
        ]

        candidates: List[ApproachCandidate] = []
        for angle in ordered_angles:
            goal_x = target_x - standoff_m * math.cos(angle)
            goal_y = target_y - standoff_m * math.sin(angle)
            if self._near_rejected(goal_x, goal_y, rejected, rejected_radius_m):
                continue

            safety_ok, safety_score, known = self._score_standoff_safety(
                goal_x,
                goal_y,
                width,
                height,
                resolution,
                origin_x,
                origin_y,
                data,
                allow_unknown=allow_unknown,
                prefer_known=prefer_known,
            )
            if not safety_ok:
                continue

            face_yaw = math.atan2(target_y - goal_y, target_x - goal_x)
            alignment = abs(_normalize_angle(face_yaw - angle))
            robot_distance = math.hypot(goal_x - robot_x, goal_y - robot_y)
            score = (
                safety_score
                + 0.35 * alignment
                + 0.08 * robot_distance
                + (0.0 if known else 0.25)
            )
            candidates.append(
                ApproachCandidate(
                    x=goal_x,
                    y=goal_y,
                    yaw=face_yaw,
                    score=score,
                    standoff_m=standoff_m,
                    angle_rad=angle,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score)
        return candidates[:max_candidates]

    def _score_standoff_safety(
        self,
        goal_x: float,
        goal_y: float,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        data: Sequence[int],
        allow_unknown: bool,
        prefer_known: bool,
    ) -> Tuple[bool, float, bool]:
        col, row = world_to_cell(goal_x, goal_y, resolution, origin_x, origin_y)
        if not (0 <= col < width and 0 <= row < height):
            return False, 999.0, False

        center_value = int(data[row * width + col])
        if self._is_occupied(center_value):
            return False, 999.0, False
        center_known = not self._is_unknown(center_value)
        if not center_known and not allow_unknown:
            return False, 999.0, False

        radius_cells = max(1, int(math.ceil(self.safety_radius_m / resolution)))
        unknown_count = 0
        occupied_count = 0
        checked = 0
        for drow in range(-radius_cells, radius_cells + 1):
            for dcol in range(-radius_cells, radius_cells + 1):
                if dcol * dcol + drow * drow > radius_cells * radius_cells:
                    continue
                check_col = col + dcol
                check_row = row + drow
                if not (0 <= check_col < width and 0 <= check_row < height):
                    occupied_count += 1
                    continue
                value = int(data[check_row * width + check_col])
                checked += 1
                if self._is_occupied(value):
                    occupied_count += 1
                elif self._is_unknown(value):
                    unknown_count += 1

        if occupied_count > 0:
            return False, 999.0, center_known
        if unknown_count > 0 and not allow_unknown:
            return False, 999.0, False

        unknown_ratio = unknown_count / max(checked, 1)
        score = unknown_ratio + (0.15 if prefer_known and not center_known else 0.0)
        return True, score, center_known and unknown_count == 0


def cell_to_world(
    col: float,
    row: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> Tuple[float, float]:
    return origin_x + col * resolution, origin_y + row * resolution


def world_to_cell(
    x: float,
    y: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> Tuple[int, int]:
    return int((x - origin_x) / resolution), int((y - origin_y) / resolution)


def side_sector_clearances(
    ranges, angle_min: float, angle_increment: float
) -> Tuple[float, float]:
    """Return minimum left/right side ranges in the robot frame."""
    left = float("inf")
    right = float("inf")
    for index, value in enumerate(ranges):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            continue
        angle = angle_min + index * angle_increment
        if math.pi / 4.0 <= angle <= 3.0 * math.pi / 4.0:
            left = min(left, value)
        elif -3.0 * math.pi / 4.0 <= angle <= -math.pi / 4.0:
            right = min(right, value)
    return left, right


def choose_lateral_escape_direction(
    left_clearance: float,
    right_clearance: float,
    wall_trigger_m: float,
    destination_clearance_m: float,
) -> float:
    """Choose a scan-confirmed clear side for a bounded lateral escape.

    Prefer moving away from a close side wall.  If the lethal start cell is
    caused by an obstacle in front/behind (so both side sectors are clear),
    choose the side with more clearance.  Never move toward a side whose
    clearance is below ``destination_clearance_m``.
    """
    if left_clearance <= wall_trigger_m and right_clearance >= destination_clearance_m:
        return -1.0
    if right_clearance <= wall_trigger_m and left_clearance >= destination_clearance_m:
        return 1.0
    left_safe = left_clearance >= destination_clearance_m
    right_safe = right_clearance >= destination_clearance_m
    if left_safe and not right_safe:
        return 1.0
    if right_safe and not left_safe:
        return -1.0
    if left_safe and right_safe:
        # Deterministic tie-break keeps recovery behavior reproducible.
        return 1.0 if left_clearance >= right_clearance else -1.0
    return 0.0


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
