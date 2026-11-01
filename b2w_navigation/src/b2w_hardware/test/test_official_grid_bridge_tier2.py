"""Tier-2 bridge upgrades (路径规划分析与优化.md 第二梯队 A2/C5).

- obstacle_confirm_hits: a single noisy cloud no longer marks a cell lethal;
  two distinct clouds are required (1 restores the old instant behaviour).
- Free-space evidence wipes hit credit, so a cleared cell must re-earn
  lethal status from scratch.
- min_clearing_period_s rate-limits the raycast clearing.
- _publish_map serializes via array('b', tobytes()) instead of tolist().
"""

from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from nav_msgs.msg import OccupancyGrid

from b2w_hardware.official_occupancy_grid_bridge import (
    OfficialOccupancyGridBridge,
)


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


class _Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _make_bridge(params=None, grid_values=None):
    node = object.__new__(OfficialOccupancyGridBridge)
    node._width = 20
    node._height = 20
    node._resolution = 0.05
    node._origin_x = -0.5
    node._origin_y = -0.5
    node._map_frame = "map"
    node._grid = np.full((20, 20), -1, dtype=np.int8)
    if grid_values is not None:
        for (row, col), value in grid_values.items():
            node._grid[row, col] = value
    node._hits = np.zeros((20, 20), dtype=np.int8)
    values = {
        "obstacle_confirm_hits": 2,
        "min_clearing_period_s": 0.25,
        "max_range_m": 8.0,
        "max_clearing_rays": 2500,
        "min_height_m": -0.30,
        "max_height_m": 1.20,
        "clear_robot_footprint": False,
    }
    values.update(params or {})
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = lambda: _Logger()
    return node


# ---------------------------------------------------------------------------
# A2: hit-counted obstacle marking
# ---------------------------------------------------------------------------


def test_first_hit_does_not_mark_lethal() -> None:
    node = _make_bridge()

    node._mark_obstacle_cells(np.array([5]), np.array([5]))

    assert node._grid[5, 5] == -1
    assert node._hits[5, 5] == 1


def test_second_cloud_confirms_lethal() -> None:
    node = _make_bridge()

    node._mark_obstacle_cells(np.array([5]), np.array([5]))
    node._mark_obstacle_cells(np.array([5]), np.array([5]))

    assert node._grid[5, 5] == 100


def test_duplicate_points_in_one_cloud_count_once() -> None:
    node = _make_bridge()

    # Ten returns on the same cell within a single cloud must not confirm it.
    node._mark_obstacle_cells(np.full(10, 5), np.full(10, 5))

    assert node._hits[5, 5] == 1
    assert node._grid[5, 5] == -1


def test_confirm_hits_one_restores_instant_marking() -> None:
    node = _make_bridge(params={"obstacle_confirm_hits": 1})

    node._mark_obstacle_cells(np.array([5]), np.array([5]))

    assert node._grid[5, 5] == 100


def test_free_cell_wipes_hit_credit() -> None:
    node = _make_bridge()

    node._mark_obstacle_cells(np.array([5]), np.array([5]))
    assert node._hits[5, 5] == 1
    # Raycast/ground evidence later clears the cell.
    node._grid[5, 5] = 0
    node._hits[node._grid == 0] = 0

    node._mark_obstacle_cells(np.array([5]), np.array([5]))

    # The stale hit is gone: this is a fresh first hit, still unconfirmed.
    assert node._hits[5, 5] == 1
    assert node._grid[5, 5] == 0


# ---------------------------------------------------------------------------
# min_clearing_period_s: raycast clearing is rate-limited
# ---------------------------------------------------------------------------


def _make_bridge_for_clearing(now_ns_box):
    node = _make_bridge()
    node._last_clearing_apply_s = 0.0
    node._latest_pose = SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
    )
    node._robot_cell = lambda: (10, 10)
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=now_ns_box[0])
    )
    raycasts = []
    node._raycast_free = lambda robot_cell, cols, rows: raycasts.append(len(cols))
    node._clear_current_robot_footprint = lambda: None
    node._raycasts = raycasts
    return node


def test_clearing_rate_limited_within_period() -> None:
    now_ns_box = [100_000_000_000]
    node = _make_bridge_for_clearing(now_ns_box)
    xyz = np.array([[0.1, 0.1, 0.0], [0.2, 0.1, 0.5]])

    node._apply_live_clearing_xyz(xyz)
    assert len(node._raycasts) == 1

    # 0.05 s later (< 0.25 s): skipped entirely.
    now_ns_box[0] += 50_000_000
    node._apply_live_clearing_xyz(xyz)
    assert len(node._raycasts) == 1


def test_clearing_runs_again_after_period() -> None:
    now_ns_box = [100_000_000_000]
    node = _make_bridge_for_clearing(now_ns_box)
    xyz = np.array([[0.1, 0.1, 0.0]])

    node._apply_live_clearing_xyz(xyz)
    now_ns_box[0] += 300_000_000  # 0.30 s > 0.25 s
    node._apply_live_clearing_xyz(xyz)

    assert len(node._raycasts) == 2


# ---------------------------------------------------------------------------
# C5: fast grid serialization
# ---------------------------------------------------------------------------


def test_publish_map_serializes_all_values() -> None:
    node = _make_bridge(
        grid_values={(0, 0): 100, (5, 5): 0, (19, 19): -1}
    )
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMsg())
    )
    node._map_pub = _Recorder()

    node._publish_map()

    msg = node._map_pub.messages[-1]
    assert isinstance(msg, OccupancyGrid)
    assert msg.info.width == 20
    assert msg.info.height == 20
    assert len(msg.data) == 400
    flat = node._grid.reshape(-1)
    assert list(msg.data) == flat.tolist()
    assert msg.data[0] == 100
    assert msg.data[5 * 20 + 5] == 0
    assert msg.data[-1] == -1
