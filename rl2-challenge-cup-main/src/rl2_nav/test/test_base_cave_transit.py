import unittest

from nav_msgs.msg import OccupancyGrid

from rl2_nav.base_cave_transit_node import BaseCaveTransitNode


class _Parameter:
    def __init__(self, value):
        self.value = value


def _node_without_ros(**parameters):
    node = BaseCaveTransitNode.__new__(BaseCaveTransitNode)
    values = {
        "traversable_cost_max": 98,
        "max_plan_candidates": 40,
    }
    values.update(parameters)
    node.get_parameter = lambda name: _Parameter(values[name])
    return node


def _grid(width=10, height=10, resolution=0.1):
    grid = OccupancyGrid()
    grid.info.width = width
    grid.info.height = height
    grid.info.resolution = resolution
    grid.info.origin.position.x = -0.5
    grid.info.origin.position.y = -0.5
    grid.data = [0] * (width * height)
    return grid


class BaseCaveGridCorrectionTest(unittest.TestCase):
    def test_world_cell_round_trip_uses_cell_center(self):
        grid = _grid()
        cell = BaseCaveTransitNode._cell_for_xy(grid, 0.02, -0.08)
        self.assertEqual(cell, (5, 4))
        x, y = BaseCaveTransitNode._xy_for_cell(grid, *cell)
        self.assertAlmostEqual(x, 0.05)
        self.assertAlmostEqual(y, -0.05)

    def test_outside_grid_is_rejected(self):
        grid = _grid()
        self.assertIsNone(BaseCaveTransitNode._cell_for_xy(grid, 2.0, 0.0))

    def test_target_snaps_off_inscribed_cell(self):
        grid = _grid()
        center = BaseCaveTransitNode._cell_for_xy(grid, 0.05, 0.05)
        grid.data[center[1] * grid.info.width + center[0]] = 99
        node = _node_without_ros()
        candidates = node._snap_candidates(grid, (0.05, 0.05, 1.2), 0.25)
        self.assertTrue(candidates)
        self.assertNotEqual(candidates[0][:2], (0.05, 0.05))
        self.assertAlmostEqual(candidates[0][2], 1.2)

    def test_unknown_is_not_selected_as_goal(self):
        grid = _grid()
        grid.data = [-1] * (grid.info.width * grid.info.height)
        node = _node_without_ros()
        self.assertEqual(node._snap_candidates(grid, (0.0, 0.0, 0.0), 0.4), [])


if __name__ == "__main__":
    unittest.main()
