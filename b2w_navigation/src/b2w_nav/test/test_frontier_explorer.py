import math
import unittest

from b2w_nav.frontier_explorer import (
    FrontierCandidate,
    FrontierExplorer,
    choose_lateral_escape_direction,
    side_sector_clearances,
)


class LateralEscapeTest(unittest.TestCase):
    def test_side_sector_clearances(self):
        ranges = [float("inf")] * 181  # 180deg, 1deg increments
        ranges[180] = 1.5  # angle +90 deg -> left sector
        ranges[0] = 0.5    # angle -90 deg -> right sector
        left, right = side_sector_clearances(ranges, -math.pi / 2.0, math.pi / 180.0)
        self.assertEqual(left, 1.5)
        self.assertEqual(right, 0.5)

    def test_choose_direction_prefers_away_from_wall(self):
        # Wall on the right -> move left (+1, y>0 is left on B2W).
        self.assertEqual(choose_lateral_escape_direction(2.0, 0.3, 0.55, 0.8), 1.0)
        # Wall on the left -> move right (-1).
        self.assertEqual(choose_lateral_escape_direction(0.3, 2.0, 0.55, 0.8), -1.0)
        # Both sides blocked -> no escape.
        self.assertEqual(choose_lateral_escape_direction(0.3, 0.3, 0.55, 0.8), 0.0)


class FrontierExplorerTest(unittest.TestCase):
    def test_frontier_size_reward_saturates_for_large_wall_boundaries(self):
        explorer = FrontierExplorer(size_weight=0.08, size_reward_cap=3.0)

        self.assertAlmostEqual(explorer._frontier_size_reward(20), 1.6)
        self.assertAlmostEqual(explorer._frontier_size_reward(100), 3.0)
        self.assertAlmostEqual(explorer._frontier_size_reward(1000), 3.0)

    def test_failed_frontier_is_not_reintroduced_by_relaxed_fallback(self):
        width = 30
        height = 20
        resolution = 0.1
        data = [0] * (width * height)
        for row in range(6, 14):
            for col in range(20, 28):
                data[row * width + col] = -1

        explorer = FrontierExplorer(
            min_frontier_size=1,
            safety_radius_m=0.1,
            min_frontier_distance_m=0.0,
        )
        initial = explorer.find_frontiers(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.5,
            robot_y=1.0,
        )
        self.assertTrue(initial)

        retried = explorer.find_frontiers(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.5,
            robot_y=1.0,
            rejected_goals=[(candidate.x, candidate.y) for candidate in initial],
            rejected_radius_m=100.0,
        )
        self.assertEqual(retried, [])

    def test_global_costmap_filter_rejects_disconnected_candidate(self):
        width = 12
        height = 8
        data = [0] * (width * height)
        # A lethal wall separates the robot from the second candidate.
        for row in range(height):
            data[row * width + 6] = 100
        candidates = [
            FrontierCandidate(x=0.35, y=0.35, size=10, distance=0.3, score=2.0),
            FrontierCandidate(x=0.95, y=0.35, size=20, distance=0.9, score=3.0),
        ]
        explorer = FrontierExplorer()
        filtered = explorer.filter_reachable_on_costmap(
            candidates=candidates,
            width=width,
            height=height,
            resolution=0.1,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.1,
            robot_y=0.35,
            traversable_cost_max=65,
            goal_tolerance_m=0.0,
        )
        self.assertEqual(filtered, [candidates[0]])

    def test_global_costmap_filter_accepts_candidate_through_gap(self):
        width = 12
        height = 8
        data = [0] * (width * height)
        for row in range(height):
            if row != 3:
                data[row * width + 6] = 100
        candidate = FrontierCandidate(
            x=0.95, y=0.35, size=20, distance=0.9, score=3.0
        )
        explorer = FrontierExplorer()
        filtered = explorer.filter_reachable_on_costmap(
            candidates=[candidate],
            width=width,
            height=height,
            resolution=0.1,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.1,
            robot_y=0.35,
            traversable_cost_max=65,
            goal_tolerance_m=0.0,
        )
        self.assertEqual(filtered, [candidate])

    def test_global_costmap_filter_rejects_when_exact_robot_cell_is_lethal(self):
        width = 10
        height = 10
        data = [0] * (width * height)
        data[2 * width + 2] = 100
        candidate = FrontierCandidate(
            x=0.7, y=0.7, size=10, distance=0.7, score=1.0
        )
        explorer = FrontierExplorer()
        filtered = explorer.filter_reachable_on_costmap(
            candidates=[candidate],
            width=width,
            height=height,
            resolution=0.1,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.25,
            robot_y=0.25,
            traversable_cost_max=65,
            goal_tolerance_m=0.0,
        )
        self.assertEqual(filtered, [])

    def test_global_costmap_filter_snaps_goal_to_connected_free_cell(self):
        width = 12
        height = 8
        resolution = 0.1
        data = [0] * (width * height)
        # Candidate cell is lethal, while the cell immediately to its left is
        # free and connected to the robot.
        data[3 * width + 8] = 100
        candidate = FrontierCandidate(
            x=0.85, y=0.35, size=12, distance=0.75, score=2.0
        )
        explorer = FrontierExplorer()
        filtered = explorer.filter_reachable_on_costmap(
            candidates=[candidate],
            width=width,
            height=height,
            resolution=resolution,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.15,
            robot_y=0.35,
            traversable_cost_max=98,
            goal_tolerance_m=0.2,
        )
        self.assertEqual(len(filtered), 1)
        self.assertNotAlmostEqual(filtered[0].x, candidate.x)
        snapped_col, snapped_row = (
            int(filtered[0].x / resolution),
            int(filtered[0].y / resolution),
        )
        self.assertLessEqual(data[snapped_row * width + snapped_col], 98)

    def test_global_costmap_filter_rejects_goal_without_wall_clearance(self):
        width = 20
        height = 12
        data = [0] * (width * height)
        # Candidate center is free but a lethal wall is only 0.2m away.
        for row in range(height):
            data[row * width + 12] = 100
        candidate = FrontierCandidate(
            x=1.05, y=0.55, size=20, distance=0.8, score=3.0
        )
        explorer = FrontierExplorer()
        filtered = explorer.filter_reachable_on_costmap(
            candidates=[candidate],
            width=width,
            height=height,
            resolution=0.1,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.25,
            robot_y=0.55,
            traversable_cost_max=98,
            goal_tolerance_m=0.0,
            goal_clearance_m=0.35,
            goal_clearance_cost_max=65,
        )
        self.assertEqual(filtered, [])

    def test_prefers_nearer_frontier_when_information_gain_is_equal(self):
        width = 40
        height = 20
        resolution = 0.1
        data = [0] * (width * height)

        for row in range(height):
            data[row * width] = 100
            data[row * width + width - 1] = 100
        for col in range(width):
            data[col] = 100
            data[(height - 1) * width + col] = 100

        # Two same-sized unknown regions create equivalent frontier gains at
        # different distances from the robot.
        for row in range(7, 13):
            for col in range(10, 13):
                data[row * width + col] = -1
            for col in range(29, 32):
                data[row * width + col] = -1

        explorer = FrontierExplorer(
            min_frontier_size=1,
            safety_radius_m=0.1,
            min_frontier_distance_m=0.0,
            info_gain_weight=0.0,
            size_weight=0.0,
            distance_weight=1.0,
        )
        candidates = explorer.find_frontiers(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=0.5,
            robot_y=1.0,
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertLessEqual(candidates[0].distance, candidates[1].distance)

    def test_prefers_frontier_with_larger_unknown_region(self):
        width = 30
        height = 20
        resolution = 0.1
        data = [0] * (width * height)

        for row in range(height):
            data[row * width + 0] = 100
            data[row * width + width - 1] = 100
        for col in range(width):
            data[col] = 100
            data[(height - 1) * width + col] = 100

        for row in range(2, 15):
            for col in range(20, 28):
                data[row * width + col] = -1
        for row in range(16, 18):
            for col in range(3, 5):
                data[row * width + col] = -1

        explorer = FrontierExplorer(
            min_frontier_size=1,
            safety_radius_m=0.1,
            min_frontier_distance_m=0.0,
            info_gain_weight=3.0,
            distance_weight=0.0,
            size_weight=0.0,
        )
        candidates = explorer.find_frontiers(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=0.0,
            origin_y=0.0,
            data=data,
            robot_x=1.0,
            robot_y=1.0,
        )

        self.assertTrue(candidates)
        self.assertGreater(candidates[0].x, 1.5)


if __name__ == "__main__":
    unittest.main()
