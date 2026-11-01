import math
import unittest

from b2w_nav.frontier_explorer import FrontierExplorer, world_to_cell


class ApproachStandoffTest(unittest.TestCase):
    def test_rejects_occupied_standoff_and_returns_safe_candidate(self):
        width = 40
        height = 40
        resolution = 0.1
        origin_x = -2.0
        origin_y = -2.0
        data = [0] * (width * height)
        target_x = 0.0
        target_y = 0.0

        occupied_x = -0.5
        occupied_y = 0.0
        col, row = world_to_cell(occupied_x, occupied_y, resolution, origin_x, origin_y)
        data[row * width + col] = 100

        explorer = FrontierExplorer(safety_radius_m=0.15)
        candidates = explorer.find_approach_standoffs(
            target_x=target_x,
            target_y=target_y,
            robot_x=-1.0,
            robot_y=0.0,
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            data=data,
            standoff_m=0.5,
            num_angles=8,
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertGreater(math.hypot(candidate.x - occupied_x, candidate.y - occupied_y), 0.14)

    def test_unknown_standoff_rejected_by_default(self):
        width = 20
        height = 20
        resolution = 0.1
        data = [-1] * (width * height)
        explorer = FrontierExplorer(safety_radius_m=0.15)
        candidates = explorer.find_approach_standoffs(
            target_x=0.0,
            target_y=0.0,
            robot_x=-1.0,
            robot_y=0.0,
            width=width,
            height=height,
            resolution=resolution,
            origin_x=-1.0,
            origin_y=-1.0,
            data=data,
            standoff_m=0.5,
        )
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
