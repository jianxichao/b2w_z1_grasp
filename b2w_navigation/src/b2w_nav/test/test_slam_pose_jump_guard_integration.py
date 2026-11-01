"""Isolated ROS integration test for the complete jump-recovery fan-out."""

import time
import unittest

import rclpy
from action_msgs.srv import CancelGoal
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger

from b2w_nav.slam_pose_jump_guard_node import (
    READY_FOR_MANUAL_REARM,
    SlamPoseJumpGuard,
)


class TestSlamPoseJumpGuardIntegration(unittest.TestCase):
    def setUp(self):
        rclpy.init(
            args=[
                "--ros-args",
                "-p",
                "stable_duration_s:=0.20",
                "-p",
                "minimum_known_map_cells:=3",
            ]
        )
        self.calls = set()
        self.mock = Node("jump_guard_mock_services")
        self.publisher = Node("jump_guard_mock_publishers")
        self.guard = SlamPoseJumpGuard()
        self.executor = MultiThreadedExecutor(num_threads=4)
        for node in (self.mock, self.publisher, self.guard):
            self.executor.add_node(node)

        self.mock.create_service(
            SetBool,
            "/frontier_explorer_node/enable",
            self._set_bool_callback("disable_frontier"),
        )
        self.mock.create_service(
            SetBool,
            "/cmd_vel_safety_node/enable",
            self._set_bool_callback("disable_velocity_gate"),
        )
        self.mock.create_service(
            SetBool,
            "/b2w_sport_cmd_bridge/enable",
            self._set_bool_callback("disable_sport_output"),
        )
        self.mock.create_service(
            CancelGoal,
            "/navigate_to_pose/_action/cancel_goal",
            self._cancel_callback,
        )
        self.mock.create_service(
            Trigger,
            "/official_occupancy_grid_bridge/reset",
            self._trigger_callback,
        )
        self.mock.create_service(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
            self._clear_global_callback,
        )
        self.mock.create_service(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
            self._clear_local_callback,
        )
        self.odom_pub = self.publisher.create_publisher(Odometry, "/slam/odom", 10)
        self.map_pub = self.publisher.create_publisher(
            OccupancyGrid, "/navigation_map", 10
        )

    def tearDown(self):
        self._spin_for(0.20)
        # Finish executor callbacks before destroying the ROS entities they use.
        self.executor.shutdown()
        for node in (self.guard, self.publisher, self.mock):
            self.executor.remove_node(node)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _set_bool_callback(self, name):
        def callback(request, response):
            self.assertFalse(request.data)
            self.calls.add(name)
            response.success = True
            return response

        return callback

    def _cancel_callback(self, _request, response):
        self.calls.add("cancel_nav2_goals")
        response.return_code = CancelGoal.Response.ERROR_NONE
        return response

    def _trigger_callback(self, _request, response):
        self.calls.add("reset_occupancy_grid")
        response.success = True
        return response

    def _clear_global_callback(self, _request, response):
        self.calls.add("clear_global_costmap")
        return response

    def _clear_local_callback(self, _request, response):
        self.calls.add("clear_local_costmap")
        return response

    def _spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def _publish_odom(self, x, y=0.0):
        msg = Odometry()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        self.odom_pub.publish(msg)

    def test_jump_runs_every_stop_and_clear_operation_then_latches(self):
        self._spin_for(0.20)
        self._publish_odom(0.0)
        self._spin_for(0.10)
        self._publish_odom(2.0)
        self._spin_for(0.30)

        grid = OccupancyGrid()
        grid.data = [0, 0, 100, -1]
        self.map_pub.publish(grid)
        for _ in range(8):
            self._publish_odom(2.0)
            self._spin_for(0.05)

        self.assertEqual(self.guard._state, READY_FOR_MANUAL_REARM)
        self.assertEqual(
            self.calls,
            {
                "disable_frontier",
                "disable_velocity_gate",
                "disable_sport_output",
                "cancel_nav2_goals",
                "reset_occupancy_grid",
                "clear_global_costmap",
                "clear_local_costmap",
            },
        )

    def test_repeated_samples_from_one_jump_only_start_one_recovery(self):
        self._spin_for(0.20)
        self._publish_odom(0.0)
        self._spin_for(0.05)
        self._publish_odom(2.0)
        self._spin_for(0.05)
        generation = self.guard._recovery_generation
        self._publish_odom(7.0)
        self._spin_for(0.05)

        self.assertEqual(self.guard._recovery_generation, generation)
        self.assertEqual(self.guard._jump_count, 1)


if __name__ == "__main__":
    unittest.main()
