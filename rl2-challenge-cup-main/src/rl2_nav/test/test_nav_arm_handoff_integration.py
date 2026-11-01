"""Isolated ROS integration tests for navigation -> arm -> navigation handoff."""

import time
import unittest

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from rl2_interfaces.action import GraspTarget
from rl2_nav.mission_supervisor_core import MissionState
from rl2_nav.mission_supervisor_node import MissionSupervisorNode


class TestNavigationArmHandoffIntegration(unittest.TestCase):
    def setUp(self):
        rclpy.init(
            args=[
                "--ros-args",
                "-p",
                "arm_integration_enabled:=true",
                "-p",
                "target_posture_enabled:=false",
                "-p",
                "target_seen_hold_s:=60.0",
                "-p",
                "arm_target_wait_timeout_s:=0.5",
                "-p",
                "arm_target_max_age_s:=0.5",
                "-p",
                "arm_action_timeout_s:=1.0",
            ]
        )
        self.events = []
        self.frontier_requests = []
        self.stop_requests = 0
        self.arm_result_mode = "success"
        self.mock = Node("nav_arm_handoff_mock")
        self.publisher = Node("nav_arm_handoff_publisher")
        self.arm_server = ActionServer(
            self.mock,
            GraspTarget,
            "/arm/grasp_target",
            execute_callback=self._execute_grasp,
            cancel_callback=lambda _request: CancelResponse.ACCEPT,
        )
        self.mock.create_service(
            SetBool,
            "/frontier_explorer_node/enable",
            self._set_frontier,
        )
        self.mock.create_service(
            Trigger,
            "/go2w_sport_cmd_bridge/stop_move",
            self._stop_move,
        )
        self.supervisor = MissionSupervisorNode()
        target_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_pub = self.publisher.create_publisher(
            PoseStamped, "/robot_arm/target_pose", target_qos
        )
        self.publisher.create_subscription(
            String,
            "/task_event",
            lambda msg: self.events.append(msg.data),
            10,
        )
        self.executor = MultiThreadedExecutor(num_threads=4)
        for node in (self.mock, self.publisher, self.supervisor):
            self.executor.add_node(node)
        self._spin_until(lambda: self.supervisor._arm_client.server_is_ready(), 2.0)

    def tearDown(self):
        self.arm_server.destroy()
        for node in (self.supervisor, self.publisher, self.mock):
            self.executor.remove_node(node)
            node.destroy_node()
        self.executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()

    def _execute_grasp(self, goal_handle):
        feedback = GraspTarget.Feedback()
        feedback.stage = "PLAN"
        feedback.progress = 0.2
        goal_handle.publish_feedback(feedback)
        if self.arm_result_mode == "timeout":
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not goal_handle.is_cancel_requested:
                time.sleep(0.02)
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            result = GraspTarget.Result()
            result.success = False
            result.message = "mock timeout ended"
            return result
        succeeded = self.arm_result_mode == "success"
        feedback.stage = "DONE" if succeeded else "FAILED"
        feedback.progress = 1.0
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        result = GraspTarget.Result()
        result.success = succeeded
        result.message = "mock grasp completed" if succeeded else "mock failure"
        return result

    def _set_frontier(self, request, response):
        self.frontier_requests.append(bool(request.data))
        response.success = True
        response.message = "ok"
        return response

    def _stop_move(self, _request, response):
        self.stop_requests += 1
        response.success = True
        response.message = "stopped"
        return response

    def _spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def _spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            self.executor.spin_once(timeout_sec=0.02)
        return bool(predicate())

    def _publish_target(self, *, age_s=0.0):
        pose = PoseStamped()
        stamp_ns = self.publisher.get_clock().now().nanoseconds - int(age_s * 1e9)
        pose.header.stamp.sec = stamp_ns // 1_000_000_000
        pose.header.stamp.nanosec = stamp_ns % 1_000_000_000
        pose.header.frame_id = "arm_base"
        pose.pose.position.x = 0.30
        pose.pose.position.y = 0.00
        pose.pose.position.z = 0.20
        pose.pose.orientation.w = 1.0
        self.target_pub.publish(pose)

    def _confirm_navigation_target(self):
        self.supervisor._state = MissionState.VERIFYING_TARGET
        self.supervisor._accept_confirmed_target(
            1.0, 2.0, source="visual", confidence=0.8
        )
        self.assertEqual(self.supervisor._state, MissionState.WAITING_ARM_TARGET)
        self.assertEqual(self.supervisor._reached_target_count, 0)

    def test_stale_pose_is_rejected_then_success_resumes_search(self):
        self._confirm_navigation_target()
        self._spin_for(0.15)
        self._publish_target(age_s=5.0)
        self._spin_for(0.15)
        self.assertEqual(self.supervisor._state, MissionState.WAITING_ARM_TARGET)

        self._publish_target()
        self.assertTrue(
            self._spin_until(
                lambda: self.supervisor._state == MissionState.TARGET_HOLDING
                and any(
                    event.startswith("mission_target_confirmed:1/3")
                    for event in self.events
                ),
                2.0,
            )
        )
        self.assertEqual(self.supervisor._reached_target_count, 1)
        self.assertGreaterEqual(self.stop_requests, 1)
        self.assertIn(False, self.frontier_requests)
        self.assertTrue(
            any(event.startswith("mission_arm_grasp_succeeded:1/3") for event in self.events)
        )

        self.supervisor._hold_until = 0.0
        self.supervisor._control_step()
        self._spin_for(0.10)
        self.assertEqual(self.supervisor._state, MissionState.SEARCHING)
        self.assertIn(True, self.frontier_requests)
        self.assertIn("mission_search_resumed:target_confirmed", self.events)

    def test_large_cross_host_clock_skew_accepts_post_handoff_heartbeat(self):
        self._confirm_navigation_target()
        self._spin_for(0.15)
        self._publish_target(age_s=1000.0)
        self.assertTrue(
            self._spin_until(
                lambda: self.supervisor._state == MissionState.TARGET_HOLDING,
                2.0,
            )
        )
        self.assertEqual(self.supervisor._reached_target_count, 1)
        self.assertTrue(
            any(event.startswith("mission_arm_grasp_succeeded:1/3") for event in self.events)
        )

    def test_failed_grasp_does_not_count_and_resumes_search(self):
        self.arm_result_mode = "failure"
        self._confirm_navigation_target()
        self._spin_for(0.15)
        self._publish_target()
        self.assertTrue(
            self._spin_until(
                lambda: self.supervisor._state == MissionState.SEARCHING,
                2.0,
            )
        )
        self.assertEqual(self.supervisor._reached_target_count, 0)
        self.assertTrue(
            any(event.startswith("mission_arm_grasp_failed:mock failure") for event in self.events)
        )
        self._spin_for(0.10)
        self.assertIn(True, self.frontier_requests)

    def test_missing_fresh_arm_target_times_out_without_counting(self):
        self._confirm_navigation_target()
        self.assertTrue(
            self._spin_until(
                lambda: self.supervisor._state == MissionState.SEARCHING,
                1.5,
            )
        )
        self._spin_for(0.10)
        self.assertEqual(self.supervisor._reached_target_count, 0)
        self.assertIn(
            "mission_arm_grasp_failed:fresh_arm_target_timeout", self.events
        )
        self.assertIn(True, self.frontier_requests)

    def test_arm_action_timeout_stops_in_idle_for_manual_reset(self):
        self.arm_result_mode = "timeout"
        self._confirm_navigation_target()
        self._spin_for(0.15)
        self._publish_target()
        self.assertTrue(
            self._spin_until(
                lambda: self.supervisor._state == MissionState.GRASPING,
                1.0,
            )
        )
        self.assertTrue(
            self._spin_until(
                lambda: self.supervisor._state == MissionState.IDLE,
                2.0,
            )
        )
        self._spin_for(0.20)
        self.assertEqual(self.supervisor._reached_target_count, 0)
        self.assertIn("mission_arm_grasp_failed:action_timeout", self.events)
        self.assertIn("mission_arm_manual_reset_required", self.events)
        self.assertNotEqual(self.frontier_requests[-1], True)
        self.assertGreaterEqual(self.stop_requests, 2)


if __name__ == "__main__":
    unittest.main()
