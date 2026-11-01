"""Regression tests for cross-target pose smoothing contamination."""

import math

from geometry_msgs.msg import PoseStamped

from b2w_perception.yolo_target_perception_node import YoloTargetPerceptionNode


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs


def _pose(x: float, y: float) -> PoseStamped:
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


def _node() -> YoloTargetPerceptionNode:
    node = object.__new__(YoloTargetPerceptionNode)
    node.pose_smoothing_alpha = 0.35
    node.pose_smoothing_reset_distance_m = 0.80
    node.smoothed_target_xy = None
    node.get_logger = lambda: _Logger()
    return node


def test_nearby_samples_keep_exponential_smoothing() -> None:
    node = _node()
    node._smooth_pose(_pose(1.0, 2.0))

    result = node._smooth_pose(_pose(1.2, 2.0))

    assert math.isclose(result.pose.position.x, 1.07, abs_tol=1e-9)
    assert node.smoothed_target_xy == (1.07, 2.0)


def test_large_map_jump_resets_instead_of_blending_two_targets() -> None:
    node = _node()
    node._smooth_pose(_pose(1.0, 2.0))

    result = node._smooth_pose(_pose(3.0, 2.5))

    assert result.pose.position.x == 3.0
    assert result.pose.position.y == 2.5
    assert node.smoothed_target_xy == (3.0, 2.5)
