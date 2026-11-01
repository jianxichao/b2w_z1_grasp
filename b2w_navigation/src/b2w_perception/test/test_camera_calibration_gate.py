from types import SimpleNamespace
from sensor_msgs.msg import Image
from b2w_perception.yolo_target_perception_node import YoloTargetPerceptionNode


def test_unmeasured_camera_can_detect_but_cannot_publish_navigation_coordinates():
    node = object.__new__(YoloTargetPerceptionNode)
    params = dict(distance_source='depth', input_timeout_sec=1.0,
                  rgb_depth_sync_tolerance_sec=.20,
                  rgb_depth_sync_use_receive_time=False,
                  camera_extrinsics_verified=False)
    node.get_parameter = lambda key: SimpleNamespace(value=params[key])
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    for k in ('rgb','camera_info','odom','depth'):
        setattr(node, k+'_received_ns', 9_900_000_000)
    node.rgb_msg = Image()
    node.depth_msg = Image()
    node.last_processed_rgb_stamp = None
    node.seen_streak, node.lost_streak = 0,0
    node.seen_confirm_frames = 1
    node.confidence_threshold = .65
    node.max_confirm_distance_m = node.max_distance_m = 3.0
    node.min_distance_m, node.target_class = .15,'can'
    node._detect_yolo_target = lambda: (320.,240.,.90,40.,60.)
    node._distance_from_depth = lambda *args: 1.5
    node._publish_annotated = lambda *args: None
    statuses = []
    node._publish_status = lambda *args, **kwargs: statuses.append((args,kwargs))
    def forbidden(*args):
        raise AssertionError('Uncalibrated camera must not create or publish map coordinates')
    node._estimate_pose_from_depth = forbidden
    node.pose_pub = node.candidate_pub = node.event_pub = SimpleNamespace(publish=forbidden)
    node._tick()
    assert statuses[-1][0][0] is True
    assert statuses[-1][1]['candidate_x'] is None
    assert statuses[-1][1]['candidate_y'] is None
