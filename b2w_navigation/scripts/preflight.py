#!/usr/bin/env python3
"""Read-only dependency/configuration/topic check. Publishes no robot requests."""
import argparse
from importlib.util import find_spec
from pathlib import Path
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def check(live=False, target=False, motion=False):
    errors = []
    from ament_index_python.packages import get_package_prefix, PackageNotFoundError
    for pkg in ('b2w_interfaces', 'b2w_hardware', 'b2w_nav', 'b2w_perception',
                'unitree_api', 'unitree_go', 'nav2_bringup', 'nav2_smac_planner',
                'pointcloud_to_laserscan', 'rmw_cyclonedds_cpp', 'slam_toolbox'):
        try:
            get_package_prefix(pkg)
        except PackageNotFoundError:
            errors.append('Missing ROS package: ' + pkg)
    for name in ('ultralytics', 'torch', 'cv2', 'numpy'):
        if find_spec(name) is None:
            errors.append('Missing Python module: ' + name)
    if not (ROOT.parent / 'sponge_best.pt').is_file():
        errors.append('Model not found: sponge_best.pt')
    config = yaml.safe_load((ROOT/'src/b2w_nav/config/b2w_robot.yaml').read_text())
    flags = config['b2w_preflight']['ros__parameters']
    camera = config['yolo_target_perception_node']['ros__parameters']
    if motion and not flags['geometry_verified']:
        errors.append('Configure geometry from measured or accepted official values; geometry_verified is false')
    if motion and not config['b2w_lidar_mount']['ros__parameters']['verified']:
        errors.append('Configure base_link -> rslidar; b2w_lidar_mount.verified is false')
    if target and not (flags['camera_extrinsics_verified'] and camera['camera_extrinsics_verified']):
        errors.append('Confirm chassis camera optical transform and depth intrinsics; camera_extrinsics_verified is false')
    iface = flags['network_interface']
    if not Path('/sys/class/net', iface).exists():
        errors.append('Network interface does not exist: ' + iface)
    if live and not errors:
        errors += check_topics(target)
    for message in errors:
        print('FAIL:', message)
    if not errors:
        print('PASS: dependencies/configuration' + (' and fresh topic data' if live else ''))
    if not motion:
        print('Geometry configured:', flags['geometry_verified'], '| source:', flags.get('geometry_source', 'local'), '| camera optical calibration:', camera['camera_extrinsics_verified'])
    return not errors


def check_topics(target):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2, LaserScan, Image, CameraInfo
    from nav_msgs.msg import Odometry, OccupancyGrid
    from unitree_go.msg import SportModeState
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = rclpy.create_node('b2w_readonly_preflight')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    topics = {
        '/rslidar_points': PointCloud2,
        '/b2w/sport/odom': Odometry,
        '/slam/odom': Odometry,
        '/scan': LaserScan,
        '/navigation_map': OccupancyGrid,
        '/lf/sportmodestate': SportModeState,
    }
    if target:
        topics.update({'/b2w/front/color/image_raw': Image,
                       '/b2w/front/aligned_depth_to_color/image_raw': Image,
                       '/b2w/front/color/camera_info': CameraInfo})
    latest = {}
    subscriptions = []
    for topic, kind in topics.items():
        def receive(msg, topic=topic):
            latest[topic] = time.monotonic()
        subscriptions.append(node.create_subscription(kind, topic, receive, qos_profile_sensor_data))
    end = time.monotonic() + 5.0
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()
        errors = ['No fresh data: '+t for t in topics if now-latest.get(t, 0)>1.0]
        if not buffer.can_transform('map', 'base_footprint', rclpy.time.Time()):
            errors.append('TF unavailable: map -> base_footprint')
        from collections import Counter
        counts=Counter(node.get_node_names())
        for critical in ('controller_server','b2w_sport_cmd_bridge','frontier_explorer_node','b2w_map_pose'):
            if counts[critical]>1:errors.append('Duplicate navigation node: '+critical)
        for name in node.get_node_names():
            if name == 'go2w_sport_cmd_bridge':
                errors.append('Original Go2W Sport bridge is also running')
        return errors
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--motion', action='store_true')
    args = parser.parse_args()
    sys.exit(0 if check(**vars(args)) else 1)
