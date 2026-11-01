#!/usr/bin/env python3
"""Run on a ROS 2 robot computer; read-only sensor graph and fresh-data inventory."""
import json
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import String

rclpy.init()
node = rclpy.create_node('b2w_readonly_sensor_inventory')
received, subscriptions = {}, []
watched = set()
end = time.monotonic()+10.0
try:
    while time.monotonic() < end:
        for topic, kinds in node.get_topic_names_and_types():
            if topic in watched:
                continue
            kind = {'nav_msgs/msg/Odometry': Odometry,
                    'sensor_msgs/msg/PointCloud2': PointCloud2,
                    'sensor_msgs/msg/Image': Image,
                    'sensor_msgs/msg/CameraInfo': CameraInfo}.get(kinds[0])
            if kind is None:
                continue
            watched.add(topic)
            def receive(msg, topic=topic):
                info = received.setdefault(topic, {'count':0})
                info['count'] += 1
                info['frame_id'] = msg.header.frame_id
                info['stamp'] = [msg.header.stamp.sec, msg.header.stamp.nanosec]
                if hasattr(msg, 'width'):
                    info['size'] = [msg.width, msg.height]
                if isinstance(msg, Image):
                    info['encoding'] = msg.encoding
                if isinstance(msg, CameraInfo):
                    info['K'] = list(msg.k)
                if isinstance(msg, Odometry):
                    info['child_frame_id'] = msg.child_frame_id
                    info['position'] = [msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z]
                if isinstance(msg, PointCloud2):
                    info['fields'] = [f.name for f in msg.fields]
            subscriptions.append(node.create_subscription(kind, topic, receive, qos_profile_sensor_data))
        rclpy.spin_once(node, timeout_sec=0.1)
    print(json.dumps({'fresh_data':received,'discovered_topics':dict(node.get_topic_names_and_types())},indent=2))
finally:
    node.destroy_node()
    rclpy.shutdown()
