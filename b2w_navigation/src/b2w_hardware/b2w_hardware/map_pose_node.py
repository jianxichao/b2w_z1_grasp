"""Adapt fresh SLAM TF and sport odom to the original mission's map pose contract."""
import copy
import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener, TransformException


class MapPose(Node):
    def __init__(self):
        super().__init__('b2w_map_pose')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.odom, self.received = None, 0.0
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/current_pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/slam/odom', 10)
        self.create_subscription(Odometry, '/b2w/sport/odom', self.receive, qos_profile_sensor_data)
        self.create_timer(0.05, self.update)

    def receive(self, msg):
        self.odom, self.received = msg, time.monotonic()

    def update(self):
        if self.odom is None or time.monotonic()-self.received > 0.5:
            return
        try:
            # Query at odometry exposure time: a stale map->odom cannot masquerade
            # as a fresh localized pose. Permit only a small TF interpolation wait.
            tf = self.buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time.from_msg(self.odom.header.stamp), timeout=Duration(seconds=0.02))
        except TransformException:
            return
        msg = copy.deepcopy(self.odom)
        msg.header.frame_id = 'map'
        p = tf.transform.translation
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = p.x, p.y, p.z
        msg.pose.pose.orientation = tf.transform.rotation
        self.odom_pub.publish(msg)
        pose = PoseStamped(header=msg.header, pose=msg.pose.pose)
        self.pose_pub.publish(pose)


def main():
    rclpy.init()
    node = MapPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
