"""Read B2W SportModeState into odom; never publishes robot control requests."""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from unitree_go.msg import SportModeState


class SportOdom(Node):
    def __init__(self):
        super().__init__('b2w_sport_odom')
        self.declare_parameter('state_topic', '/lf/sportmodestate')
        self.origin = None
        self.last_stamp = None
        self.last_received = None
        self.pub = self.create_publisher(Odometry, '/b2w/sport/odom', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(SportModeState, str(self.get_parameter('state_topic').value), self.receive, qos_profile_sensor_data)
        self.create_timer(1.0, self.watchdog)

    def watchdog(self):
        if self.last_received is None or time.monotonic()-self.last_received > 0.5:
            self.get_logger().warning('Sport odometry missing/stale; not publishing TF or odom')

    def receive(self, state):
        # Unitree IMU quaternion is [w,x,y,z]. Read only, reject invalid frames.
        q = [float(x) for x in state.imu_state.quaternion]
        x, y = (float(v) for v in state.position[:2])
        values = [x, y, *q, *state.velocity, state.yaw_speed]
        norm = math.sqrt(sum(v*v for v in q))
        if state.error_code or not all(math.isfinite(v) for v in values) or not 0.9 <= norm <= 1.1:
            return
        # A cached state repeated by DDS is not fresh odometry.
        stamp = (int(state.stamp.sec), int(state.stamp.nanosec))
        if stamp != (0, 0) and stamp == self.last_stamp:
            return
        self.last_stamp = stamp
        self.last_received = time.monotonic()
        w, qx, qy, qz = [v/norm for v in q]
        yaw = math.atan2(2*(w*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        if self.origin is None:
            self.origin = (x,y,yaw)
        ox, oy, oyaw = self.origin
        c, s = math.cos(oyaw), math.sin(oyaw)
        dx, dy = x-ox, y-oy
        heading = math.remainder(yaw-oyaw, 2*math.pi)
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id, msg.child_frame_id = 'odom', 'base_footprint'
        msg.pose.pose.position.x, msg.pose.pose.position.y = c*dx+s*dy, -s*dx+c*dy
        msg.pose.pose.orientation.z = math.sin(heading/2)
        msg.pose.pose.orientation.w = math.cos(heading/2)
        # Unitree sport velocity is represented in its world frame. Rotate into base.
        vx, vy = (float(v) for v in state.velocity[:2])
        msg.twist.twist.linear.x = math.cos(yaw)*vx + math.sin(yaw)*vy
        msg.twist.twist.linear.y = -math.sin(yaw)*vx + math.cos(yaw)*vy
        msg.twist.twist.angular.z = float(state.yaw_speed)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.1
        self.pub.publish(msg)
        tf = TransformStamped()
        tf.header, tf.child_frame_id = msg.header, msg.child_frame_id
        tf.transform.translation.x, tf.transform.translation.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        tf.transform.rotation = msg.pose.pose.orientation
        self.tf.sendTransform(tf)
        # URDF mounts are relative to body base_link, not the ground footprint.
        # B2W firmware currently reports body_height=0; position.z supplies the
        # flat-floor body height. Do not substitute the overall robot height.
        height = float(state.body_height)
        if not 0.05 < height < 1.5:
            height = float(state.position[2])
        if math.isfinite(height) and 0.05 < height < 1.5:
            body = TransformStamped()
            body.header.stamp = msg.header.stamp
            body.header.frame_id, body.child_frame_id = 'base_footprint', 'base_link'
            body.transform.translation.z = height
            # R_body_relative_footprint = Rz(-absolute_yaw) * R_imu.
            cy, sy = math.cos(yaw/2), math.sin(yaw/2)
            body.transform.rotation.w = cy*w + sy*qz
            body.transform.rotation.x = cy*qx + sy*qy
            body.transform.rotation.y = cy*qy - sy*qx
            body.transform.rotation.z = cy*qz - sy*w
            self.tf.sendTransform(body)



def main():
    rclpy.init()
    node = SportOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
