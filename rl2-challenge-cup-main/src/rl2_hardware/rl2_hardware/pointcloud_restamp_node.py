"""Restamp official dock point clouds into the user-PC ROS clock domain."""

from sensor_msgs.msg import PointCloud2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class PointCloudRestampNode(Node):
    """Bridge a PointCloud2 without changing its points or coordinate frame.

    The Unitree dock and user PC do not necessarily share wall-clock time. TF2
    therefore rejects the otherwise valid mapping cloud as being far outside
    its transform cache. The receive time is the only timestamp available in
    the PC clock domain, so use it for local Nav2 processing.
    """

    def __init__(self) -> None:
        super().__init__("pointcloud_restamp_node")
        self.declare_parameter("input_topic", "/unitree/slam_mapping/points")
        self.declare_parameter("output_topic", "/unitree/slam_mapping/points_synced")
        self.declare_parameter("frame_id", "map")

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_topic").value),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Restamping official mapping cloud into the user-PC clock domain."
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        if self._frame_id:
            msg.header.frame_id = self._frame_id
        self._publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PointCloudRestampNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
