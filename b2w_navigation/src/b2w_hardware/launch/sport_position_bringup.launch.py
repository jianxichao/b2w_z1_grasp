"""Sport odometry and mutually exclusive velocity/position bridge; no Nav2 or SLAM."""
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def reject_existing(context):
    import time
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    c=Context();rclpy.init(context=c,signal_handler_options=SignalHandlerOptions.NO);n=rclpy.create_node('sport_position_stack_check',context=c)
    e=SingleThreadedExecutor(context=c);e.add_node(n)
    try:
        end=time.monotonic()+1.
        while time.monotonic()<end:e.spin_once(timeout_sec=.05)
        if set(n.get_node_names())&{'b2w_sport_cmd_bridge','controller_server','b2w_sport_odom'}:
            raise RuntimeError('An existing navigation/Sport bridge is running; stop its launch before starting this standalone stack')
    finally:e.shutdown();n.destroy_node();rclpy.shutdown(context=c)
    return []


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=reject_existing),
        Node(package='b2w_hardware',executable='sport_odom_node',output='screen'),
        Node(package='b2w_hardware',executable='b2w_sport_cmd_bridge',output='screen',
             parameters=[PathJoinSubstitution([FindPackageShare('b2w_hardware'),'config','b2w_sport_bridge.yaml'])])])
