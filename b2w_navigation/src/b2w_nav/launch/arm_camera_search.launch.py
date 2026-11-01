"""Raised-arm navigation; the separate wrist command owns search and optional approach."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,IncludeLaunchDescription,LogInfo,OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def reject_existing_stack(context):
    import time
    import rclpy
    from rclpy.context import Context
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    ros_context=Context()
    rclpy.init(context=ros_context,signal_handler_options=SignalHandlerOptions.NO)
    node=Node('wrist_search_stack_check',context=ros_context)
    executor=SingleThreadedExecutor(context=ros_context);executor.add_node(node)
    try:
        end=time.monotonic()+1.0
        while time.monotonic()<end:executor.spin_once(timeout_sec=.1)
        conflicts=set(node.get_node_names()) & {'controller_server','b2w_sport_cmd_bridge','frontier_explorer_node'}
        if conflicts:
            raise RuntimeError('Existing navigation stack detected: '+', '.join(sorted(conflicts))+
                               '. Stop it in its original terminal before launching the raised-arm search stack.')
    finally:
        executor.shutdown();node.destroy_node();rclpy.shutdown(context=ros_context)
    return []


def generate_launch_description():
    share=FindPackageShare('b2w_nav')
    return LaunchDescription([
        DeclareLaunchArgument('start_rviz',default_value='true'),
        OpaqueFunction(function=reject_existing_stack),
        LogInfo(msg='Raised-arm search/optional approach: enlarged footprint, 0.10m/s. Mission supervisor and body-spin search are OFF. Start search_with_wrist.py separately.'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([
            share,'launch','autonomous_mapping_sport_bringup.launch.py'])),launch_arguments={
                'sensor_backend':'builtin','start_rviz':LaunchConfiguration('start_rviz'),
                'start_perception':'false','start_supervisor':'false','mission_auto_start':'false',
                'frontier_max_goals':'3','frontier_auto_wall_escape':'false',
                'nav_params_file':PathJoinSubstitution([share,'config','nav2_arm_search.yaml']),
                'robot_params_file':PathJoinSubstitution([share,'config','b2w_arm_search.yaml']),
            }.items()),
    ])
