"""Build a new 2-D map from native B2W lidar and read-only sport odometry."""
import math
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, GroupAction
from launch_ros.actions import SetRemap
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('b2w_nav'))
    cfg = yaml.safe_load((share/'config/b2w_robot.yaml').read_text())
    mount = cfg['b2w_lidar_mount']['ros__parameters']
    actions = [
        LogInfo(msg='B2W native lidar mapping: no SLAM API 1801/1102 is called; velocity gates remain closed.'),
        Node(package='b2w_hardware', executable='sport_odom_node', output='screen'),
        Node(package='b2w_hardware', executable='pointcloud_restamp_node',
             name='b2w_native_cloud_clock_bridge', output='screen',
             parameters=[{'input_topic':'/rslidar_points','output_topic':'/b2w/lidar/points',
                          'frame_id':''}]),
        Node(package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
             name='b2w_lidar_to_scan', output='screen',
             remappings=[('cloud_in','/b2w/lidar/points'),('scan','/scan')],
             parameters=[{'target_frame':'base_footprint','transform_tolerance':0.10,
                          'min_height':0.10,'max_height':1.50,'range_min':0.25,
                          'range_max':8.0,'angle_min':-math.pi,'angle_max':math.pi,
                          'angle_increment':0.00873,'scan_time':0.1,'use_inf':True}]),
        GroupAction([
            SetRemap(src='/map', dst='/navigation_map'),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(
                str(Path(get_package_share_directory('slam_toolbox'))/'launch/online_async_launch.py')),
                launch_arguments={'slam_params_file':str(share/'config/slam_toolbox.yaml'),
                                  'use_sim_time':'false'}.items()),
        ]),
        Node(package='b2w_hardware', executable='map_pose_node', output='screen'),
    ]
    for key in ('b2w_lidar_mount', 'b2w_front_depth_mount', 'b2w_rear_depth_mount'):
        mount = cfg[key]['ros__parameters']
        if mount['verified']:
            actions.append(Node(package='tf2_ros', executable='static_transform_publisher',
                name=key+'_tf', arguments=[
                    '--x',str(mount['x']), '--y',str(mount['y']), '--z',str(mount['z']),
                    '--roll',str(mount['roll']), '--pitch',str(mount['pitch']), '--yaw',str(mount['yaw']),
                    '--frame-id',str(mount['parent_frame']), '--child-frame-id',str(mount['frame_id'])]))
        else:
            actions.append(LogInfo(msg=key+' is not configured; no transform is published.'))
    return LaunchDescription(actions)
