#!/usr/bin/env bash
set -eo pipefail

echo "=== API 1804 status (expect succeed:true) ==="
timeout 8s ros2 topic echo /unitree_slam/command_status \
  std_msgs/msg/String --no-daemon --spin-time 3 || true

echo "=== relocation odometry frequency (expect stable ~20 Hz) ==="
timeout 8s ros2 topic hz /unitree/slam_relocation/odom --wall-time || true

echo "=== saved global map (expect one non-empty PointCloud2) ==="
timeout 8s ros2 topic echo /unitree/slam_relocation/global_map \
  sensor_msgs/msg/PointCloud2 --once --no-daemon --spin-time 3 --no-arr || true

echo "=== current map pose ==="
timeout 8s ros2 topic echo /slam/current_pose geometry_msgs/msg/PoseStamped \
  --once --no-daemon --spin-time 3 || true

echo "=== map -> base_footprint TF ==="
timeout 5s ros2 run tf2_ros tf2_echo map base_footprint || true

echo "Verification is not complete until RViz alignment is checked while wheels are locked."

