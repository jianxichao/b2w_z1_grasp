#!/usr/bin/env bash

# ROS/colcon setup files legitimately read unset tracing variables. Source
# every overlay with nounset disabled, then enable strict mode for this script.
set +u

# Run this script on the Unitree expansion dock (192.168.123.18).
# The dock's custom CycloneDDS build is required in the busy Unitree domain.
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
source /home/unitree/realsense2_ros2_ws/install/setup.bash
set -euo pipefail

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ros2 launch realsense2_camera rs_launch.py \
  camera_name:=d435i \
  serial_no:="'261622077826'" \
  depth_module.profile:=640,480,15 \
  rgb_camera.profile:=424,240,15 \
  enable_depth:=true \
  enable_color:=true \
  enable_infra1:=false \
  enable_infra2:=false \
  enable_gyro:=false \
  enable_accel:=false \
  enable_sync:=true \
  pointcloud.enable:=false \
  tf_publish_rate:=0.0 &
launch_pid=$!

cleanup() {
  kill -INT "$launch_pid" 2>/dev/null || true
  wait "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wrapper 4.0.4 calls the launch option align_depth.enable, but its actual
# runtime parameter is align.enable. Set that parameter once the node exists.
aligned=false
for _ in $(seq 1 30); do
  if ros2 param list /d435i/d435i 2>/dev/null | grep -q 'align.enable'; then
    ros2 param set /d435i/d435i align.enable true >/dev/null
    if ros2 param get /d435i/d435i align.enable 2>/dev/null | grep -q 'True'; then
      aligned=true
      break
    fi
  fi
  sleep 1
done

if [[ "$aligned" != true ]]; then
  echo "ERROR: D435i did not expose align.enable within 30 seconds." >&2
  exit 1
fi

echo "D435i ROS 2 is running: color, depth and aligned depth are enabled."
wait "$launch_pid"
