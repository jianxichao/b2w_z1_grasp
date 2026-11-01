#!/usr/bin/env bash
set -eo pipefail

map_path="${1:-/home/unitree/rl2_site_map_v1.pcd}"
base_x="${2:--0.9018463769314599}"
base_y="${3:-3.1729447911619566}"
base_yaw="${4:--1.4567911021181614}"
read -r qz qw < <(python3 -c \
  'import math,sys; y=float(sys.argv[1]); print(math.sin(y/2), math.cos(y/2))' \
  "$base_yaw")

# Recorded base pose in the persistent map. This initializes localization only;
# it does not command robot motion.
ros2 service call /unitree_slam/init_pose \
  rl2_interfaces/srv/UnitreeInitPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: ${base_x}, y: ${base_y}, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: ${qz}, w: ${qw}}}}, address: '${map_path}'}"

echo "Relocation request queued. Do not enable motion until verification passes."
