#!/usr/bin/env bash
set -eo pipefail

map_path="${1:-/home/unitree/rl2_site_map_v1.pcd}"
echo "Saving the active Unitree map to: ${map_path}"
ros2 service call /unitree_slam/end_mapping \
  rl2_interfaces/srv/UnitreeEndMapping \
  "{address: '${map_path}'}"
echo "Request queued. Verify api_id=1802, succeed=true and 'Save pcd successfully':"
timeout 15s ros2 topic echo /unitree_slam/command_status \
  std_msgs/msg/String --no-daemon --spin-time 3 || true

