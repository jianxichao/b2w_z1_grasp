#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly UNITREE_SETUP="${UNITREE_SETUP:-/home/kazuhalry/learning/unitree_ros2/setup_go2w_humble.sh}"

set +u
source "$UNITREE_SETUP"
source "$PROJECT_ROOT/install/setup.bash"
set -u

dds_interface="$(ip -o -4 addr show up | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')"
if [[ -z "$dds_interface" ]]; then
  echo "ERROR: no UP 192.168.123.x interface" >&2
  exit 1
fi
dds_config="/tmp/rl2_cyclonedds_${USER}.xml"
cat >"$dds_config" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$dds_interface" priority="default" multicast="default"/>
      </Interfaces>
      <AllowMulticast>spdp</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
EOF
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export CYCLONEDDS_URI="file://$dds_config"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/rl2_three_target_box}"
mkdir -p "$ROS_LOG_DIR"

exec python3 "$PROJECT_ROOT/scripts/three_target_box_mission.py"
