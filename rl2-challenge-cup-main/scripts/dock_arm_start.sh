#!/usr/bin/env bash
set -euo pipefail

ROOT="${DOCK_ARM_ROOT:-/home/unitree/rl2_arm_runtime}"
REQUESTED_MODE="${1:-dry-run}"
REQUESTED_VISION="${2:-no-vision}"
case "$REQUESTED_MODE" in
  dry-run|real) ;;
  *) echo "usage: $0 [dry-run|real]" >&2; exit 2 ;;
esac
case "$REQUESTED_VISION" in
  vision|no-vision) ;;
  *) echo "usage: $0 [dry-run|real] [vision|no-vision]" >&2; exit 2 ;;
esac

set +u
source /opt/ros/foxy/setup.bash
source "$ROOT/install/setup.bash"
set -u

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
# Foxy's bundled libddsc crashes on this dock.  The dock-provided CycloneDDS
# workspace is ABI-compatible with rmw_cyclonedds_cpp and is already used by
# the Unitree services.  Keep both its library and XML paired together.
export LD_LIBRARY_PATH="/home/unitree/cyclonedds_ws/install/cyclonedds/lib:${LD_LIBRARY_PATH:-}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///home/unitree/cyclonedds_ws/cyclonedds.xml}"

if [[ "$REQUESTED_MODE" == real ]]; then
  ip link show "${ARX_CAN_INTERFACE:-can1}" >/dev/null 2>&1 || {
    echo "CAN is not ready; run scripts/dock_arm_setup_can.sh first" >&2
    exit 3
  }
fi

exec ros2 launch rl2_manipulation arm.launch.py \
  mode:="$REQUESTED_MODE" \
  robot_config_root:="$ROOT/src/robot_arm" \
  start_vision:="$([[ "$REQUESTED_VISION" == vision ]] && echo true || echo false)"
