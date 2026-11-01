#!/usr/bin/env bash
# Arm a single bounded Nav2 yaw probe and always disarm afterward.

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly OUTPUT_DIR="${PROJECT_ROOT}/logs/angular_alignment"
readonly STAMP="$(date +%Y%m%d_%H%M%S)"
readonly TEST_ANGLE_DEG="${1:-90}"
readonly OUTPUT_CSV="${2:-${OUTPUT_DIR}/yaw_${TEST_ANGLE_DEG}deg_${STAMP}.csv}"

set +u
source "${HOME}/learning/unitree_ros2/setup_go2w_humble.sh"
source "${PROJECT_ROOT}/install/setup.bash"
set -u

cleanup() {
  "${PROJECT_ROOT}/scripts/stop_all_motion.sh" || true
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "${OUTPUT_CSV}")"
"${PROJECT_ROOT}/scripts/measurement_preflight.sh"

ros2 service call /frontier_explorer_node/enable \
  std_srvs/srv/SetBool '{data: false}' >/dev/null
ros2 service call /go2w_sport_cmd_bridge/enable \
  std_srvs/srv/SetBool '{data: true}'
ros2 service call /cmd_vel_safety_node/enable \
  std_srvs/srv/SetBool '{data: true}'

"${PROJECT_ROOT}/scripts/angular_alignment_probe.py" \
  --mode spin \
  --relative-yaw-deg "${TEST_ANGLE_DEG}" \
  --timeout-s 20 \
  --confirm-motion \
  --output "${OUTPUT_CSV}"
