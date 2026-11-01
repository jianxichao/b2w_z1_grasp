#!/usr/bin/env bash
# One bounded 0.20m NavigateToPose test with a 45-degree final heading change.

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly OUTPUT_DIR="${PROJECT_ROOT}/logs/angular_alignment"
readonly STAMP="$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT_CSV="${1:-${OUTPUT_DIR}/near_goal_020m_yaw45_${STAMP}.csv}"

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
  --mode navigate \
  --forward-m 0.20 \
  --relative-yaw-deg 45 \
  --timeout-s 25 \
  --confirm-motion \
  --output "${OUTPUT_CSV}"
