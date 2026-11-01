#!/usr/bin/env bash
# Read-only readiness check for the real-robot evidence collection scripts.
set -euo pipefail
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "${HOME}/learning/unitree_ros2/setup_go2w_humble.sh"
source "${PROJECT_ROOT}/install/setup.bash"
set -u
fail() { echo "[FAIL] $*" >&2; exit 1; }

echo "=== RL2 measurement preflight (read-only) ==="
ip -br addr show enp59s0 | grep -q '192\.168\.123\.' || fail "robot network missing"
echo "[PASS] robot network"
for topic_spec in \
  '/slam/odom nav_msgs/msg/Odometry' \
  '/scan sensor_msgs/msg/LaserScan' \
  '/lf/sportmodestate unitree_go/msg/SportModeState'; do
  read -r topic type <<<"${topic_spec}"
  timeout 4s ros2 topic echo "${topic}" "${type}" \
    --once --no-daemon --spin-time 3 >/dev/null 2>&1 \
    || fail "no fresh sample on ${topic}"
  echo "[PASS] ${topic} is live"
done
for node in controller_server planner_server bt_navigator; do
  state="$(ros2 lifecycle get "/${node}" --no-daemon 2>/dev/null || true)"
  [[ "${state}" == *'active [3]'* ]] || fail "/${node} is not active: ${state}"
  echo "[PASS] /${node} active"
done
guard="$({ timeout 4s ros2 topic echo \
  /rl2_nav/slam_pose_jump_guard/status std_msgs/msg/String \
  --field data --once --no-daemon --spin-time 3; } 2>/dev/null || true)"
[[ "${guard}" == *'"state":"MONITORING"'* ]] || fail "SLAM guard not MONITORING"
[[ "${guard}" == *'"pose_stream_lost":false'* ]] || fail "SLAM pose stream lost"
echo "[PASS] SLAM jump guard healthy"
services="$(ros2 service list --no-daemon 2>/dev/null)"
for service in \
  /cmd_vel_safety_node/enable \
  /go2w_sport_cmd_bridge/enable \
  /go2w_sport_cmd_bridge/stop_move \
  /frontier_explorer_node/enable; do
  grep -Fxq "${service}" <<<"${services}" || fail "missing service ${service}"
  echo "[PASS] ${service}"
done
nav_info="$(ros2 action info /navigate_to_pose 2>/dev/null || true)"
spin_info="$(ros2 action info /spin 2>/dev/null || true)"
[[ "${nav_info}" == *'Action servers: 1'* ]] || fail "/navigate_to_pose unavailable"
[[ "${spin_info}" == *'Action servers: 1'* ]] || fail "/spin unavailable"
echo "[PASS] Nav2 action servers"
echo "[PASS] software readiness verified; no motion gate was changed"
echo "Human confirmation is still required for unlock, clearance, remote and emergency stop."
