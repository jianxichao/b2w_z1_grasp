#!/usr/bin/env bash
# Convenient PC-side entry point for the verified 0/90-degree storage-box positions.
set -euo pipefail
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-}"
confirm="${2:-}"
case "${action}" in
  probe) service=/storage_box/probe ;;
  open|90) service=/storage_box/open ;;
  close|0) service=/storage_box/close ;;
  *)
    echo "Usage: $0 probe | {open|90|close|0} --confirm-motion" >&2
    exit 2
    ;;
esac
if [[ "${action}" != probe && "${confirm}" != "--confirm-motion" ]]; then
  echo "REFUSED: open/close moves the servo; append --confirm-motion after checking clearance." >&2
  exit 3
fi
set +u
source "${HOME}/learning/unitree_ros2/setup_go2w_humble.sh"
source "${PROJECT_ROOT}/install/setup.bash"
set -u
if ! ros2 service list --no-daemon 2>/dev/null | grep -Fxq "${service}"; then
  "${PROJECT_ROOT}/scripts/start_storage_box_bridge.sh"
  for _ in $(seq 1 30); do
    ros2 service list --no-daemon 2>/dev/null | grep -Fxq "${service}" && break
    sleep 0.2
  done
fi
ros2 service list --no-daemon 2>/dev/null | grep -Fxq "${service}" \
  || { echo "storage-box ROS service unavailable" >&2; exit 4; }
ros2 service call "${service}" std_srvs/srv/Trigger '{}'
