#!/usr/bin/env bash
set -euo pipefail
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly UNIT_FILE="${PROJECT_ROOT}/舵机交接/rl2-storage-box-bridge.service"
readonly UNITREE_SETUP="${UNITREE_SETUP:-${HOME}/learning/unitree_ros2/setup_go2w_humble.sh}"
readonly USER_UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
readonly INSTALLED_UNIT="${USER_UNIT_DIR}/rl2-storage-box-bridge.service"

if [[ ! -f "${UNITREE_SETUP}" ]]; then
  echo "Unitree ROS setup not found: ${UNITREE_SETUP}" >&2
  exit 1
fi

mkdir -p "${USER_UNIT_DIR}"
sed \
  -e "s|@PROJECT_ROOT@|${PROJECT_ROOT}|g" \
  -e "s|@UNITREE_SETUP@|${UNITREE_SETUP}|g" \
  "${UNIT_FILE}" >"${INSTALLED_UNIT}"
systemctl --user daemon-reload
# Restart even when a bridge process survived a robot/network power cycle.
# CycloneDDS binds its interface when the process starts; merely leaving the
# old user service running can preserve a stale DDS participant that exposes
# services locally but cannot communicate with the dock after reconnection.
# Restarting this ROS/SSH bridge never sends a servo command.
systemctl --user restart rl2-storage-box-bridge.service
systemctl --user --no-pager --full status rl2-storage-box-bridge.service | sed -n '1,12p'
