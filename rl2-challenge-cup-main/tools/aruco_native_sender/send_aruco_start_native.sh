#!/usr/bin/env bash
set -euo pipefail

sender_dir="/home/unitree/aruco_native_sender"
runtime_iface="eth1"

if [[ "${1:-}" != "--confirm-motion" ]]; then
  echo "REFUSED: this command can move the robot." >&2
  echo "After checking the pile, marker, clearance and remote control, run:" >&2
  echo "  $0 --confirm-motion" >&2
  exit 2
fi

ip link show "$runtime_iface" >/dev/null
if ! pgrep -x aruco_recharge >/dev/null; then
  echo "ERROR: aruco_recharge is not running." >&2
  exit 3
fi

cd "$sender_dir"
exec env LD_LIBRARY_PATH="$sender_dir/lib" \
  ./aruco_native_sender "$runtime_iface"
