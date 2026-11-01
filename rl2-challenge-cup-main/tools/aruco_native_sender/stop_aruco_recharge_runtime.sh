#!/usr/bin/env bash
set -euo pipefail

if pgrep -x aruco_recharge >/dev/null; then
  pkill -INT -x aruco_recharge
  echo "Sent SIGINT to aruco_recharge."
else
  echo "aruco_recharge is not running."
fi

cat <<'EOF'
The temporary eth1 interface was intentionally retained.
To remove it after the process has stopped:
  sudo ip link delete eth1
Use the remote controller immediately if the robot has not stopped safely.
EOF
