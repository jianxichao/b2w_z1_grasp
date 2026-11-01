#!/usr/bin/env bash
# Read-only audit for Unitree's independent official ArUco recharge stack.
# Usage: ./scripts/recharge_preflight.sh [unitree@192.168.123.18]
set -euo pipefail

dock_host="${1:-unitree@192.168.123.18}"

echo '=== Local network ==='
ip -br addr | grep -E 'enp|eth|enx' || true
ping -c 2 -W 1 "${dock_host#*@}" || true

echo '=== Dock: official ArUco recharge program ==='
ssh "$dock_host" 'bash -s' <<'REMOTE'
set -euo pipefail
echo '--- processes ---'
pgrep -af 'aruco_recharge|aruco_charge|unitree_slam|mid360_driver|realsense' || true

echo '--- locating official executable (read-only) ---'
mapfile -t candidates < <(find /unitree /home/unitree -maxdepth 5 -type f \
  \( -name aruco_recharge -o -name aruco_charge \) -perm -111 -print 2>/dev/null)
if [ "${#candidates[@]}" -eq 0 ]; then
  echo 'MISSING: no executable named aruco_recharge or aruco_charge was found.'
  echo 'Install/copy the official ArUco recharge package supplied for this dock SDK.'
else
  for exe in "${candidates[@]}"; do
    echo "EXECUTABLE: $exe"
    file "$exe" || true
    echo 'dynamic libraries:'
    ldd "$exe" | grep -E 'not found|ddsc|ddscxx|unitree_sdk2' || true
  done
fi

echo '--- relevant DDS topics (Foxy command, no publishing) ---'
if command -v ros2 >/dev/null 2>&1; then
  ros2 topic list 2>/dev/null | grep -E '^/(aruco_cmd|aruco_state|frontvideostream)$' || true
fi
REMOTE

cat <<'EOF'
=== Interpretation ===
PASS requires all of the following before the official aruco_recharge program
is started:
1. Unitree's compatible charging pile is powered, stable, and has its official
   ArUco marker installed (do not substitute an arbitrary printed marker).
2. The robot-head camera stream /frontvideostream is healthy, and its
   intrinsics/extrinsics match the recharge package configuration.
3. The dock contains a runnable aruco_recharge/aruco_charge executable with
   no 'not found' dependency reported by ldd.
4. /aruco_cmd and /aruco_state appear after the official program starts.
5. No rl_sar, Nav2, Sport bridge, or other motion commander is running.

This script does not publish to DDS and cannot move the robot.
EOF
